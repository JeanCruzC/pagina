import os
import sys
import types
import json
from contextlib import contextmanager

import pytest
from flask import template_rendered

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())
sys.modules.setdefault('website.utils.kpis_core', types.SimpleNamespace())

from website import create_app
from website.utils import allowlist as allowlist_module

app = create_app()
add_to_allowlist = allowlist_module.add_to_allowlist


@contextmanager
def captured_templates(app):
    recorded = []
    def record(sender, template, context, **extra):
        recorded.append((template, context))
    template_rendered.connect(record, app)
    try:
        yield recorded
    finally:
        template_rendered.disconnect(record, app)


def _csrf_token(client, path):
    resp = client.get(path)
    html = resp.get_data(as_text=True)
    import re
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    return match.group(1) if match else None


@pytest.fixture(autouse=True)
def temp_allowlist(tmp_path):
    allowlist_module.ALLOWLIST_FILE = tmp_path / "allowlist.json"
    yield


def login(client):
    add_to_allowlist('user@example.com', 'secret')
    token = _csrf_token(client, '/login')
    client.post(
        '/login',
        data={'email': 'user@example.com', 'password': 'secret', 'csrf_token': token},
        follow_redirects=True,
    )


def test_erlang_requires_login():
    client = app.test_client()
    response = client.get('/apps/erlang')
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/login')


def test_erlang_authenticated_get():
    client = app.test_client()
    login(client)
    with captured_templates(app) as templates:
        response = client.get('/apps/erlang')
    assert response.status_code == 200
    assert templates and templates[0][0].name == 'apps/erlang.html'


def test_erlang_post_calculates_metrics():
    client = app.test_client()
    login(client)
    # The template does not include a CSRF field, obtain one from another page
    token = _csrf_token(client, '/login')
    matrix = [[1.0]*24] + [[0.0]*24 for _ in range(6)]
    data = {'matrix': json.dumps(matrix), 'csrf_token': token}
    with captured_templates(app) as templates:
        response = client.post('/apps/erlang', data=data)
    assert response.status_code == 200
    template, context = templates[0]
    assert template.name == 'apps/erlang.html'
    assert context['metrics']['working_days'] == 1
    assert 'demand' in context['heatmaps']
