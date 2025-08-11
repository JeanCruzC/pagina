import os
import sys
import types
from contextlib import contextmanager

import pytest
from flask import template_rendered

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())

from website import create_app
from website.utils import allowlist as allowlist_module

app = create_app()


@contextmanager
def captured_templates(app):
    recorded = []
    def record(sender, template, context, **extra):
        recorded.append(template)
    template_rendered.connect(record, app)
    try:
        yield recorded
    finally:
        template_rendered.disconnect(record, app)


@pytest.fixture(autouse=True)
def temp_allowlist(tmp_path):
    allowlist_module.ALLOWLIST_FILE = tmp_path / "allowlist.json"
    yield


def test_subscribe_success_route():
    client = app.test_client()
    with captured_templates(app) as templates:
        response = client.get('/subscribe/success')
    assert response.status_code == 200
    assert templates and templates[0].name == 'subscribe_success.html'
