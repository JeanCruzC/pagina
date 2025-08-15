import os
import sys
import types
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())

from website import create_app
from website.utils import allowlist as allowlist_module

app = create_app()
add_to_allowlist = allowlist_module.add_to_allowlist


def _get_html_and_token(client, path):
    resp = client.get(path)
    html = resp.get_data(as_text=True)
    import re
    m = re.search(r'name="csrf_token" value="([^"]+)"', html)
    return html, m.group(1) if m else None


@pytest.fixture(autouse=True)
def temp_allowlist(tmp_path):
    allowlist_module.ALLOWLIST_FILE = tmp_path / "allowlist.json"
    yield


def login(client):
    add_to_allowlist('user@example.com', 'secret')
    _, token = _get_html_and_token(client, '/login')
    client.post(
        '/login',
        data={'email': 'user@example.com', 'password': 'secret', 'csrf_token': token},
        follow_redirects=True,
    )


def test_apps_timeseries_requires_login():
    client = app.test_client()
    response = client.get('/apps/timeseries')
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/login')


def test_apps_timeseries_page_contains_csrf_and_heading():
    client = app.test_client()
    login(client)
    html, token = _get_html_and_token(client, '/apps/timeseries')
    assert token
    assert 'Timeseries App' in html
