import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())
sys.modules.setdefault('website.utils.kpis_core', types.SimpleNamespace())

from website import create_app
from website.utils import allowlist as allowlist_module

app = create_app()
add_to_allowlist = allowlist_module.add_to_allowlist


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
    response = client.get('/apps/erlang')
    assert response.status_code == 200
    assert b'erlang-form' in response.data or b'Erlang app coming soon' in response.data


@pytest.mark.xfail(reason="POST handler not yet implemented")
def test_erlang_post_placeholder():
    client = app.test_client()
    login(client)
    token = _csrf_token(client, '/apps/erlang')
    response = client.post(
        '/apps/erlang',
        data={'sample': 'data', 'csrf_token': token},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b'coming soon' in response.data
