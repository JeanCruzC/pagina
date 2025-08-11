import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())

import website.app as app_module

app = app_module.create_app()
add_to_allowlist = app_module.add_to_allowlist


def _csrf_token(client):
    resp = client.get('/login')
    html = resp.get_data(as_text=True)
    import re
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    return match.group(1) if match else None


@pytest.fixture(autouse=True)
def temp_allowlist(tmp_path):
    app_module.ALLOWLIST_FILE = tmp_path / "allowlist.json"
    yield


def test_no_access_with_wrong_credentials():
    add_to_allowlist('user@example.com', 'secret')
    client = app.test_client()
    token = _csrf_token(client)
    response = client.post(
        '/login',
        data={'email': 'user@example.com', 'password': 'wrong', 'csrf_token': token},
        follow_redirects=False,
    )
    assert response.status_code == 200
    assert b'Iniciar sesi' in response.data


def test_access_with_valid_credentials():
    add_to_allowlist('user@example.com', 'secret')
    client = app.test_client()
    token = _csrf_token(client)
    response = client.post(
        '/login',
        data={'email': 'user@example.com', 'password': 'secret', 'csrf_token': token},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/generador')
