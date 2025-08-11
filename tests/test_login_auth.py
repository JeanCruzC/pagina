import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())

from website import create_app
import website.auth as auth_module

app = create_app()
add_to_allowlist = auth_module.add_to_allowlist


@pytest.fixture(autouse=True)
def temp_allowlist(tmp_path):
    auth_module.ALLOWLIST_FILE = tmp_path / "allowlist.json"
    yield


def test_no_access_with_wrong_credentials():
    add_to_allowlist('user@example.com', 'secret')
    client = app.test_client()
    response = client.post(
        '/login',
        data={'email': 'user@example.com', 'password': 'wrong'},
        follow_redirects=False,
    )
    assert response.status_code == 200
    assert b'Iniciar sesi' in response.data


def test_access_with_valid_credentials():
    add_to_allowlist('user@example.com', 'secret')
    client = app.test_client()
    response = client.post(
        '/login',
        data={'email': 'user@example.com', 'password': 'secret'},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/generador')
