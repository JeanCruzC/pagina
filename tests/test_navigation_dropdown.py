import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())

import website.app as app_module

app = app_module.app
add_to_allowlist = app_module.add_to_allowlist


@pytest.fixture(autouse=True)
def temp_allowlist(tmp_path):
    app_module.ALLOWLIST_FILE = tmp_path / "allowlist.json"
    yield


def login(client):
    add_to_allowlist('user@example.com', 'secret')
    client.post('/login', data={'email': 'user@example.com', 'password': 'secret'}, follow_redirects=True)


def test_dropdown_public():
    client = app.test_client()
    response = client.get('/login')
    assert response.status_code == 200
    assert b'Invitado' in response.data
    assert b'Iniciar sesi' in response.data


def test_dropdown_authenticated():
    client = app.test_client()
    login(client)
    response = client.get('/generador')
    assert response.status_code == 200
    assert b'user@example.com' in response.data
    assert b'Salir' in response.data
