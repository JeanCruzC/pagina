import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app as app_package
from app.auth import utils as auth_utils

app = app_package.create_app()
add_to_allowlist = auth_utils.add_to_allowlist


@pytest.fixture(autouse=True)
def temp_allowlist(tmp_path):
    auth_utils.ALLOWLIST_FILE = tmp_path / "allowlist.json"
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
