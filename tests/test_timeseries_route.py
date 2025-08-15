import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())

from website import create_app
from website.utils import allowlist as allowlist_module

app = create_app()


def _csrf_token(client, endpoint):
    resp = client.get(endpoint)
    html = resp.get_data(as_text=True)
    import re
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    return match.group(1) if match else None


@pytest.fixture(autouse=True)
def temp_allowlist(tmp_path):
    allowlist_module.ALLOWLIST_FILE = tmp_path / "allowlist.json"
    yield


def login(client):
    allowlist_module.add_to_allowlist('user@example.com', 'secret')
    token = _csrf_token(client, '/login')
    client.post(
        '/login',
        data={'email': 'user@example.com', 'password': 'secret', 'csrf_token': token},
        follow_redirects=True,
    )


def test_timeseries_get():
    client = app.test_client()
    login(client)
    resp = client.get('/timeseries')
    assert resp.status_code == 200
    assert b'Series de Tiempo' in resp.data


def test_timeseries_post():
    client = app.test_client()
    login(client)
    token = _csrf_token(client, '/timeseries')
    resp = client.post(
        '/timeseries',
        data={'method': 'rolling', 'window': '3', 'alpha': '0.5', 'csrf_token': token},
    )
    assert resp.status_code == 200
    assert b'tsChart' in resp.data

