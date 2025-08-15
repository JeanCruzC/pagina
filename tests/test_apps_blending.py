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


def test_blending_requires_login():
    client = app.test_client()
    response = client.get('/apps/blending')
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/login')


def test_blending_authenticated_get():
    client = app.test_client()
    login(client)
    response = client.get('/apps/blending')
    assert response.status_code == 200
    assert b'blending-form' in response.data


def test_blending_post_calculates(monkeypatch):
    from website.other import blending_core

    def fake_calc(*args, **kwargs):
        return {
            'service_level': 0.9,
            'outbound_capacity': 15.0,
            'optimal_threshold': 2,
            'figure': '{}',
        }

    monkeypatch.setattr(blending_core, 'calculate_blending_metrics', fake_calc)

    client = app.test_client()
    login(client)
    token = _csrf_token(client, '/apps/blending')
    response = client.post(
        '/apps/blending',
        data={
            'forecast': '100',
            'aht': '30',
            'agents': '10',
            'awt': '20',
            'lines': '10',
            'patience': '100',
            'threshold': '2',
            'csrf_token': token,
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'SLA Inbound' in html
    assert 'Capacidad Outbound' in html
    assert 'Threshold Óptimo' in html
