import os
import sys
import types

import pytest
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())
sys.modules.setdefault('website.other.timeseries_core', types.SimpleNamespace())
sys.modules.setdefault('website.other.erlang_core', types.SimpleNamespace())
sys.modules.setdefault('website.other.modelo_predictivo_core', types.SimpleNamespace())

from website import create_app
from website.utils import allowlist as allowlist_module

app = create_app()


@pytest.fixture(autouse=True)
def temp_allowlist(tmp_path):
    allowlist_module.ALLOWLIST_FILE = tmp_path / "allowlist.json"
    yield


def test_contacto_route():
    client = app.test_client()
    response = client.get('/contacto')
    assert response.status_code == 200
    assert b'Contacto' in response.data


def _csrf_token(client):
    resp = client.get('/contacto')
    html = resp.get_data(as_text=True)
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    return match.group(1) if match else None


def test_contacto_htmx_post():
    client = app.test_client()
    token = _csrf_token(client)
    response = client.post(
        '/contacto',
        data={'name': 'Tester', 'csrf_token': token},
        headers={'HX-Request': 'true'},
    )
    assert response.status_code == 200
    assert b'Gracias' in response.data

