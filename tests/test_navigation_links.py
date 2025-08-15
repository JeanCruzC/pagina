import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())
sys.modules.setdefault('plotly', types.SimpleNamespace(graph_objects=types.SimpleNamespace(Figure=object)))
sys.modules.setdefault('plotly.graph_objects', sys.modules['plotly'].graph_objects)
sys.modules.setdefault('website.other.timeseries_core', types.SimpleNamespace())
sys.modules.setdefault('website.other.erlang_core', types.SimpleNamespace())
sys.modules.setdefault('website.other.modelo_predictivo_core', types.SimpleNamespace())
sys.modules.setdefault('website.utils.kpis_core', types.SimpleNamespace())

from flask import Blueprint

_apps_bp = Blueprint("apps", __name__, url_prefix="/apps")

@_apps_bp.route("/erlang")
def erlang():
    return ""

@_apps_bp.route("/timeseries")
def timeseries():
    return ""

@_apps_bp.route("/predictivo")
def predictivo():
    return ""

@_apps_bp.route("/kpis")
def kpis():
    return ""

sys.modules.setdefault('website.blueprints.apps', types.SimpleNamespace(apps_bp=_apps_bp))

from website import create_app
from website.utils import allowlist as allowlist_module

app = create_app()
add_to_allowlist = allowlist_module.add_to_allowlist


def _csrf_token(client):
    resp = client.get('/login')
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
    token = _csrf_token(client)
    client.post(
        '/login',
        data={'email': 'user@example.com', 'password': 'secret', 'csrf_token': token},
        follow_redirects=True,
    )


def test_nav_links_authenticated():
    client = app.test_client()
    login(client)
    response = client.get('/generador')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert '/apps/erlang' in html
    assert 'Apps</a>' in html
    assert '/apps/timeseries' in html
    assert '/apps/predictivo' in html
    assert '/apps/kpis' in html
