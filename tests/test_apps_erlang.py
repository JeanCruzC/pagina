import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())
sys.modules.setdefault('website.utils.kpis_core', types.SimpleNamespace())

# Stub heavy optional dependencies used by the application so tests can run
# without installing the full requirements.
sys.modules.setdefault('pandas', types.SimpleNamespace())
class _DummyFigure:  # minimal stand-in for plotly.graph_objects.Figure
    pass

sys.modules.setdefault(
    'plotly',
    types.SimpleNamespace(graph_objects=types.SimpleNamespace(Figure=_DummyFigure)),
)
sys.modules.setdefault('plotly.graph_objects', types.SimpleNamespace(Figure=_DummyFigure))

# Stub unused service/other modules imported by the apps blueprint
sys.modules.setdefault('website.other.erlang_core', types.SimpleNamespace())
sys.modules.setdefault('website.other.timeseries_core', types.SimpleNamespace())
sys.modules.setdefault('website.other.modelo_predictivo_core', types.SimpleNamespace())
sys.modules.setdefault('website.other.erlang_visual', types.SimpleNamespace())
sys.modules.setdefault('website.other.comparativo_core', types.SimpleNamespace())
sys.modules.setdefault('website.other.staffing_core', types.SimpleNamespace())
sys.modules.setdefault('website.other.batch_core', types.SimpleNamespace())
sys.modules.setdefault('website.services.erlang', types.SimpleNamespace())
sys.modules.setdefault('website.services.erlang_o', types.SimpleNamespace())

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
    assert b'erlang-form' in response.data


def test_erlang_post_calculates(monkeypatch):
    from website.other import erlang_core

    def fake_calc(**kwargs):
        return {
            "service_level": 0.8,
            "asa": 20,
            "occupancy": 0.5,
            "required_agents": 5,
        }

    monkeypatch.setattr(
        erlang_core, "calculate_erlang_metrics", fake_calc, raising=False
    )

    client = app.test_client()
    login(client)
    token = _csrf_token(client, '/apps/erlang')
    response = client.post(
        '/apps/erlang',
        data={
            'calls': '100',
            'aht': '30',
            'awl': '20',
            'agents': '10',
            'interval': '3600',
            'target_sl': '0.8',
            'csrf_token': token,
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'SL:' in html
    assert 'ASA:' in html
    assert 'Ocupación:' in html
    assert 'Requeridos:' in html
    assert '80' in html
    assert '20' in html
    assert '50' in html
    assert '5' in html


def test_erlang_subroute_authenticated():
    client = app.test_client()
    login(client)
    response = client.get('/apps/erlang/visual')
    assert response.status_code == 200
