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



def test_erlang_metrics_mode(monkeypatch):
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
            'forecast': '100',
            'aht': '30',
            'agents': '10',
            'sl_target': '0.8',
            'awl': '20',
            'agents_max': '20',
            'calc_type': 'metrics',
            'advanced': 'on',
            'lines': '30',
            'patience': '60',
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


def test_erlang_required_agents_mode(monkeypatch):
    from website.other import erlang_core

    def fake_calc(**kwargs):
        return {
            "service_level": 0.75,
            "asa": 22,
            "occupancy": 0.6,
            "required_agents": 7,
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
            'forecast': '200',
            'aht': '45',
            'agents': '5',
            'sl_target': '0.9',
            'awl': '30',
            'agents_max': '50',
            'calc_type': 'required_agents',
            'advanced': 'on',
            'lines': '40',
            'patience': '100',
            'csrf_token': token,
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Requeridos' in html
    assert '7' in html


def test_erlang_download_endpoints(monkeypatch):
    from website.other import erlang_core
    import pandas as pd

    class DummyDF:
        def to_csv(self, buf, index=False):
            buf.write(b"data")

        def to_excel(self, writer, index=False):
            pass

    class DummyWriter:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(erlang_core, "compute_erlang", lambda **_: DummyDF(), raising=False)
    monkeypatch.setattr(pd, "DataFrame", DummyDF, raising=False)
    monkeypatch.setattr(pd, "ExcelWriter", DummyWriter, raising=False)

    client = app.test_client()
    login(client)
    csv_resp = client.get(
        '/apps/erlang/download?fmt=csv&calls=1&aht=1&awl=1&agents=1&target_sl=0.8&interval=3600'
    )
    xlsx_resp = client.get(
        '/apps/erlang/download?fmt=xlsx&calls=1&aht=1&awl=1&agents=1&target_sl=0.8&interval=3600'
    )
    assert csv_resp.status_code == 200
    assert xlsx_resp.status_code == 200
    assert csv_resp.headers['Content-Disposition'].startswith('attachment')
    assert xlsx_resp.headers['Content-Disposition'].startswith('attachment')


def test_compute_erlang_structures(monkeypatch):
    from website.blueprints.apps import compute_erlang
    from website.other import erlang_core
    from website.services import erlang as erlang_service

    def fake_calc_metrics(**kwargs):
        return {
            "service_level": 0.8,
            "asa": 20,
            "occupancy": 0.5,
            "required_agents": 5,
        }

    def fake_sla_x(arrival_rate, aht, a, awt, lines, patience):
        return 0.8

    def fake_wait(arrival_rate, aht, a):
        return 10.0

    def fake_occ(arrival_rate, aht, a):
        return 0.5

    monkeypatch.setattr(
        erlang_core, "calculate_erlang_metrics", fake_calc_metrics, raising=False
    )
    monkeypatch.setattr(erlang_service, "sla_x", fake_sla_x, raising=False)
    monkeypatch.setattr(erlang_core, "waiting_time_erlang_c", fake_wait, raising=False)
    monkeypatch.setattr(erlang_core, "occupancy_erlang_c", fake_occ, raising=False)

    payload = {
        "forecast": "100",
        "aht": "30",
        "agents": "3",
        "sl_target": "0.8",
        "awl": "20",
        "agents_max": "5",
        "calc_type": "metrics",
    }
    result = compute_erlang(payload)
    assert result["dimension_bar"] == {
        "min": 3,
        "max": 5,
        "actual": 3,
        "recomendado": 5,
    }
    assert result["sensitivity"]["agents"] == [3, 4, 5]
    assert len(result["download"]["csv_rows"]) == 5
    assert len(result["download"]["xlsx_rows"]) == 5


def test_erlang_subroute_authenticated():
    client = app.test_client()
    login(client)
    response = client.get('/apps/erlang/visual')
    assert response.status_code == 200
