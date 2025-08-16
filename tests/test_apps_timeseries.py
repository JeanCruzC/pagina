import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())
sys.modules.setdefault('website.utils.kpis_core', types.SimpleNamespace())
class _Fig:
    def to_json(self):
        return "{}"
sys.modules.setdefault(
    'plotly',
    types.SimpleNamespace(graph_objects=types.SimpleNamespace(Figure=_Fig), express=types.SimpleNamespace()),
)
sys.modules.setdefault('plotly.graph_objects', types.SimpleNamespace(Figure=_Fig))
sys.modules.setdefault('plotly.express', types.SimpleNamespace())
_metrics_stub = types.SimpleNamespace(mean_absolute_error=lambda *a, **k: 0, mean_squared_error=lambda *a, **k: 0)
sys.modules.setdefault('sklearn', types.SimpleNamespace(metrics=_metrics_stub))
sys.modules.setdefault('sklearn.metrics', _metrics_stub)
sys.modules.setdefault('seaborn', types.SimpleNamespace(set_theme=lambda *a, **k: None))
sys.modules.setdefault('scipy', types.SimpleNamespace(optimize=types.SimpleNamespace()))
sys.modules.setdefault('scipy.optimize', types.SimpleNamespace())
_hw_stub = types.SimpleNamespace(ExponentialSmoothing=object)
sys.modules.setdefault('statsmodels', types.SimpleNamespace(tsa=types.SimpleNamespace(holtwinters=_hw_stub)))
sys.modules.setdefault('statsmodels.tsa', types.SimpleNamespace(holtwinters=_hw_stub))
sys.modules.setdefault('statsmodels.tsa.holtwinters', _hw_stub)
sys.modules['website.other.timeseries_core'] = types.SimpleNamespace(
    run=lambda params, file_storage=None: {
        'metrics': {'a': 1},
        'recommendation': 'ok',
        'weekly_table': [[1]],
        'heatmap': _Fig(),
        'interactive': _Fig(),
    }
)
import website.blueprints.apps as apps_module
apps_module.timeseries_core = sys.modules['website.other.timeseries_core']
apps_module.go = types.SimpleNamespace(Figure=_Fig)

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


def test_timeseries_requires_login():
    client = app.test_client()
    response = client.get('/apps/timeseries')
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/login')


def test_timeseries_authenticated_get():
    client = app.test_client()
    login(client)
    response = client.get('/apps/timeseries')
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'timeseries-form' in html
    assert 'type="file"' in html
    assert 'weight_last' in html
    assert 'scope' in html


def test_timeseries_post_file_upload():
    import io

    client = app.test_client()
    login(client)
    token = _csrf_token(client, '/apps/timeseries')
    csv_content = (
        'fecha,intervalo,planif. contactos,contactos\n'
        '2024-01-01,00:00:00,10,12\n'
        '2024-01-02,00:00:00,10,8\n'
    )
    data = {
        'weight_last': '0.7',
        'weight_prev': '0.3',
        'scope': 'Total',
        'view': 'Día',
        'csrf_token': token,
        'file': (io.BytesIO(csv_content.encode('utf-8')), 'test.csv'),
    }
    response = client.post(
        '/apps/timeseries',
        data=data,
        content_type='multipart/form-data',
        follow_redirects=True,
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'timeseries-kpis' in html
    assert 'timeseries-recommendations' in html
    assert 'timeseries-weekly-table' in html
    assert 'timeseries-heatmap' in html
    assert 'timeseries-interactive' in html


def test_series_alias_route():
    client = app.test_client()
    login(client)
    response = client.get('/apps/series')
    assert response.status_code == 200
