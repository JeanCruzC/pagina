import os
import sys
import types
from io import BytesIO
import importlib

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault(
    'plotly',
    types.SimpleNamespace(graph_objects=types.SimpleNamespace(), express=types.SimpleNamespace()),
)
sys.modules.setdefault('plotly.graph_objects', types.SimpleNamespace())
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
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())
sys.modules.pop('website.utils.kpis_core', None)
import website.blueprints.core as core_blueprint
importlib.reload(core_blueprint)

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


def test_kpis_requires_login():
    client = app.test_client()
    response = client.get('/kpis')
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/login')


def test_kpis_post_generates_response():
    client = app.test_client()
    login(client)
    token = _csrf_token(client, '/kpis')
    data = {
        'file': (BytesIO(b'col1,col2\n1,2\n3,4'), 'test.csv'),
        'csrf_token': token,
    }
    response = client.post('/kpis', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b'KPIs' in response.data
