import os
import sys
import types
from io import BytesIO

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())

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


def test_resultados_without_result_shows_message():
    client = app.test_client()
    login(client)
    response = client.get('/resultados')
    assert response.status_code == 200
    assert b'No hay resultados' in response.data


def test_generador_stores_and_renders_result():
    client = app.test_client()
    login(client)
    sys.modules['website.scheduler'].run_complete_optimization = (
        lambda *a, **k: ({'metrics': {}}, b'', b'')
    )
    token = _csrf_token(client, '/generador')
    data = {'excel': (BytesIO(b'data'), 'test.xlsx'), 'csrf_token': token}
    response = client.post(
        '/generador',
        data=data,
        content_type='multipart/form-data',
        headers={'Accept': 'application/json'},
    )
    assert response.status_code == 202
    job_id = response.get_json()['job_id']
    import time
    for _ in range(20):
        status = client.get(f'/generador/status/{job_id}').get_json()['status']
        if status == 'finished':
            break
        time.sleep(0.1)
    result_page = client.get('/resultados')
    assert result_page.status_code == 200
    assert b'Resultados' in result_page.data
