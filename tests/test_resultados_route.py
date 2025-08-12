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


def test_resultados_redirects_without_result():
    client = app.test_client()
    login(client)
    response = client.get('/resultados')
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/generador')


def test_generador_stores_and_renders_result():
    client = app.test_client()
    login(client)
    sys.modules['website.scheduler'].run_complete_optimization = (
        lambda *a, **k: ({'metrics': {}}, b'')
    )
    token = _csrf_token(client, '/generador')
    data = {'excel': (BytesIO(b'data'), 'test.xlsx'), 'csrf_token': token}
    response = client.post('/generador', data=data, content_type='multipart/form-data', follow_redirects=False)
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/resultados')
    result_page = client.get('/resultados')
    assert result_page.status_code == 200
    assert b'Resultados' in result_page.data
    # After rendering once, the result should be cleared
    response_again = client.get('/resultados')
    assert response_again.status_code == 302
    assert response_again.headers['Location'].endswith('/generador')
