import io
import os
import sys
import types

import io
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


def test_predictivo_requires_login():
    client = app.test_client()
    response = client.get('/apps/predictivo')
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/login')


def test_predictivo_authenticated_get():
    client = app.test_client()
    login(client)
    response = client.get('/apps/predictivo')
    assert response.status_code == 200
    assert b'predictivo-form' in response.data


def test_predictivo_post_returns_results():
    client = app.test_client()
    login(client)
    token = _csrf_token(client, '/apps/predictivo')
    csv_content = 'date,value\n2023-01-01,1\n2023-02-01,2\n2023-03-01,3\n'
    data = {
        'steps_horizon': '2',
        'csrf_token': token,
        'file': (io.BytesIO(csv_content.encode('utf-8')), 'data.csv'),
    }
    response = client.post(
        '/apps/predictivo',
        data=data,
        content_type='multipart/form-data',
        follow_redirects=True,
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'predictivo-table' in html
    assert 'predictivo-download' in html


def test_predictivo_download_endpoint():
    client = app.test_client()
    login(client)
    token = _csrf_token(client, '/apps/predictivo')
    csv_content = 'date,value\n2023-01-01,1\n2023-02-01,2\n'
    data = {
        'steps_horizon': '1',
        'csrf_token': token,
        'file': (io.BytesIO(csv_content.encode('utf-8')), 'data.csv'),
    }
    response = client.post(
        '/apps/predictivo',
        data=data,
        content_type='multipart/form-data',
        follow_redirects=True,
    )
    html = response.get_data(as_text=True)
    import re
    match = re.search(r'id="predictivo-download"[^>]*href="([^"]+)"', html)
    assert match, 'download link not found'
    download_url = match.group(1)
    dl_resp = client.get(download_url)
    assert dl_resp.status_code == 200
    assert dl_resp.headers['Content-Disposition'].startswith('attachment')
    assert dl_resp.data
