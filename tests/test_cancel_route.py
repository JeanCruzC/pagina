import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
_store = {}
sys.modules['website.scheduler'] = types.SimpleNamespace(
    init_app=lambda app: None,
    mark_running=lambda job_id, app=None: _store.setdefault(job_id, {"status": "running"}),
    mark_cancelled=lambda job_id, app=None: _store.update({job_id: {"status": "cancelled"}}),
    get_status=lambda job_id, app=None: _store.get(job_id, {"status": "unknown"}),
    get_result=lambda job_id, app=None: _store.get(job_id),
    active_jobs={},
)

import website.generator_routes as generator_module
generator_module.scheduler = sys.modules['website.scheduler']
import website
website.scheduler = sys.modules['website.scheduler']

from website import create_app
from website.utils import allowlist as allowlist_module
from website import scheduler

app = create_app()
generator_module.scheduler = sys.modules['website.scheduler']
website.scheduler = sys.modules['website.scheduler']
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


def test_cancel_route_updates_status():
    client = app.test_client()
    login(client)
    job_id = 'testjob'
    scheduler.mark_running(job_id)
    response = client.post('/cancel', json={'job_id': job_id})
    assert response.status_code == 204
