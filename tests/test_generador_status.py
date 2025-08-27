import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())

from website import create_app
from website.utils import allowlist as allowlist_module
from website.generator_routes import JOBS, _worker

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


@pytest.fixture(autouse=True)
def clear_jobs():
    JOBS.clear()
    yield
    JOBS.clear()


def login(client):
    add_to_allowlist('user@example.com', 'secret')
    token = _csrf_token(client, '/login')
    client.post(
        '/login',
        data={'email': 'user@example.com', 'password': 'secret', 'csrf_token': token},
        follow_redirects=True,
    )


def test_worker_failure_stores_error(monkeypatch):
    def failing_run(*args, **kwargs):
        raise RuntimeError('boom')
    from website import generator_routes as gr
    monkeypatch.setattr(gr, 'scheduler', types.SimpleNamespace(run_complete_optimization=failing_run))
    job_id = 'jobfail'
    _worker(app, job_id, b'', {}, False)
    assert JOBS[job_id] == {'status': 'error', 'error': 'boom'}


def test_generador_status_transitions():
    client = app.test_client()
    login(client)

    # unknown job
    response = client.get('/generador/status/nope')
    assert response.get_json() == {'status': 'unknown'}

    # running job
    job_id = 'running'
    JOBS[job_id] = {'status': 'running'}
    response = client.get(f'/generador/status/{job_id}')
    assert response.get_json() == {'status': 'running'}

    # finished job
    job_id = 'done'
    JOBS[job_id] = {'status': 'finished', 'result': {'x': 1}}
    response = client.get(f'/generador/status/{job_id}')
    assert response.get_json() == {'status': 'finished'}
    with client.session_transaction() as sess:
        assert sess['resultado'] == {'x': 1}

    # error job
    job_id = 'fail'
    JOBS[job_id] = {'status': 'error', 'error': 'oops'}
    response = client.get(f'/generador/status/{job_id}')
    assert response.get_json() == {'status': 'error', 'error': 'oops'}
