import os
import sys
import types
import time
from io import BytesIO

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

# Stub heavy optional dependencies to speed up tests
try:  # pragma: no cover - optional heavy deps
    import numpy  # noqa: F401
except Exception:  # pragma: no cover - allow tests without numpy
    sys.modules['numpy'] = types.SimpleNamespace(ndarray=object, generic=object)
try:  # pragma: no cover
    import pandas  # noqa: F401
except Exception:  # pragma: no cover
    _Dummy = type('Dummy', (), {})
    sys.modules['pandas'] = types.SimpleNamespace(DataFrame=_Dummy, Series=_Dummy)

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


def test_cancel_before_worker_enters_run_complete_optimization(monkeypatch):
    client = app.test_client()
    login(client)

    run_called = {"flag": False}

    def fake_run(*a, **k):
        run_called["flag"] = True
        return {}, b"", b""

    sched = types.SimpleNamespace(
        init_app=lambda app: None,
        mark_running=lambda job_id, app=None: _store.setdefault(job_id, {"status": "running"}),
        mark_cancelled=lambda job_id, app=None: _store.update({job_id: {"status": "cancelled"}}),
        get_status=lambda job_id, app=None: _store.get(job_id, {"status": "unknown"}),
        get_result=lambda job_id, app=None: _store.get(job_id),
        active_jobs={},
        run_complete_optimization=fake_run,
    )
    monkeypatch.setitem(sys.modules, 'website.scheduler', sched)
    monkeypatch.setattr(generator_module, 'scheduler', sched)
    monkeypatch.setattr(website, 'scheduler', sched)

    def waiting_worker(app, job_id, file_bytes, config, generate_charts):
        with app.app_context():
            for _ in range(50):
                if sched.get_status(job_id)['status'] == 'cancelled':
                    return
                time.sleep(0.1)
            fake_run(BytesIO(file_bytes), config=config, generate_charts=generate_charts, job_id=job_id)

    monkeypatch.setattr(generator_module, '_worker', waiting_worker)

    token = _csrf_token(client, '/generador')
    data = {'excel': (BytesIO(b'data'), 'test.xlsx'), 'csrf_token': token}
    resp = client.post(
        '/generador',
        data=data,
        content_type='multipart/form-data',
        headers={'Accept': 'application/json'},
    )
    assert resp.status_code == 202
    job_id = resp.get_json()['job_id']

    cancel_resp = client.post('/cancel', json={'job_id': job_id})
    assert cancel_resp.status_code == 204
    time.sleep(0.5)

    assert sys.modules['website.scheduler'].get_status(job_id)['status'] == 'cancelled'
    assert sys.modules['website.scheduler'].active_jobs == {}
    assert run_called['flag'] is False


def test_download_parent_path_returns_404():
    client = app.test_client()
    login(client)
    response = client.get('/download/../../etc/passwd')
    assert response.status_code == 404
