import os
import sys
import types
import time
from io import BytesIO

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
STORE = {"jobs": {}, "results": {}}
scheduler_ns = types.SimpleNamespace(
    init_app=lambda app: None,
    mark_running=lambda job_id, app=None: STORE["jobs"].update({job_id: {"status": "running"}}),
    mark_finished=lambda job_id, result, excel_path, csv_path, app=None: (
        STORE["jobs"].update({job_id: {"status": "finished"}}),
        STORE["results"].update(
            {
                job_id: {
                    "result": result,
                    "excel_path": excel_path,
                    "csv_path": csv_path,
                    "timestamp": time.time(),
                }
            }
        ),
    ),
    mark_error=lambda job_id, msg, app=None: STORE["jobs"].update({job_id: {"status": "error", "error": msg}}),
    mark_cancelled=lambda job_id, app=None: STORE["jobs"].update({job_id: {"status": "cancelled"}}),
    get_status=lambda job_id, app=None: STORE["jobs"].get(job_id, {"status": "unknown"}),
    get_result=lambda job_id, app=None: STORE["results"].get(job_id),
    get_payload=lambda job_id, app=None: STORE["results"].get(job_id),
    update_progress=lambda job_id, info, app=None: STORE["jobs"].setdefault(job_id, {"status": "running"}).setdefault("progress", {}).update(info),
    run_complete_optimization=lambda *a, **k: ({}, b"", b""),
    active_jobs={},
    _store=lambda app=None: STORE,
)
sys.modules['website.scheduler'] = scheduler_ns

import website.generator_routes as generator_module
generator_module.scheduler = scheduler_ns
import website
website.scheduler = scheduler_ns

from website import create_app
from website.utils import allowlist as allowlist_module

app = create_app()
generator_module.scheduler = scheduler_ns
website.scheduler = scheduler_ns
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


def test_resultados_without_result_returns_500():
    client = app.test_client()
    login(client)
    response = client.get('/resultados/unknown')
    assert response.status_code == 500
    assert b'Resultado no disponible' in response.data


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
        status_resp = client.get(f'/generador/status/{job_id}')
        status_data = status_resp.get_json()
        if status_data['status'] == 'finished':
            break
        time.sleep(0.1)
    assert status_data['status'] == 'finished'
    assert status_data['redirect'] == f'/resultados/{job_id}'
    result_page = client.get(f'/resultados/{job_id}')
    assert result_page.status_code == 200
    assert b'Resultados' in result_page.data


def test_refresh_returns_progress_fields():
    client = app.test_client()
    login(client)
    job_id = "job123"
    # Reset scheduler to mocked namespace in case other tests replaced it
    sys.modules['website.scheduler'] = scheduler_ns
    generator_module.scheduler = scheduler_ns
    website.scheduler = scheduler_ns
    STORE["results"][job_id] = {
        "result": {"pulp_results": {"assignments": {}}, "greedy_results": {}},
        "excel_path": None,
        "csv_path": None,
        "timestamp": time.time(),
    }
    scheduler_ns.update_progress(job_id, {"jean_iter": 1, "jean_status": "solving", "jean_time": 0.5})
    resp = client.get(f'/resultados/{job_id}/refresh')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['jean_iter'] == 1
    assert data['jean_status'] == 'solving'
    assert data['jean_time'] == 0.5
