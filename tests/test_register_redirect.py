import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
module = types.ModuleType("scheduler")
module.active_jobs = {}
module._stop_thread = lambda t: None
sys.modules.setdefault('website.scheduler', module)

from website import create_app

app = create_app()


def test_register_redirects_to_login():
    client = app.test_client()
    response = client.get('/register')
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/login')
