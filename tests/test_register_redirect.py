import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())

from website import create_app

app = create_app()


def test_register_redirects_to_login():
    client = app.test_client()
    response = client.get('/register')
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/login')
