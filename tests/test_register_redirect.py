import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app as app_package

app = app_package.create_app()


def test_register_redirects_to_login():
    client = app.test_client()
    response = client.get('/register')
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/login')
