import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace())

from website import create_app

app = create_app()


def test_subscribe_success_route():
    client = app.test_client()
    response = client.get('/subscribe/success')
    assert response.status_code == 200
    assert 'Suscripción completa' in response.get_data(as_text=True)
