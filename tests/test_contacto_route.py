import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace(get_payload=lambda *a, **k: None))

from website import create_app
from website.utils import allowlist as allowlist_module

app = create_app()


@pytest.fixture(autouse=True)
def temp_allowlist(tmp_path):
    allowlist_module.ALLOWLIST_FILE = tmp_path / "allowlist.json"
    yield


def test_contacto_route():
    client = app.test_client()
    response = client.get('/contacto')
    assert response.status_code == 200
    assert b'Contacto' in response.data

