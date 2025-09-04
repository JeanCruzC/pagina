import os
import sys
import types
import time
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

STORE = {"results": {}}
sys.modules['website.scheduler'] = types.SimpleNamespace(
    _store=lambda app=None: STORE,
    run_complete_optimization=lambda *a, **k: None,
)

from website import create_app
from website.blueprints import core as core_module

core = importlib.reload(core_module)

app = create_app()


def test_cleanup_by_age():
    with app.app_context():
        STORE["results"].clear()
        STORE["results"]["old"] = {"timestamp": time.time() - 7200}
        STORE["results"]["new"] = {"timestamp": time.time()}
        core.cleanup_results(max_age=3600)
        assert "old" not in STORE["results"]
        assert "new" in STORE["results"]


def test_cleanup_by_size():
    with app.app_context():
        STORE["results"].clear()
        now = time.time()
        for i in range(5):
            STORE["results"][str(i)] = {"timestamp": now + i}
        core.cleanup_results(max_entries=3)
        assert len(STORE["results"]) == 3
        assert set(STORE["results"].keys()) == {"2", "3", "4"}
