import os
import sys
import types
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('website.scheduler', types.SimpleNamespace(get_payload=lambda *a, **k: None))

from website import create_app
from website.blueprints import core

app = create_app()


def test_cleanup_by_age():
    with app.app_context():
        core._RESULTS.clear()
        core._RESULTS['old'] = {'timestamp': time.time() - 7200}
        core._RESULTS['new'] = {'timestamp': time.time()}
        core.cleanup_results(max_age=3600)
        assert 'old' not in core._RESULTS
        assert 'new' in core._RESULTS


def test_cleanup_by_size():
    with app.app_context():
        core._RESULTS.clear()
        now = time.time()
        for i in range(5):
            core._RESULTS[str(i)] = {'timestamp': now + i}
        core.cleanup_results(max_entries=3)
        assert len(core._RESULTS) == 3
        assert set(core._RESULTS.keys()) == {'2', '3', '4'}
