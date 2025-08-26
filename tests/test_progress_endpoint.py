import os
import sys
import threading
import time


def test_progress_endpoint_reports_completion():
    orig_scheduler = sys.modules.get('website.scheduler')
    orig_core = sys.modules.get('website.blueprints.core')
    orig_website = sys.modules.get('website')
    stubbed = {}
    try:
        for mod in ['website.scheduler', 'website.blueprints.core', 'website']:
            sys.modules.pop(mod, None)
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import types
        stubbed_names = {
            'plotly': types.SimpleNamespace(
                graph_objects=types.SimpleNamespace(), express=types.SimpleNamespace()
            ),
            'plotly.graph_objects': types.SimpleNamespace(),
            'plotly.express': types.SimpleNamespace(),
            'sklearn': types.SimpleNamespace(
                metrics=types.SimpleNamespace(
                    mean_absolute_error=lambda *a, **k: 0,
                    mean_squared_error=lambda *a, **k: 0,
                )
            ),
            'sklearn.metrics': types.SimpleNamespace(
                mean_absolute_error=lambda *a, **k: 0,
                mean_squared_error=lambda *a, **k: 0,
            ),
            'seaborn': types.SimpleNamespace(set_theme=lambda *a, **k: None),
            'scipy': types.SimpleNamespace(optimize=types.SimpleNamespace()),
            'scipy.optimize': types.SimpleNamespace(),
            'statsmodels': types.SimpleNamespace(
                tsa=types.SimpleNamespace(
                    holtwinters=types.SimpleNamespace(ExponentialSmoothing=object)
                )
            ),
            'statsmodels.tsa': types.SimpleNamespace(
                holtwinters=types.SimpleNamespace(ExponentialSmoothing=object)
            ),
            'statsmodels.tsa.holtwinters': types.SimpleNamespace(ExponentialSmoothing=object),
        }
        for name, module in stubbed_names.items():
            stubbed[name] = sys.modules.get(name)
            sys.modules[name] = module
        from website import create_app
        from website.scheduler import log_solver_progress, PROGRESS
        app = create_app()
        job_id = 'job1'
        stop_event = threading.Event()
        thread = threading.Thread(target=log_solver_progress, args=(1, stop_event, job_id))
        thread.daemon = True
        thread.start()
        time.sleep(1.2)
        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess['user'] = 'tester'
            resp = client.get(f'/progress/{job_id}')
            assert resp.get_json()['percent'] == 99
            stop_event.set()
            thread.join()
            resp = client.get(f'/progress/{job_id}')
            assert resp.get_json()['percent'] == 100
    finally:
        if orig_scheduler is not None:
            sys.modules['website.scheduler'] = orig_scheduler
        else:
            sys.modules.pop('website.scheduler', None)
        if orig_core is not None:
            sys.modules['website.blueprints.core'] = orig_core
        else:
            sys.modules.pop('website.blueprints.core', None)
        if orig_website is not None:
            sys.modules['website'] = orig_website
        else:
            sys.modules.pop('website', None)
        for name, module in stubbed.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)
