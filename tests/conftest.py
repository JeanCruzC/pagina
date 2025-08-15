import sys, types

# Stub heavy optional modules to keep tests lightweight
ts_core = types.SimpleNamespace()
def _ts_run(params, file_storage=None):
    return {
        'metrics': {'demo': 1},
        'recommendation': 'ok',
        'weekly_table': [{'semana_iso': 1, 'planificados': 1, 'reales': 1, 'desvio_abs': 0, 'desvio_pct': 0}],
        'heatmap': {},
        'interactive': {},
    }
ts_core.run = _ts_run
sys.modules.setdefault('website.other.timeseries_core', ts_core)

mp_core = types.SimpleNamespace()
def _mp_run(file, steps):
    return {
        'metrics': {'mae': 0.0},
        'table': [{'a': 1}],
        'file_bytes': b'dummy'
    }
mp_core.run = _mp_run
sys.modules.setdefault('website.other.modelo_predictivo_core', mp_core)

sys.modules.setdefault('website.other.timeseries_full_core', types.SimpleNamespace())
