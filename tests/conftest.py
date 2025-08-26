import os
import sys
import types

# Ensure repository root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Provide a scheduler stub with PROGRESS to satisfy blueprint imports when tests
# replace the module without defining it.
sys.modules.setdefault('website.scheduler', types.SimpleNamespace(PROGRESS={}))

# Stub optional heavy dependency with minimal Workbook implementation
class _Sheet:
    def append(self, row):
        pass


class _Workbook:
    def __init__(self, *args, **kwargs):
        self.active = _Sheet()

    def save(self, *args, **kwargs):
        pass


def _load_workbook(*args, **kwargs):
    return _Workbook()


sys.modules.setdefault('openpyxl', types.SimpleNamespace(Workbook=_Workbook, load_workbook=_load_workbook))
