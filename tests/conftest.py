import sys
import types

_Dummy = type("Dummy", (), {})

# Provide lightweight stubs for optional heavy dependencies
sys.modules.setdefault("pandas", types.SimpleNamespace(DataFrame=_Dummy, Series=_Dummy))
