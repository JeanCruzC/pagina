"""Legacy wrapper for forecasting logic.

The original Streamlit implementation has been deprecated in favour of a
Flask blueprint.  The forecasting utilities now live in
``website.other.predictivo_core``.
"""
from website.other.predictivo_core import (
    forecast,
    detectar_frecuencia,
    calcular_estacionalidad,
)

__all__ = ["forecast", "detectar_frecuencia", "calcular_estacionalidad"]

if __name__ == "__main__":  # pragma: no cover - manual usage only
    print(
        "Modelo_Predictivo.py is now a thin wrapper. Use the Flask interface "
        "at /apps/predictivo instead."
    )
