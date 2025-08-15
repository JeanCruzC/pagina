import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from website.other import predictivo_core


def test_forecast_returns_dataframe():
    series = pd.Series(
        [1, 2, 3, 4, 5], index=pd.date_range("2024-01-01", periods=5, freq="D")
    )
    result = predictivo_core.forecast(series, steps=2)
    fc_df = result["forecast"]
    assert len(fc_df) == 2
    assert set(["ARIMA", "WES"]).issubset(fc_df.columns)
