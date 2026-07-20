import pandas as pd

from app.routes.api import _infer_frequency


def test_infer_frequency_hourly_data():
    df = pd.DataFrame({"datetime": ["2026-03-01 10:00:00", "2026-03-01 11:00:00"]})
    assert _infer_frequency(df) == "H"


def test_infer_frequency_daily_data():
    df = pd.DataFrame({"datetime": ["2026-03-01", "2026-03-02"]})
    assert _infer_frequency(df) == "D"

