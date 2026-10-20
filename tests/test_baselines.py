import numpy as np
import pandas as pd
import pytest

from src.baselines import drift_forecast, estimate_drift, naive_forecast


def test_naive_repeats_last_log_price():
    X = np.arange(24, dtype=float).reshape(2, 4, 3)
    out = naive_forecast(X, 5)
    assert out.shape == (2, 5)
    np.testing.assert_allclose(out[0], X[0, -1, 0])
    np.testing.assert_allclose(out[1], X[1, -1, 0])


def test_drift_adds_tau_times_mu():
    X = np.ones((1, 3, 1)) * 4.0
    out = drift_forecast(X, 5, 0.01)
    np.testing.assert_allclose(out[0], 4.0 + 0.01 * np.arange(1, 6))


def test_estimate_drift_uses_only_train_range_and_skips_excluded():
    idx = pd.bdate_range("2012-01-02", periods=100)
    r = np.where(np.arange(100) < 60, 0.001, 0.5)          # 40 ngày sau là "tương lai" cực đoan
    df = pd.DataFrame({"wti_logret": r, "is_excluded": False}, index=idx)
    assert estimate_drift(df, idx[0], idx[59]) == pytest.approx(0.001)
    df.loc[idx[10:20], "is_excluded"] = True
    df.loc[idx[10:20], "wti_logret"] = 9.0
    assert estimate_drift(df, idx[0], idx[59]) == pytest.approx(0.001)
