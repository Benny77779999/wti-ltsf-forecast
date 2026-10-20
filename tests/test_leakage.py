"""Kiểm tra chống rò rỉ thời gian ở tầng dữ liệu (dùng dữ liệu giả lập, không cần mạng)."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import align_to_calendar, build_dataset


def make_cfg():
    return {"data": {
        "target": "wti",
        "study_start": "2010-01-01",
        "fred_series": {"wti": "X", "brent": "X", "ovx": "X", "fed_funds": "X"},
        "lag_rules": {"wti": 0, "brent": 0, "ovx": 1, "fed_funds": 1},
        "max_stale_days": 5,
        "exclusion_window": {"start": "2020-03-01", "end": "2020-06-30"},
        "jump_threshold": 0.15,
        "repeat_run_threshold": 4,
    }}


def make_raw(n=60, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2012-01-02", periods=n)
    raw = pd.DataFrame(index=days)
    raw["wti"] = 80 + rng.normal(0, 1, n).cumsum()
    raw["brent"] = raw["wti"] + 5 + rng.normal(0, 0.3, n)
    raw["ovx"] = 30 + rng.normal(0, 1, n)
    raw["fed_funds"] = 0.1
    raw.index.name = "date"
    return raw


def test_align_never_uses_future_value():
    cal = pd.DatetimeIndex(["2012-01-03", "2012-01-04", "2012-01-05"])
    s = pd.Series([10.0, 99.0], index=pd.DatetimeIndex(["2012-01-02", "2012-01-05"]))
    out = align_to_calendar(s, cal, max_stale_days=5)
    assert out["2012-01-03"] == 10.0          # lấy giá trước
    assert out["2012-01-04"] == 10.0          # chưa được dùng 99 của ngày 05
    assert out["2012-01-05"] == 99.0


def test_align_drops_stale_values():
    cal = pd.DatetimeIndex(["2012-01-02", "2012-01-20"])
    s = pd.Series([10.0], index=pd.DatetimeIndex(["2012-01-02"]))
    out = align_to_calendar(s, cal, max_stale_days=5)
    assert out["2012-01-02"] == 10.0
    assert np.isnan(out["2012-01-20"])        # cũ hơn 5 ngày -> NaN


def test_lagged_variable_uses_previous_session():
    raw, cfg = make_raw(), make_cfg()
    out = build_dataset(raw, cfg, apply_lags=True)
    assert out["ovx"].iloc[5] == pytest.approx(raw["ovx"].iloc[4])
    assert np.isnan(out["ovx"].iloc[0])


def test_target_and_unlagged_vars_are_same_day():
    raw, cfg = make_raw(), make_cfg()
    out = build_dataset(raw, cfg, apply_lags=True)
    assert out["wti"].equals(raw["wti"])
    assert out["brent"].iloc[7] == pytest.approx(raw["brent"].iloc[7])


def test_no_lag_mode_is_same_day():
    raw, cfg = make_raw(), make_cfg()
    out = build_dataset(raw, cfg, apply_lags=False)
    assert out["ovx"].iloc[5] == pytest.approx(raw["ovx"].iloc[5])


def test_cannot_lag_target():
    cfg = make_cfg()
    cfg["data"]["lag_rules"]["wti"] = 1
    with pytest.raises(ValueError):
        build_dataset(make_raw(), cfg)


def test_negative_price_handling_and_exclusion_flag():
    days = pd.bdate_range("2020-04-15", periods=8)
    raw = pd.DataFrame({"wti": [20, 18, 10, -37, 10, 12, 13, 14.0],
                        "brent": 25.0, "ovx": 80.0, "fed_funds": 0.05}, index=days)
    out = build_dataset(raw, make_cfg())
    neg_day = days[3]
    assert out.loc[neg_day, "flag_nonpositive"]
    assert np.isnan(out.loc[neg_day, "wti_logret"])      # không log giá âm
    assert np.isfinite(out.loc[neg_day, "wti_diff"])     # chênh lệch giá vẫn hợp lệ
    assert out["is_excluded"].all()                      # nằm trong cửa sổ loại
    assert out["flag_jump"].loc[neg_day]
