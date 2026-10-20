import numpy as np
import pandas as pd
import pytest

from src.dataset import (TARGET, ChannelScaler, add_log_price, make_folds, make_windows,
                         window_availability)
from src.preprocessing import build_dataset, missing_business_days
from src.features import build_calendar_features, days_to_expiry

WF = {"test_years": [2017, 2018, 2019, 2020, 2021], "train_start": "2010-01-01",
      "val_samples": 100, "min_train_samples": 300, "min_test_samples": 30}


def make_df(seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2008-01-01", "2022-12-30")
    wti = 70 * np.exp(rng.normal(0, 0.02, len(days)).cumsum() * 0.3)
    df = pd.DataFrame({"wti": wti, "x2": rng.normal(size=len(days))}, index=days)
    df.loc["2020-04-20", "wti"] = -37.0
    df["is_excluded"] = (days >= "2020-03-01") & (days <= "2020-06-30")
    return add_log_price(df)


def test_windows_match_source_and_skip_excluded():
    df = make_df()
    ws = make_windows(df, L=30, T=5, feature_cols=[TARGET, "x2"], study_start="2010-01-01")
    pos = {d: i for i, d in enumerate(df.index)}
    for k in (0, len(ws) // 2, len(ws) - 1):
        i = pos[ws.origin[k]]
        np.testing.assert_allclose(ws.X[k, :, 0], df[TARGET].to_numpy()[i - 29:i + 1], rtol=1e-6)
        np.testing.assert_allclose(ws.Y[k], df[TARGET].to_numpy()[i + 1:i + 6], rtol=1e-6)
        assert df.index[i + 5] == ws.target_end[k]
    # không mẫu nào chạm giai đoạn loại
    ex = df.index[df["is_excluded"].to_numpy()]
    for k in range(len(ws)):
        i = pos[ws.origin[k]]
        span = df.index[i - 29:i + 6]
        assert not span.isin(ex).any()
    assert not np.isnan(ws.X).any() and not np.isnan(ws.Y).any()


def test_origin_not_before_study_start_but_input_may_use_buffer():
    df = make_df()
    ws = make_windows(df, L=120, T=5, study_start="2010-01-01")
    assert ws.origin.min() >= pd.Timestamp("2010-01-01")
    assert ws.origin.min() < pd.Timestamp("2010-02-01")   # cửa sổ dùng đệm 2008-2009


def test_availability_shows_2020_hole():
    df = make_df()
    av = window_availability(df, L=120, T=5, study_start="2010-01-01")
    assert av.loc[2020] < av.loc[2019] * 0.5


def test_folds_are_ordered_and_purged():
    df = make_df()
    ws = make_windows(df, L=30, T=5, study_start="2010-01-01")
    folds, skipped = make_folds(ws, WF)
    assert folds
    for f in folds:
        assert len(set(f.train) & set(f.val)) == 0 and len(set(f.val) & set(f.test)) == 0
        first_val, first_test = ws.origin[f.val[0]], ws.origin[f.test[0]]
        assert ws.target_end[f.train].max() <= first_val       # purge train -> val
        assert ws.target_end[f.val].max() <= first_test        # purge val -> test
        assert ws.origin[f.train].max() < first_val < first_test
        assert f.train_end_date <= first_test


def test_scaler_ignores_future_and_excluded_data():
    df = make_df()
    end = "2015-12-31"
    s1 = ChannelScaler().fit(df, [TARGET], "2010-01-01", end)
    df2 = df.copy()
    df2.loc[df2.index > end, TARGET] = 999.0              # thay đổi tương lai
    s2 = ChannelScaler().fit(df2, [TARGET], "2010-01-01", end)
    np.testing.assert_allclose(s1.mean, s2.mean)
    np.testing.assert_allclose(s1.std, s2.std)
    y = np.array([[1.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(s1.inverse_target(s1.transform_target(y)), y, rtol=1e-5)


def test_calendar_features():
    idx = pd.bdate_range("2024-06-17", "2024-06-21")        # tuần có Juneteenth (thứ Tư 19/6)
    f = build_calendar_features(idx, {"fomc_dates_file": "does_not_exist.csv", "near_expiry_days": 5})
    assert f.loc["2024-06-20", "is_eia_day"] == 1.0         # dời sang thứ Năm
    assert f.loc["2024-06-19", "is_eia_day"] == 0.0
    assert (f[[f"dow_{k}" for k in range(5)]].sum(axis=1) == 1).all()
    d = days_to_expiry(pd.bdate_range("2023-01-02", "2023-12-29"))
    assert (d >= 0).all()
    assert (d.diff().dropna() <= 0).sum() > 0.9 * len(d)    # giảm dần giữa các lần đáo hạn
    assert (d.diff() > 0).sum() >= 10                       # reset khi qua hợp đồng mới


def test_windows_do_not_span_data_gaps():
    df = make_df()
    df["gap_before"] = False
    g = df.index.get_loc(pd.Timestamp("2015-03-10"))
    df.iloc[g, df.columns.get_loc("gap_before")] = True
    ws = make_windows(df, L=30, T=5, study_start="2010-01-01")
    pos = {d: i for i, d in enumerate(df.index)}
    for o in ws.origin:
        assert not (pos[o] - 29 < g <= pos[o] + 5)
    assert df.index[g - 6] in ws.origin        # kết thúc ngay trước đoạn thiếu: hợp lệ
    assert df.index[g - 5] not in ws.origin    # chạm dòng sau đoạn thiếu: bị loại
    assert df.index[g + 29] in ws.origin       # đoạn thiếu nằm ngay trước dòng đầu cửa sổ: hợp lệ


def test_build_dataset_flags_gap_and_missing_days_counted():
    from tests.test_leakage import make_cfg, make_raw

    raw = make_raw(60)
    raw.iloc[20:28, raw.columns.get_loc("wti")] = np.nan          # 8 phiên thiếu
    out = build_dataset(raw, make_cfg())
    assert out["gap_before"].sum() == 1
    assert out["gap_before"].idxmax() == raw.index[28]
    m = missing_business_days(out.index)
    assert m.sum() >= 8
