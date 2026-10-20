"""Kiểm thử chức năng dự đoán thử một ngày bất kỳ (mô hình thật sự chạy)."""
import numpy as np
import pandas as pd
import pytest

from src.dataset import add_log_price
from src.live_predict import (LivePredictError, build_training, model_catalog, next_session_dates,
                              predict_day, resolve_origin, window_problem)

CFG = {"data": {"study_start": "2010-01-01"}}
WF = {"train_start": "2010-01-01", "val_samples": 100, "min_train_samples": 300}
EXP = {"seeds": 1, "epochs": 3, "lr": 0.001, "batch_size": 32, "moving_avg": 5,
       "feature_groups": {"spread": ["spread_brent_wti"], "usd": ["usdcad_log", "usd_broad_log"]},
       "level_branch_variants": [False], "improve1": {"wd_grid": [0.0, 0.1]}}


def make_df(seed=0, end="2017-12-29"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-01", end)
    n = len(days)
    df = pd.DataFrame({"wti": 60 * np.exp(rng.normal(0, 0.02, n).cumsum() * 0.5),
                       "usdcad": 1.2 + rng.normal(size=n).cumsum() * 0.002,
                       "usd_broad": 100 + rng.normal(size=n).cumsum() * 0.05,
                       "ovx": 30 + rng.normal(size=n), "fed_funds": 0.1,
                       "spread_brent_wti": 5 + rng.normal(size=n)}, index=days)
    df["wti_logret"] = np.log(df["wti"]).diff()
    df["is_excluded"] = False
    df["gap_before"] = False
    df.index.name = "date"
    return add_log_price(df)


def kw(**extra):
    return dict(exp=EXP, wf=WF, cfg=CFG, **extra)


# ---------------- không cần torch ----------------
def test_resolve_origin_snaps_to_previous_session():
    df = make_df()
    assert resolve_origin(df, "2015-06-13") == pd.Timestamp("2015-06-12")        # thứ Bảy -> thứ Sáu
    with pytest.raises(LivePredictError):
        resolve_origin(df, "2009-01-01")


def test_window_problem_detects_each_cause():
    df = make_df()
    o = pd.Timestamp("2015-06-12")
    cols = ["wti_log", "ovx"]
    assert window_problem(df, o, 30, cols) is None
    assert "ít hơn" in window_problem(df, pd.Timestamp("2010-01-08"), 30, cols)
    d1 = df.copy(); d1.loc["2015-06-01":"2015-06-03", "is_excluded"] = True
    assert "bị loại" in window_problem(d1, o, 30, cols)
    d2 = df.copy(); d2.loc["2015-06-02", "gap_before"] = True
    assert "thiếu dữ liệu của nguồn" in window_problem(d2, o, 30, cols)
    d3 = df.copy(); d3.loc["2015-06-02", "ovx"] = np.nan
    assert "ovx" in window_problem(d3, o, 30, cols)
    d4 = df.copy(); d4.loc["2015-05-01", "gap_before"] = True                    # nằm ngoài cửa sổ 30 phiên
    assert window_problem(d4, o, 30, cols) is None


def test_next_sessions_use_data_then_project_business_days():
    df = make_df()
    d, proj = next_session_dates(df, pd.Timestamp("2015-06-12"))
    assert not proj and list(d) == list(df.index[df.index.get_loc(pd.Timestamp("2015-06-12")) + 1:][:5])
    d2, proj2 = next_session_dates(df, df.index[-1])                             # hết dữ liệu
    assert proj2 and len(d2) == 5 and (d2 > df.index[-1]).all() and (d2.weekday < 5).all()
    assert d2[0] == pd.Timestamp("2018-01-02")                                   # 1/1/2018 là ngày lễ


def test_training_set_only_uses_data_up_to_origin_and_is_purged():
    df = make_df()
    origin = pd.Timestamp("2016-03-15")
    trunc = df.loc[:origin]
    ws, train, val = build_training(trunc, ["wti_log"], 30, WF, "2010-01-01")
    assert ws.target_end.max() <= origin
    assert ws.target_end[train].max() <= ws.origin[val[0]]
    assert len(val) == WF["val_samples"] and len(set(train) & set(val)) == 0
    with pytest.raises(LivePredictError):
        build_training(df.loc[:"2010-12-31"], ["wti_log"], 30, WF, "2010-01-01")


def test_catalog_has_all_model_families():
    cat = model_catalog(EXP)
    assert {"naive", "drift", "linear", "nlinear", "dlinear", "hybrid_B", "hybrid_C_usd", "nlinear_zr"} <= set(cat)


def test_naive_and_drift_need_no_training_and_ignore_the_future(tmp_path):
    df = make_df()
    r = predict_day(df, "naive", 30, "2015-06-12", **kw(cache_dir=tmp_path))
    np.testing.assert_allclose(r["forecast"], r["last_price"])
    assert r["last_price"] == pytest.approx(df["wti"].loc["2015-06-12"])
    pos = df.index.get_loc(pd.Timestamp("2015-06-12"))
    np.testing.assert_allclose(r["actual"], df["wti"].iloc[pos + 1:pos + 6].to_numpy())
    d1 = predict_day(df, "drift", 30, "2015-06-12", **kw(cache_dir=tmp_path))
    df2 = df.copy(); df2.iloc[pos + 1:, df2.columns.get_loc("wti")] = 1e6
    df2["wti_logret"] = np.log(df2["wti"]).diff()
    d2 = predict_day(df2, "drift", 30, "2015-06-12", **kw(cache_dir=tmp_path))
    np.testing.assert_allclose(d1["forecast"], d2["forecast"])                    # drift không nhìn tương lai
    assert not list(tmp_path.glob("*.pt"))                                         # baseline không tạo file mô hình


def test_unknown_model_and_bad_inputs(tmp_path):
    df = make_df()
    with pytest.raises(LivePredictError):
        predict_day(df, "khong_co", 30, "2015-06-12", **kw(cache_dir=tmp_path))
    with pytest.raises(LivePredictError):
        predict_day(df, "nlinear", 30, "2010-02-01", **kw(cache_dir=tmp_path))    # quá sớm: không đủ dữ liệu / cửa sổ


# ---------------- cần torch ----------------
torch = pytest.importorskip("torch")


def test_trained_model_forecasts_from_window_and_never_sees_future(tmp_path):
    df = make_df()
    a = predict_day(df, "nlinear", 30, "2015-06-12", **kw(cache_dir=tmp_path / "a", use_cache=False))
    assert a["forecast"].shape == (5,) and np.isfinite(a["forecast"]).all()
    assert a["info"]["n_train"] > 0 and a["info"]["train_end_date"] <= str(a["origin"].date())
    pos = df.index.get_loc(pd.Timestamp("2015-06-12"))
    df2 = df.copy()
    cols = ["wti", "usdcad", "usd_broad", "ovx", "spread_brent_wti"]
    df2.iloc[pos + 1:, [df2.columns.get_loc(c) for c in cols]] *= 50.0             # phá hoại toàn bộ tương lai
    df2["wti_logret"] = np.log(df2["wti"]).diff(); df2 = add_log_price(df2.drop(columns=["wti_log"]))
    b = predict_day(df2, "nlinear", 30, "2015-06-12", **kw(cache_dir=tmp_path / "b", use_cache=False))
    np.testing.assert_allclose(a["forecast"], b["forecast"])                      # dự báo không phụ thuộc tương lai


def test_same_seed_reproducible_and_cache_reuse_and_invalidation(tmp_path):
    df = make_df()
    r1 = predict_day(df, "dlinear", 20, "2014-09-10", seed=2, **kw(cache_dir=tmp_path))
    r2 = predict_day(df, "dlinear", 20, "2014-09-10", seed=2, **kw(cache_dir=tmp_path))
    assert not r1["from_cache"] and r2["from_cache"]
    np.testing.assert_allclose(r1["forecast"], r2["forecast"])
    r3 = predict_day(df, "dlinear", 20, "2014-09-10", seed=2, **kw(cache_dir=tmp_path / "x", use_cache=False))
    np.testing.assert_allclose(r1["forecast"], r3["forecast"], rtol=1e-5)          # huấn luyện lại cùng seed cho cùng kết quả
    df_changed = df.copy()
    df_changed.loc[:"2014-09-10", "wti"] *= 1.01
    df_changed = add_log_price(df_changed.drop(columns=["wti_log"]))
    r4 = predict_day(df_changed, "dlinear", 20, "2014-09-10", seed=2, **kw(cache_dir=tmp_path))
    assert not r4["from_cache"]                                                    # dữ liệu đổi -> không dùng cache cũ


def test_multichannel_tuned_and_invalid_window(tmp_path):
    df = make_df()
    r = predict_day(df, "hybrid_C_usd", 30, "2016-01-20", **kw(cache_dir=tmp_path))
    assert np.isfinite(r["forecast"]).all() and r["input_columns"] == ["wti_log", "usdcad_log", "usd_broad_log"]
    t = predict_day(df, "nlinear_zr", 10, "2016-01-20", **kw(cache_dir=tmp_path))
    assert t["info"]["chosen_wd"] in EXP["improve1"]["wd_grid"]
    bad = df.copy(); bad.loc["2016-01-05":"2016-01-08", "is_excluded"] = True
    with pytest.raises(LivePredictError):
        predict_day(bad, "nlinear", 30, "2016-01-20", **kw(cache_dir=tmp_path))
