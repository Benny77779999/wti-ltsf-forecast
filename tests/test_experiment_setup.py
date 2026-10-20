import numpy as np
import pandas as pd
import pytest

from src.dataset import TARGET, add_log_price, make_windows, select_channels
from src.experiments import all_channel_columns, build_specs, fold_table, parse_windows, prediction_path

EXP = {"feature_groups": {"spread": ["spread_brent_wti"], "usd": ["usdcad_log", "usd_broad_log"],
                          "ovx": ["ovx"], "fed": ["fed_funds"]},
       "level_branch_variants": [False, True]}
WF = {"main_windows": [5, 30, 120], "supplementary_windows": [480]}


def test_specs_cover_baselines_A_B_C_with_level_variants():
    specs = build_specs(EXP)
    names = [s["name"] for s in specs]
    assert names[:2] == ["naive", "drift"]
    assert {"linear", "nlinear", "dlinear", "hybrid_B"} <= set(names)
    assert len(names) == 2 + 3 + 3 + 1 + 5 * 2                     # 4 nhóm + 'all', mỗi nhóm 2 biến thể
    assert len(set(names)) == len(names)
    allspec = next(s for s in specs if s["name"] == "hybrid_C_all_lvl")
    assert allspec["cols"] == [TARGET, "spread_brent_wti", "usdcad_log", "usd_broad_log", "ovx", "fed_funds"]
    assert allspec["level"] is True
    assert all(s["cols"][0] == TARGET for s in specs)


def test_group_filter_and_all_columns():
    assert all_channel_columns(EXP) == ["spread_brent_wti", "usdcad_log", "usd_broad_log", "ovx", "fed_funds"]
    names = [s["name"] for s in build_specs(EXP, groups=["ovx"])]
    assert "hybrid_C_ovx" in names and "hybrid_C_usd" not in names and "hybrid_C_all" not in names


def test_parse_windows_and_path(tmp_path):
    assert parse_windows(None, WF) == [5, 30, 120]
    assert parse_windows(["supp"], WF) == [480]
    assert parse_windows(["all"], WF) == [5, 30, 120, 480]
    assert parse_windows(["30", "120"], WF) == [30, 120]
    p = prediction_path(tmp_path, 30, "hybrid_B", 2023, 2)
    assert p.parts[-2:] == ("L30", "hybrid_B__2023__s2.npz")


def make_df():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2008-01-01", "2013-12-31")
    n = len(days)
    df = pd.DataFrame({"wti": 60 * np.exp(rng.normal(0, .02, n).cumsum()),
                       "usdcad": 1.2 + 0.01 * rng.normal(size=n).cumsum() * 0.1,
                       "usd_broad": 100 + rng.normal(size=n).cumsum() * 0.1,
                       "ovx": 30 + rng.normal(size=n), "fed_funds": 0.1,
                       "spread_brent_wti": 5 + rng.normal(size=n)}, index=days)
    df["is_excluded"] = False
    return add_log_price(df)


def test_add_log_price_creates_channel_columns():
    df = make_df()
    assert {TARGET, "usdcad_log", "usd_broad_log"} <= set(df.columns)
    np.testing.assert_allclose(df["usdcad_log"], np.log(df["usdcad"]))


def test_select_channels_keeps_same_samples_and_picks_right_columns():
    df = make_df()
    cols = [TARGET] + all_channel_columns(EXP)
    ws = make_windows(df, 30, 5, cols, study_start="2010-01-01")
    sub = select_channels(ws, [TARGET, "ovx"])
    assert sub.X.shape == (len(ws), 30, 2) and sub.columns == [TARGET, "ovx"]
    np.testing.assert_array_equal(sub.X[:, :, 0], ws.X[:, :, 0])
    np.testing.assert_array_equal(sub.X[:, :, 1], ws.X[:, :, cols.index("ovx")])
    np.testing.assert_array_equal(sub.origin.values, ws.origin.values)
    with pytest.raises(ValueError):
        select_channels(ws, ["ovx", TARGET])


def test_fold_table_marks_skipped():
    from src.dataset import Fold
    f = Fold(2018, np.arange(10), np.arange(3), np.arange(5), pd.Timestamp("2017-12-29"))
    t = fold_table([f], [(2017, "chỉ 718 mẫu trước test")])
    assert list(t["test_year"]) == [2017, 2018]
    assert t.loc[0, "status"].startswith("skipped") and t.loc[1, "n_test"] == 5
