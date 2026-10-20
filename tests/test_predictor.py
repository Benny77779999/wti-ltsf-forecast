import numpy as np
import pandas as pd
import pytest

from app.predictor import Forecasts, future_dates, history_window, label

T = 5


def make_project(tmp_path):
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2022-01-03", "2023-12-29")
    df = pd.DataFrame({"wti": 70 * np.exp(rng.normal(0, 0.02, len(days)).cumsum())}, index=days)
    pdir = tmp_path / "preds"
    for year in (2022, 2023):
        pos = np.where(days.year == year)[0]
        pos = pos[(pos >= 40) & (pos < len(days) - T)]
        origin = days[pos]
        logp = np.log(df["wti"].to_numpy())
        last = logp[pos]
        actual = np.stack([logp[pos + j + 1] for j in range(T)], axis=1)
        for name, seeds, shift in (("naive", [0], 0.0), ("hybrid_B", [0, 1], 0.01), ("hybrid_C_usd_lvl", [0], 0.02)):
            for s in seeds:
                pred = np.repeat(last[:, None], T, 1) + shift * (s + 1)
                d = pdir / "L30"; d.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(d / f"{name}__{year}__s{s}.npz",
                                    origin=origin.values.astype("datetime64[ns]").astype("int64"),
                                    pred_log=pred, actual_log=actual, last_log=last)
    return df, pdir


def test_discovery_and_labels(tmp_path):
    df, pdir = make_project(tmp_path)
    fc = Forecasts(pdir)
    assert fc.windows() == [30] and fc.years(30) == [2022, 2023]
    assert fc.models(30) == ["naive", "hybrid_B", "hybrid_C_usd_lvl"]
    assert label("hybrid_C_usd_lvl") == "Lai đa biến (C): usd + nhánh mức"
    assert label("nlinear") == "NLinear"


def test_forecast_values_seed_average_and_conversion(tmp_path):
    df, pdir = make_project(tmp_path)
    fc = Forecasts(pdir)
    origin = fc.origins(30, 2023)[10]
    n = fc.forecast_at(30, "naive", origin)
    np.testing.assert_allclose(n["pred"], n["last"])                      # naive = giá cuối
    assert n["last"] == pytest.approx(df["wti"].loc[origin])              # log -> USD khớp dữ liệu gốc
    h = fc.forecast_at(30, "hybrid_B", origin)
    assert h["n_seeds"] == 2
    np.testing.assert_allclose(h["pred"], n["last"] * np.exp(0.015))      # trung bình của 0.01 và 0.02
    fut = future_dates(df, origin, 5)
    np.testing.assert_allclose(n["actual"], df["wti"].loc[fut].to_numpy())  # giá thực khớp 5 phiên kế tiếp


def test_table_and_missing_inputs(tmp_path):
    df, pdir = make_project(tmp_path)
    fc = Forecasts(pdir)
    origin = fc.origins(30, 2022)[5]
    t = fc.forecast_table(30, origin, ["naive", "hybrid_B", "does_not_exist"])
    assert list(t["_key"]) == ["naive", "hybrid_B"] and "sai số TB (%)" in t.columns
    assert fc.forecast_at(30, "naive", pd.Timestamp("2030-01-01")) is None
    h = history_window(df, origin, 30)
    assert len(h) == 30 and h.index[-1] == origin
