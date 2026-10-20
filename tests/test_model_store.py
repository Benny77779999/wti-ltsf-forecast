import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.dataset import add_log_price
from src.live_predict import LivePredictError
from src.model_store import available_models, list_sets, predict_saved, train_all
from tests.test_live_predict import CFG, EXP, WF, make_df


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    d = tmp_path_factory.mktemp("store")
    live = {"store": {"dir": str(d), "windows": [10], "cutoffs": ["2015-12-31", "last"], "seed": 0}}
    df = make_df()
    dirs = train_all(df, CFG, WF, EXP, live, names=["nlinear", "nlinear_mv", "hybrid_B", "hybrid_C_usd", "nlinear_zr"],
                     log=lambda m: None)
    return df, live, dirs


def test_all_selected_models_trained_and_saved_with_manifest(store):
    df, live, dirs = store
    assert len(dirs) == 2 and all((d / "manifest.json").exists() for d in dirs)
    sets = list_sets(live)
    assert {s["cutoff"] for s in sets} == {"2015-12-31", str(df.index[-1].date())}
    man = sets[-1]
    assert {m["name"] for m in man["models"]} == {"nlinear", "nlinear_mv", "hybrid_B", "hybrid_C_usd", "nlinear_zr"}
    assert all((man["dir"] / m["file"]).exists() for m in man["models"])
    assert available_models(man, 10)[:2] == ["naive", "drift"]


def test_rerun_skips_existing_models(store):
    df, live, dirs = store
    logs = []
    train_all(df, CFG, WF, EXP, live, names=["nlinear"], log=logs.append)
    assert not any("đã lưu" in m for m in logs)


def test_saved_model_forecast_matches_status_and_ignores_future(store):
    df, live, _ = store
    man = [s for s in list_sets(live) if s["cutoff"] == "2015-12-31"][0]
    r_in = predict_saved(df, man, EXP, "nlinear", 10, "2014-06-10")
    r_out = predict_saved(df, man, EXP, "nlinear", 10, "2016-06-10")
    assert r_in["in_sample"] and not r_out["in_sample"]
    assert r_out["forecast"].shape == (5,) and np.isfinite(r_out["forecast"]).all()
    pos = df.index.get_loc(pd.Timestamp("2016-06-10"))
    df2 = df.copy()
    df2.iloc[pos + 1:, [df2.columns.get_loc(c) for c in ("wti", "usdcad", "ovx")]] *= 50.0
    df2 = add_log_price(df2.drop(columns=["wti_log"]))
    again = predict_saved(df2, man, EXP, "nlinear", 10, "2016-06-10")
    np.testing.assert_allclose(r_out["forecast"], again["forecast"])        # chỉ dùng cửa sổ quá khứ


def test_multivariate_saved_model_and_baselines_and_errors(store):
    df, live, _ = store
    man = list_sets(live)[0]
    mv = predict_saved(df, man, EXP, "nlinear_mv", 10, "2017-06-01")
    assert mv["input_columns"] == ["wti_log", "spread_brent_wti", "usdcad_log", "usd_broad_log"] and np.isfinite(mv["forecast"]).all()
    nv = predict_saved(df, man, EXP, "naive", 10, "2017-06-01")
    np.testing.assert_allclose(nv["forecast"], nv["last_price"])
    assert predict_saved(df, man, EXP, "drift", 10, "2017-06-01")["info"]["drift_per_session"] == man["drift_per_session"]
    with pytest.raises(LivePredictError):
        predict_saved(df, man, EXP, "nlinear", 99, "2017-06-01")             # chưa lưu L=99
    with pytest.raises(LivePredictError):
        predict_saved(df, man, EXP, "khong_co", 10, "2017-06-01")


def test_status_in_sample_partial_and_future(store):
    df, live, _ = store
    sets = {s["cutoff"]: s for s in list_sets(live)}
    early = sets["2015-12-31"]
    assert predict_saved(df, early, EXP, "nlinear", 10, "2015-06-10")["status"] == "trong mẫu"
    assert predict_saved(df, early, EXP, "nlinear", 10, "2015-12-29")["status"] == "một phần"
    assert predict_saved(df, early, EXP, "nlinear", 10, "2016-01-04")["status"] == "ngoài mẫu"
    last = sets[str(df.index[-1].date())]
    assert predict_saved(df, last, EXP, "nlinear", 10, df.index[-1])["status"] == "ngoài mẫu"     # dự báo cho tương lai
