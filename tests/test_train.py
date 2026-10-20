"""Test khói cho huấn luyện (CPU, vài epoch, dữ liệu giả lập)."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.dataset import TARGET, add_log_price, make_folds, make_windows
from src.evaluate import load_prediction_file
from src.train import run_baseline, run_fold, save_predictions

WF = {"test_years": [2017, 2018], "train_start": "2010-01-01", "val_samples": 100,
      "min_train_samples": 300, "min_test_samples": 30}


def make_df(seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2008-01-01", "2019-12-31")
    wti = 60 * np.exp(rng.normal(0, 0.02, len(days)).cumsum())
    df = pd.DataFrame({"wti": wti, "x2": rng.normal(size=len(days)),
                       "x3": rng.normal(size=len(days))}, index=days)
    df["wti_logret"] = np.log(df["wti"]).diff()
    df["is_excluded"] = False
    return add_log_price(df)


def setup(cols=(TARGET,), L=30):
    df = make_df()
    ws = make_windows(df, L, 5, list(cols), study_start="2010-01-01")
    folds, _ = make_folds(ws, WF)
    assert folds
    return df, ws, folds[0]


@pytest.mark.parametrize("name", ["linear", "nlinear", "dlinear", "hybrid"])
def test_each_model_runs_and_outputs_valid(name):
    df, ws, fold = setup()
    r = run_fold(df, ws, fold, name, WF, seed=0, epochs=2)
    assert r["pred_log"].shape == (len(fold.test), 5)
    assert np.isfinite(r["pred_log"]).all()
    np.testing.assert_allclose(r["actual_log"], ws.Y[fold.test])
    assert len(r["val_loss"]) == 2 and 0 <= r["best_epoch"] < 2


def test_multichannel_hybrid_runs():
    df, ws, fold = setup(cols=(TARGET, "x2", "x3"))
    r = run_fold(df, ws, fold, "hybrid", WF, seed=0, epochs=2)
    assert r["pred_log"].shape == (len(fold.test), 5) and np.isfinite(r["pred_log"]).all()


def test_same_seed_is_reproducible():
    df, ws, fold = setup()
    a = run_fold(df, ws, fold, "dlinear", WF, seed=3, epochs=3)
    b = run_fold(df, ws, fold, "dlinear", WF, seed=3, epochs=3)
    np.testing.assert_allclose(a["pred_log"], b["pred_log"])


def test_scaler_does_not_see_the_future():
    df, ws, fold = setup(cols=(TARGET, "x2"))
    a = run_fold(df, ws, fold, "hybrid", WF, seed=1, epochs=2)
    df2 = df.copy()
    df2.loc[df2.index > fold.train_end_date, [TARGET, "x2"]] = 1e3      # đổi dữ liệu SAU giai đoạn train
    b = run_fold(df2, ws, fold, "hybrid", WF, seed=1, epochs=2)
    np.testing.assert_allclose(a["pred_log"], b["pred_log"])


def test_baselines_and_save_roundtrip(tmp_path):
    df, ws, fold = setup()
    nv = run_baseline(df, ws, fold, "naive", WF)
    np.testing.assert_allclose(nv["pred_log"], np.repeat(nv["last_log"][:, None], 5, axis=1))
    dr = run_baseline(df, ws, fold, "drift", WF)
    assert np.isfinite(dr["pred_log"]).all()
    save_predictions(tmp_path / "x.npz", nv)
    back = load_prediction_file(tmp_path / "x.npz")
    assert back["origin"].dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(back["origin"], nv["origin"])
    np.testing.assert_allclose(back["pred_log"], nv["pred_log"])
