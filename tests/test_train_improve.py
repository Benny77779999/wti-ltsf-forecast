"""Test (cần torch) cho khởi tạo 0, weight decay và chọn weight decay trên validation."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.dataset import TARGET
from src.improvements import shrink_result
from src.models import Hybrid, NLinear, build_model
from src.train import run_fold, run_fold_tuned, save_predictions, train_one
from src.evaluate import load_prediction_file
from tests.test_train import WF, setup


def test_zero_init_models_start_exactly_at_naive():
    x = torch.randn(6, 30)
    m = NLinear(30, 5, zero_init=True)
    torch.testing.assert_close(m(x), x[:, -1:].repeat(1, 5))
    x3 = torch.randn(4, 30, 3)
    h = Hybrid(30, 5, n_channels=3, level_branch=True, zero_init=True)
    torch.testing.assert_close(h(x3), x3[:, -1, 0:1].repeat(1, 5))
    with pytest.raises(ValueError):
        build_model("linear", 30, 5, zero_init=True)
    with pytest.raises(ValueError):
        build_model("dlinear", 30, 5, zero_init=True)


def test_weight_decay_shrinks_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 30)).astype(np.float32).cumsum(axis=1) * 0.1
    Y = (X[:, -1:] + 0.3 * rng.normal(size=(400, 5))).astype(np.float32)
    norms = []
    for wd in (0.0, 1.0):
        m = NLinear(30, 5, zero_init=True)
        m, *_ = train_one(m, X[:300], Y[:300], X[300:], Y[300:], epochs=15, lr=0.01, seed=0, weight_decay=wd)
        norms.append(float(m.linear.weight.norm()))
    assert norms[1] < norms[0]


def test_tuned_run_returns_valid_choice_and_validation_outputs(tmp_path):
    df, ws, fold = setup()
    grid = [0.0, 0.1, 1.0]
    r = run_fold_tuned(df, ws, fold, "nlinear", WF, grid, seed=0, epochs=3)
    assert r["chosen_wd"] in grid
    assert r["val_pred_log"].shape == (len(fold.val), 5) and r["val_last_log"].shape == (len(fold.val),)
    assert r["pred_log"].shape == (len(fold.test), 5)
    # giá trị chọn phải là giá trị có val loss nhỏ nhất trong lưới
    vals = {wd: run_fold(df, ws, fold, "nlinear", WF, seed=0, epochs=3, weight_decay=wd,
                         zero_init=True)["best_val_loss"] for wd in grid}
    assert r["chosen_wd"] == min(vals, key=vals.get)
    sh = shrink_result(r)
    assert 0.0 <= sh["shrink_lambda"] <= 1.0
    save_predictions(tmp_path / "x.npz", sh)
    back = load_prediction_file(tmp_path / "x.npz")
    assert float(back["chosen_wd"]) == r["chosen_wd"] and "val_pred_log" in back


def test_tuned_run_is_reproducible():
    df, ws, fold = setup(cols=(TARGET, "x2"))
    a = run_fold_tuned(df, ws, fold, "hybrid", WF, [0.0, 0.1], seed=2, epochs=2)
    b = run_fold_tuned(df, ws, fold, "hybrid", WF, [0.0, 0.1], seed=2, epochs=2)
    np.testing.assert_allclose(a["pred_log"], b["pred_log"])
    assert a["chosen_wd"] == b["chosen_wd"]
