import numpy as np
import pytest

from src.evaluate import (align_on_origins, diebold_mariano, directional_accuracy, dm_by_horizon,
                          mae_by_horizon, metrics_table, rmse_by_horizon)


def test_rmse_mae_known_values():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    actual = np.array([[0.0, 2.0], [5.0, 4.0]])
    np.testing.assert_allclose(rmse_by_horizon(pred, actual), [np.sqrt((1 + 4) / 2), 0.0])
    np.testing.assert_allclose(mae_by_horizon(pred, actual), [1.5, 0.0])


def test_directional_accuracy_and_naive_is_nan():
    last = np.array([0.0, 0.0, 0.0])
    actual = np.array([[1.0], [-1.0], [1.0]])
    pred = np.array([[2.0], [-2.0], [-2.0]])
    assert directional_accuracy(pred, actual, last)[0] == pytest.approx(2 / 3)
    assert np.isnan(directional_accuracy(np.zeros((3, 1)), actual, last)[0])


def test_metrics_table_ratio_vs_naive():
    rng = np.random.default_rng(0)
    actual = rng.normal(size=(200, 5)); last = np.zeros(200)
    naive = np.zeros((200, 5))
    t = metrics_table(naive, actual, last, naive_pred=naive)
    np.testing.assert_allclose(t["rmse_ratio_vs_naive"], 1.0)
    assert list(t["tau"]) == [1, 2, 3, 4, 5]


def test_dm_detects_clear_difference_and_sign():
    rng = np.random.default_rng(1)
    e_good = rng.normal(0, 1.0, 600)
    e_bad = rng.normal(0, 2.0, 600)
    r = diebold_mariano(e_bad, e_good, h=1)
    assert r["p_value"] < 0.001 and r["mean_diff"] > 0          # mô hình 1 tệ hơn
    r2 = diebold_mariano(e_good, e_bad, h=1)
    assert r2["p_value"] < 0.001 and r2["mean_diff"] < 0


def test_dm_does_not_reject_for_identical_quality():
    rng = np.random.default_rng(2)
    pvals = [diebold_mariano(rng.normal(size=400), rng.normal(size=400), h=3)["p_value"] for _ in range(60)]
    assert np.mean(np.array(pvals) < 0.05) < 0.2                # tỉ lệ bác bỏ sai gần mức danh nghĩa


def test_dm_handles_degenerate_and_small_samples():
    e = np.ones(50)
    assert np.isnan(diebold_mariano(e, e, h=1)["stat"])
    assert np.isnan(diebold_mariano(e[:5], e[:5], h=1)["p_value"])


def test_dm_by_horizon_shape():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(300, 5))
    df = dm_by_horizon(a + rng.normal(size=(300, 5)), a + 0.1 * rng.normal(size=(300, 5)), a)
    assert len(df) == 5 and list(df["tau"]) == [1, 2, 3, 4, 5]


def test_align_on_origins_keeps_common_sorted():
    o = lambda days: np.array(days, dtype="datetime64[D]").astype("datetime64[ns]")
    a = {"origin": o(["2020-01-01", "2020-01-02", "2020-01-03"]), "x": np.array([1, 2, 3])}
    b = {"origin": o(["2020-01-02", "2020-01-03", "2020-01-06"]), "x": np.array([20, 30, 60])}
    ra, rb = align_on_origins(a, b)
    np.testing.assert_array_equal(ra["x"], [2, 3]); np.testing.assert_array_equal(rb["x"], [20, 30])
