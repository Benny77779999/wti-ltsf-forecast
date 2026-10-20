import numpy as np
import pytest

from app.predictor import label
from src.experiments import build_improve_specs, spec_outputs
from src.improvements import apply_shrink, fit_shrink_lambda, shrink_result
from src.report import default_pairs

import pandas as pd


def make_val(rng, n=400, signal=1.0):
    last = rng.normal(4, 0.1, n)
    d = rng.normal(0, 0.05, (n, 5)).cumsum(axis=1)
    actual = last[:, None] + d
    pred = last[:, None] + signal * d + rng.normal(0, 0.0, (n, 5))
    return pred, actual, last


def test_lambda_is_one_for_perfect_and_zero_for_useless_or_inverse():
    rng = np.random.default_rng(0)
    pred, actual, last = make_val(rng)
    assert fit_shrink_lambda(pred, actual, last) == pytest.approx(1.0)
    noise_pred = last[:, None] + rng.normal(0, 0.05, actual.shape)           # không liên quan
    assert fit_shrink_lambda(noise_pred, actual, last) < 0.15
    inverse = last[:, None] - (actual - last[:, None])                         # đảo dấu -> bị chặn ở 0
    assert fit_shrink_lambda(inverse, actual, last) == 0.0
    assert fit_shrink_lambda(np.repeat(last[:, None], 5, 1), actual, last) == 0.0   # dự báo = naive


def test_lambda_recovers_known_scale_and_is_clipped():
    rng = np.random.default_rng(1)
    last = rng.normal(4, 0.1, 500)
    d = rng.normal(0, 0.05, (500, 5))
    actual = last[:, None] + d
    pred = last[:, None] + 2.0 * d                       # mô hình dự báo quá đà gấp đôi
    assert fit_shrink_lambda(pred, actual, last) == pytest.approx(0.5, abs=1e-6)
    pred_weak = last[:, None] + 0.25 * d                 # quá yếu: hệ số tối ưu 4 bị chặn ở 1
    assert fit_shrink_lambda(pred_weak, actual, last) == 1.0


def test_apply_shrink_endpoints():
    rng = np.random.default_rng(2)
    pred, actual, last = make_val(rng, 50)
    np.testing.assert_allclose(apply_shrink(pred, last, 1.0), pred)
    np.testing.assert_allclose(apply_shrink(pred, last, 0.0), np.repeat(last[:, None], 5, 1))


def test_shrink_result_uses_validation_only_and_keeps_keys():
    rng = np.random.default_rng(3)
    vp, va, vl = make_val(rng, 200, signal=0.5)
    tp, ta, tl = make_val(rng, 100, signal=0.5)
    res = {"pred_log": tp, "actual_log": ta, "last_log": tl, "val_pred_log": vp, "val_actual_log": va,
           "val_last_log": vl, "origin": np.arange(100)}
    out = shrink_result(res)
    assert out["shrink_lambda"] == fit_shrink_lambda(vp, va, vl)       # chỉ phụ thuộc validation
    np.testing.assert_array_equal(out["actual_log"], ta)
    assert set(res) <= set(out)
    res2 = dict(res); res2["actual_log"] = ta + 100.0                  # đổi nhãn test không đổi lambda
    assert shrink_result(res2)["shrink_lambda"] == out["shrink_lambda"]


def test_improve_specs_and_outputs():
    exp = {"feature_groups": {"usd": ["usdcad_log", "usd_broad_log"]}, "improve1": {"wd_grid": [0.0, 0.1]}}
    specs = build_improve_specs(exp)
    assert [s["name"] for s in specs] == ["nlinear_zr", "hybrid_B_zr", "hybrid_C_usd_zr"]
    assert spec_outputs(specs[0]) == ["nlinear_zr", "nlinear_zr_shrink"]
    assert specs[2]["cols"] == ["wti_log", "usdcad_log", "usd_broad_log"]
    assert all(s["zero_init"] and s["wd_grid"] == [0.0, 0.1] for s in specs)


def test_default_pairs_include_variant_vs_base_comparisons():
    names = ["naive", "nlinear", "hybrid_B", "nlinear_zr", "nlinear_zr_shrink", "hybrid_B_zr",
             "hybrid_C_usd", "hybrid_C_usd_zr"]
    avg = pd.DataFrame({"model": names, "ratio_vs_naive": np.linspace(1, 1.1, len(names))})
    pairs = default_pairs(names, avg)
    assert ("nlinear_zr", "nlinear") in pairs and ("nlinear_zr_shrink", "nlinear_zr") in pairs
    assert ("nlinear_zr_shrink", "nlinear") in pairs and ("hybrid_B_zr", "hybrid_B") in pairs
    assert ("hybrid_C_usd_zr", "hybrid_C_usd") in pairs and ("nlinear_zr", "naive") in pairs


def test_labels_for_new_variants():
    assert label("nlinear_zr") == "NLinear (khởi tạo 0 + ridge)"
    assert label("hybrid_B_zr_shrink") == "Lai NLinear+DLinear (B) (khởi tạo 0 + ridge) + co về naive"
    assert label("hybrid_C_usd_zr") == "Lai đa biến (C): usd (khởi tạo 0 + ridge)"
