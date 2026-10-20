"""Cải tiến vòng 1: co dự báo về naive (forecast combination), chỉ dùng validation để chọn hệ số.

Ý tưởng: dự báo cuối = giá cuối + lambda * (dự báo mô hình - giá cuối), 0 <= lambda <= 1.
lambda = 0 là naive, lambda = 1 là mô hình gốc. lambda được ước lượng bằng bình phương tối thiểu trên
VALIDATION của từng fold (một hệ số chung cho mọi bước tau), không dùng test.

Lưu ý: epoch tốt nhất của mô hình cũng được chọn trên validation, nên lambda có thể hơi lạc quan.
"""
from __future__ import annotations

import numpy as np


def fit_shrink_lambda(val_pred: np.ndarray, val_actual: np.ndarray, val_last: np.ndarray,
                      lo: float = 0.0, hi: float = 1.0) -> float:
    d_hat = np.asarray(val_pred, float) - np.asarray(val_last, float)[:, None]
    d = np.asarray(val_actual, float) - np.asarray(val_last, float)[:, None]
    denom = float(np.sum(d_hat ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.sum(d_hat * d) / denom, lo, hi))


def apply_shrink(pred: np.ndarray, last: np.ndarray, lam: float) -> np.ndarray:
    last = np.asarray(last, float)[:, None]
    return last + lam * (np.asarray(pred, float) - last)


def shrink_result(result: dict) -> dict:
    """Bản sao của kết quả với dự báo test đã co về naive; lưu thêm shrink_lambda."""
    lam = fit_shrink_lambda(result["val_pred_log"], result["val_actual_log"], result["val_last_log"])
    out = dict(result)
    out["pred_log"] = apply_shrink(result["pred_log"], result["last_log"], lam)
    out["shrink_lambda"] = lam
    return out
