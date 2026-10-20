"""Đánh giá dự báo trên THAY ĐỔI so với log-giá đóng cuối (tức log-return tích luỹ), theo từng bước.

Mọi hàm nhận mảng đã ở thang log-giá chưa chuẩn hoá:
    pred   [N, T]  log-giá dự báo
    actual [N, T]  log-giá thực
    last   [N]     log-giá đóng cuối đã biết tại origin
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def forecast_errors(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return pred - actual


def rmse_by_horizon(pred, actual) -> np.ndarray:
    return np.sqrt(np.mean((pred - actual) ** 2, axis=0))


def mae_by_horizon(pred, actual) -> np.ndarray:
    return np.mean(np.abs(pred - actual), axis=0)


def directional_accuracy(pred, actual, last) -> np.ndarray:
    """Tỉ lệ đoán đúng chiều thay đổi so với giá cuối, theo bước. NaN nếu dự báo luôn bằng giá cuối."""
    p, a = np.sign(pred - last[:, None]), np.sign(actual - last[:, None])
    valid = p != 0
    hit = (p == a) & valid
    with np.errstate(invalid="ignore", divide="ignore"):
        return hit.sum(axis=0) / valid.sum(axis=0)


def metrics_table(pred, actual, last, naive_pred=None) -> pd.DataFrame:
    """Bảng chỉ số theo bước tau = 1..T; kèm tỉ lệ RMSE so với naive nếu có."""
    T = pred.shape[1]
    out = pd.DataFrame({
        "tau": np.arange(1, T + 1),
        "rmse": rmse_by_horizon(pred, actual),
        "mae": mae_by_horizon(pred, actual),
        "dir_acc": directional_accuracy(pred, actual, last),
    })
    if naive_pred is not None:
        out["rmse_ratio_vs_naive"] = out["rmse"] / rmse_by_horizon(naive_pred, actual)
    return out


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """Kiểm định Diebold-Mariano (hai phía, loss bình phương) có hiệu chỉnh Harvey-Leybourne-Newbold.

    e1, e2: sai số dự báo của hai mô hình trên CÙNG các mẫu, ở cùng một bước dự báo h.
    Phương sai dùng hiệu chỉnh tự tương quan với độ trễ h-1 vì các origin liên tiếp có sai số
    chồng lấn. H0: hai mô hình chính xác như nhau.
    mean_diff > 0 nghĩa là mô hình 1 có loss lớn hơn (tệ hơn) mô hình 2.
    """
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = len(d)
    if n < 10:
        return {"stat": np.nan, "p_value": np.nan, "mean_diff": float(np.mean(d)) if n else np.nan, "n": n}
    dbar = d.mean()
    dc = d - dbar
    var = np.dot(dc, dc) / n
    for k in range(1, h):
        var += 2.0 * np.dot(dc[k:], dc[:-k]) / n
    var /= n
    if var <= 0:
        return {"stat": np.nan, "p_value": np.nan, "mean_diff": float(dbar), "n": n}
    stat = dbar / np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)          # hiệu chỉnh HLN
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"stat": float(stat), "p_value": float(p), "mean_diff": float(dbar), "n": n}


def dm_by_horizon(pred1, pred2, actual) -> pd.DataFrame:
    """DM cho từng bước tau = 1..T. Hai bộ dự báo phải ứng với cùng các mẫu theo cùng thứ tự thời gian."""
    rows = []
    for j in range(actual.shape[1]):
        r = diebold_mariano(pred1[:, j] - actual[:, j], pred2[:, j] - actual[:, j], h=j + 1)
        rows.append({"tau": j + 1, **r})
    return pd.DataFrame(rows)


def load_prediction_file(path) -> dict:
    z = np.load(path)
    out = {k: z[k] for k in z.files}
    out["origin"] = out["origin"].astype("datetime64[ns]")
    return out


def align_on_origins(a: dict, b: dict):
    """Giữ các origin chung của hai kết quả, theo thứ tự thời gian (để so sánh công bằng)."""
    common, ia, ib = np.intersect1d(a["origin"], b["origin"], return_indices=True)
    pick = lambda r, i: {k: (v[i] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(r["origin"]) else v)
                         for k, v in r.items()}
    return pick(a, ia), pick(b, ib)
