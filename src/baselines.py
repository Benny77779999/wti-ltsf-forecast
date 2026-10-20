"""Baseline không học: naive và random walk có drift (dùng trên log-giá CHƯA chuẩn hoá)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(X: np.ndarray, T: int) -> np.ndarray:
    """Giữ nguyên log-giá cuối của cửa sổ. X: [N, L, C] (kênh 0 = log-giá) -> [N, T]."""
    return np.repeat(X[:, -1, 0][:, None], T, axis=1)


def estimate_drift(df: pd.DataFrame, start, end) -> float:
    """Trung bình log-return mỗi phiên trong giai đoạn TRAIN (bỏ giai đoạn loại)."""
    idx = df.index
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end)) & ~df["is_excluded"].to_numpy(bool)
    r = df.loc[mask, "wti_logret"].dropna()
    if len(r) < 30:
        raise ValueError("Quá ít dữ liệu để ước lượng drift.")
    return float(r.mean())


def drift_forecast(X: np.ndarray, T: int, mu: float) -> np.ndarray:
    """Random walk có drift: log-giá cuối + tau * mu."""
    return naive_forecast(X, T) + mu * np.arange(1, T + 1)[None, :]
