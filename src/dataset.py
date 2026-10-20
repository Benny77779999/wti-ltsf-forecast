"""Cửa sổ trượt, chia walk-forward có purge, và scaler chỉ fit trên train.

Quy ước:
  * Chuỗi mô hình dùng là log-giá WTI (cột `wti_log`), luôn là kênh 0 của X.
  * Một mẫu có `origin` = ngày cuối của cửa sổ đầu vào; mục tiêu là T phiên sau origin.
  * Mẫu bị loại nếu BẤT KỲ phiên nào trong [origin-L+1, origin+T] thuộc giai đoạn loại
    (is_excluded), hoặc thiếu giá trị ở các kênh đang dùng, hoặc cửa sổ chạy qua một đoạn thiếu
    dữ liệu (cột gap_before).
  * Đầu vào được phép lùi vào giai đoạn đệm (trước study_start); origin thì không.
Module này chỉ dùng numpy/pandas (không cần torch).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TARGET = "wti_log"


def add_log_price(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm wti_log (kênh mục tiêu) và log của tỷ giá/chỉ số USD nếu có."""
    out = df.copy()
    out[TARGET] = np.log(out["wti"].where(out["wti"] > 0))
    for src, dst in (("usdcad", "usdcad_log"), ("usd_broad", "usd_broad_log")):
        if src in out.columns:
            out[dst] = np.log(out[src].where(out[src] > 0))
    return out


@dataclass
class WindowSet:
    X: np.ndarray               # [N, L, C]
    Y: np.ndarray               # [N, T] log-giá tương lai
    origin: pd.DatetimeIndex    # ngày cuối cửa sổ đầu vào
    target_end: pd.DatetimeIndex
    columns: list
    L: int
    T: int

    def __len__(self):
        return len(self.origin)


def _valid_origins(invalid: np.ndarray, L: int, T: int, gap: np.ndarray | None = None) -> np.ndarray:
    n = len(invalid)
    cs = np.concatenate([[0], np.cumsum(invalid)])
    origins = np.arange(L - 1, n - T)
    start, end = origins - L + 1, origins + T            # end: bao gồm
    ok = (cs[end + 1] - cs[start]) == 0
    if gap is not None:
        # gap[j] = True nghĩa là có đoạn thiếu giữa dòng j-1 và j; chỉ tính khi cả hai nằm trong cửa sổ
        cg = np.concatenate([[0], np.cumsum(gap)])
        ok &= (cg[end + 1] - cg[start + 1]) == 0
    return origins[ok]


def make_windows(df: pd.DataFrame, L: int, T: int, feature_cols=(TARGET,),
                 study_start=None) -> WindowSet:
    cols = list(feature_cols)
    if cols[0] != TARGET:
        raise ValueError(f"Kênh 0 phải là {TARGET}.")
    arr = df[cols].to_numpy(dtype=float)
    invalid = df["is_excluded"].to_numpy(dtype=bool) | np.isnan(arr).any(axis=1)
    gap = df["gap_before"].to_numpy(dtype=bool) if "gap_before" in df.columns else None
    origins = _valid_origins(invalid, L, T, gap)
    if study_start is not None:
        origins = origins[df.index[origins] >= pd.Timestamp(study_start)]

    starts = origins - L + 1
    X = sliding_window_view(arr, L, axis=0)[starts].transpose(0, 2, 1)   # [N, L, C]
    Y = sliding_window_view(arr[:, 0], T)[origins + 1]                   # [N, T]
    return WindowSet(X=np.ascontiguousarray(X, dtype=np.float32),
                     Y=np.ascontiguousarray(Y, dtype=np.float32),
                     origin=df.index[origins], target_end=df.index[origins + T],
                     columns=cols, L=L, T=T)


def select_channels(ws: WindowSet, cols) -> WindowSet:
    """Lấy tập con kênh (giữ nguyên các mẫu, để mọi mô hình dùng CÙNG tập mẫu)."""
    cols = list(cols)
    if cols[0] != TARGET:
        raise ValueError(f"Kênh 0 phải là {TARGET}.")
    ix = [ws.columns.index(c) for c in cols]
    return WindowSet(X=np.ascontiguousarray(ws.X[:, :, ix]), Y=ws.Y, origin=ws.origin,
                     target_end=ws.target_end, columns=cols, L=ws.L, T=ws.T)


def window_availability(df: pd.DataFrame, L: int, T: int, feature_cols=(TARGET,),
                        study_start=None) -> pd.Series:
    """Số mẫu dùng được theo năm của origin (để thấy giai đoạn loại 2020 làm mất bao nhiêu)."""
    ws = make_windows(df, L, T, feature_cols, study_start)
    return pd.Series(1, index=ws.origin).groupby(ws.origin.year).sum().rename(f"L={L}")


# ----------------------------------------------------------------------------
# Scaler: chỉ fit trên dữ liệu thuộc giai đoạn train
# ----------------------------------------------------------------------------
class ChannelScaler:
    def fit(self, df: pd.DataFrame, columns, start, end):
        idx = df.index
        mask = ((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
                & ~df["is_excluded"].to_numpy(dtype=bool))
        rows = df.loc[mask, list(columns)].dropna()
        if len(rows) < 30:
            raise ValueError("Quá ít dữ liệu để fit scaler.")
        self.columns = list(columns)
        self.mean = np.array(rows.mean(), dtype=float)
        std = np.array(rows.std(ddof=0), dtype=float)
        std[std < 1e-8] = 1.0
        self.std = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:           # [..., C]
        return ((X - self.mean) / self.std).astype(np.float32)

    def transform_target(self, Y: np.ndarray) -> np.ndarray:
        return ((Y - self.mean[0]) / self.std[0]).astype(np.float32)

    def inverse_target(self, Y: np.ndarray) -> np.ndarray:
        return Y * self.std[0] + self.mean[0]


# ----------------------------------------------------------------------------
# Walk-forward (expanding window) có purge
# ----------------------------------------------------------------------------
@dataclass
class Fold:
    test_year: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_end_date: pd.Timestamp     # ngày nhãn cuối của train, dùng để fit scaler


def make_folds(ws: WindowSet, wf: dict):
    """Chia theo thời gian. Purge: nhãn của train/val phải kết thúc trước origin đầu tiên
    của đoạn kế tiếp, để không train bằng giá nằm trong giai đoạn đánh giá.

    Trả về (danh sách Fold, danh sách (năm, lý do) các fold bị bỏ).
    """
    folds, skipped = [], []
    for year in wf["test_years"]:
        t0, t1 = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        test = np.where((ws.origin >= t0) & (ws.origin < t1))[0]
        if len(test) < wf["min_test_samples"]:
            skipped.append((year, f"chỉ {len(test)} mẫu test"))
            continue
        first_test = ws.origin[test[0]]
        pool = np.where((ws.target_end <= first_test)
                        & (ws.origin >= pd.Timestamp(wf["train_start"])))[0]
        if len(pool) < wf["val_samples"] + wf["min_train_samples"]:
            skipped.append((year, f"chỉ {len(pool)} mẫu trước test"))
            continue
        val = pool[-wf["val_samples"]:]
        first_val = ws.origin[val[0]]
        train = pool[ws.target_end[pool] <= first_val]
        if len(train) < wf["min_train_samples"]:
            skipped.append((year, f"chỉ {len(train)} mẫu train sau purge"))
            continue
        folds.append(Fold(year, train, val, test, ws.target_end[train[-1]]))
    return folds, skipped
