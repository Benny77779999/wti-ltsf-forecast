"""Huấn luyện một mô hình trên một fold walk-forward và dự báo tập test.

Quy trình (theo tài liệu, Mục 9.5): MSE, Adam, StepLR(step=20, gamma=0.5), 50 epoch, batch 32.
Bổ sung: giữ trọng số của epoch có val loss thấp nhất (chọn trên validation, không đụng test).
Scaler chỉ fit trên giai đoạn train của fold (xem dataset.ChannelScaler).
Chia batch thủ công (randperm có seed) thay cho DataLoader để chạy nhanh hơn; ngữ nghĩa giữ nguyên
(xáo trộn mỗi epoch, bỏ batch lẻ cuối).
"""
from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.baselines import drift_forecast, estimate_drift, naive_forecast
from src.dataset import ChannelScaler, Fold, WindowSet
from src.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_inputs(model_name: str, X: np.ndarray) -> np.ndarray:
    """Mô hình một kênh nhận [N, L]; mô hình nhiều kênh (hybrid hoặc C > 1) nhận [N, L, C]."""
    return X if (model_name.lower() == "hybrid" or X.shape[-1] > 1) else X[:, :, 0]


def predict(model: nn.Module, X: np.ndarray, device: str = "cpu", batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_one(model: nn.Module, Xtr, Ytr, Xva, Yva, epochs: int = 50, lr: float = 1e-3,
              batch_size: int = 32, device: str = "cpu", seed: int = 0, weight_decay: float = 0.0):
    """Trả về (mô hình với trọng số tốt nhất trên val, lịch sử loss, epoch tốt nhất)."""
    set_seed(seed)
    model.to(device)
    gen = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr)).to(device)
    Yt = torch.from_numpy(np.ascontiguousarray(Ytr)).to(device)
    Xva_t = torch.from_numpy(np.ascontiguousarray(Xva)).to(device)
    Yva_t = torch.from_numpy(np.ascontiguousarray(Yva)).to(device)
    n = len(Xt)
    bs = min(batch_size, n)
    n_batches = max(1, n // bs)                              # bỏ batch lẻ cuối (drop_last)

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)

    hist = {"train": [], "val": []}
    best_val, best_state, best_epoch = np.inf, None, -1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        total = 0.0
        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs]
            opt.zero_grad()
            loss = criterion(model(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
            total += loss.item()
        hist["train"].append(total / n_batches)

        model.eval()
        with torch.no_grad():
            v = criterion(model(Xva_t), Yva_t).item()
        hist["val"].append(v)
        if v < best_val:
            best_val, best_epoch = v, ep
            best_state = copy.deepcopy(model.state_dict())
        sched.step()

    model.load_state_dict(best_state)
    return model, hist, best_epoch


def _pack(ws: WindowSet, idx: np.ndarray, pred_log: np.ndarray, extra: dict | None = None) -> dict:
    out = {
        "origin": ws.origin[idx].values.astype("datetime64[ns]"),
        "pred_log": np.asarray(pred_log, dtype=float),
        "actual_log": ws.Y[idx].astype(float),
        "last_log": ws.X[idx, -1, 0].astype(float),      # log-giá đóng cuối đã biết tại origin
    }
    out.update(extra or {})
    return out


def run_fold(df: pd.DataFrame, ws: WindowSet, fold: Fold, model_name: str, wf: dict, seed: int = 0,
             epochs: int = 50, lr: float = 1e-3, batch_size: int = 32, moving_avg: int = 5,
             device: str = "cpu", level_branch: bool = False, weight_decay: float = 0.0,
             zero_init: bool = False) -> dict:
    """Huấn luyện `model_name` trên fold, dự báo test, trả về log-giá dự báo và thực tế."""
    set_seed(seed)          # phải đặt seed TRƯỚC khi khởi tạo mô hình, nếu không trọng số ban đầu ngẫu nhiên
    scaler = ChannelScaler().fit(df, ws.columns, wf["train_start"], fold.train_end_date)
    Xs, Ys = scaler.transform(ws.X), scaler.transform_target(ws.Y)
    model = build_model(model_name, ws.L, ws.T, n_channels=len(ws.columns), moving_avg=moving_avg,
                        level_branch=level_branch, zero_init=zero_init)

    model, hist, best_epoch = train_one(
        model, model_inputs(model_name, Xs[fold.train]), Ys[fold.train],
        model_inputs(model_name, Xs[fold.val]), Ys[fold.val],
        epochs=epochs, lr=lr, batch_size=batch_size, device=device, seed=seed, weight_decay=weight_decay)

    pred_s = predict(model, model_inputs(model_name, Xs[fold.test]), device)
    pred_val_s = predict(model, model_inputs(model_name, Xs[fold.val]), device)
    return _pack(ws, fold.test, scaler.inverse_target(pred_s),
                 {"best_epoch": best_epoch, "best_val_loss": float(hist["val"][best_epoch]),
                  "weight_decay": float(weight_decay),
                  "train_loss": np.array(hist["train"]), "val_loss": np.array(hist["val"]),
                  # dự báo trên tập validation, dùng để chọn hệ số co về naive (src/improvements.py)
                  "val_pred_log": scaler.inverse_target(pred_val_s),
                  "val_actual_log": ws.Y[fold.val].astype(float),
                  "val_last_log": ws.X[fold.val, -1, 0].astype(float)})


def run_fold_tuned(df: pd.DataFrame, ws: WindowSet, fold: Fold, model_name: str, wf: dict,
                   wd_grid, seed: int = 0, zero_init: bool = True, **kwargs) -> dict:
    """Huấn luyện với từng weight_decay trong lưới, chọn giá trị có val loss thấp nhất (chỉ dùng validation)."""
    best = None
    for wd in wd_grid:
        r = run_fold(df, ws, fold, model_name, wf, seed=seed, weight_decay=wd, zero_init=zero_init, **kwargs)
        if best is None or r["best_val_loss"] < best["best_val_loss"]:
            best = r
    best["chosen_wd"] = best["weight_decay"]
    return best


def run_baseline(df: pd.DataFrame, ws: WindowSet, fold: Fold, kind: str, wf: dict) -> dict:
    """Baseline không học, dùng log-giá chưa chuẩn hoá."""
    Xt = ws.X[fold.test]
    if kind == "naive":
        pred = naive_forecast(Xt, ws.T)
    elif kind == "drift":
        mu = estimate_drift(df, wf["train_start"], fold.train_end_date)
        pred = drift_forecast(Xt, ws.T, mu)
    else:
        raise ValueError(f"Baseline không hợp lệ: {kind}")
    return _pack(ws, fold.test, pred)


def save_predictions(path, result: dict) -> None:
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    arrays["origin"] = result["origin"].astype("datetime64[ns]").astype("int64")
    arrays["best_epoch"] = np.array(result.get("best_epoch", -1))
    for k in ("best_val_loss", "weight_decay", "chosen_wd", "shrink_lambda"):
        if k in result:
            arrays[k] = np.array(float(result[k]))
    np.savez_compressed(path, **arrays)
