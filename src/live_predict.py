"""Dự đoán thử một ngày bất kỳ: mô hình THỰC SỰ chạy dự báo từ cửa sổ dữ liệu quá khứ.

Quy trình cho ngày d và L phiên:
  1. Cắt dữ liệu tại d: không có bất kỳ dòng nào sau d được đưa vào huấn luyện hay dự báo.
  2. Huấn luyện mô hình trên các mẫu nằm hoàn toàn trong dữ liệu đã cắt (walk-forward: validation là
     250 mẫu cuối có purge, scaler chỉ fit trên giai đoạn train). Naive và drift không cần huấn luyện.
  3. Đưa L phiên cuối (đến d) qua mô hình để lấy log-giá 5 phiên tiếp theo, rồi đổi về USD.
  4. Giá thực tế sau d (nếu dữ liệu đã có) chỉ được lấy để so sánh SAU khi đã có dự báo.
Mô hình đã huấn luyện được lưu (saved_models/live) để dùng lại; chỉ cần torch khi phải huấn luyện.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import estimate_drift
from src.config import PROJECT_ROOT, load_config, resolve_path
from src.dataset import ChannelScaler, add_log_price, make_windows
from src.experiments import build_improve_specs, build_specs
from src.features import _CBD

T = 5


class LivePredictError(ValueError):
    """Lỗi người dùng có thể sửa được (ngày/L không hợp lệ, thiếu dữ liệu...)."""


# ----------------------------------------------------------------------------
# Cấu hình, dữ liệu, danh mục mô hình
# ----------------------------------------------------------------------------
def load_all_configs():
    cfg = load_config()
    wf = load_config(PROJECT_ROOT / "configs" / "walk_forward.yaml")
    exp = load_config(PROJECT_ROOT / "configs" / "experiments.yaml")
    live_path = PROJECT_ROOT / "configs" / "live.yaml"
    live = load_config(live_path) if live_path.exists() else {}
    return cfg, wf, exp, live


def load_frame(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve_path(cfg, "processed_dir") / "wti_dataset.csv"
    if not path.exists():
        raise LivePredictError("Chưa có data/processed/wti_dataset.csv. Chạy run_data_pipeline.py trước.")
    return add_log_price(pd.read_csv(path, parse_dates=["date"], index_col="date"))


def model_catalog(exp: dict) -> dict:
    """Tên mô hình -> spec (cột đầu vào, có nhánh mức, khởi tạo 0, lưới weight decay...)."""
    specs = build_specs(exp)
    if "improve1" in exp:
        specs = specs + build_improve_specs(exp)
    return {s["name"]: s for s in specs}


def resolve_origin(df: pd.DataFrame, date) -> pd.Timestamp:
    """Phiên giao dịch gần nhất tại hoặc trước `date`."""
    d = pd.Timestamp(date)
    idx = df.index[df.index <= d]
    if len(idx) == 0:
        raise LivePredictError(f"Ngày {d.date()} nằm trước dữ liệu (bắt đầu {df.index[0].date()}).")
    return idx[-1]


def window_problem(df: pd.DataFrame, origin: pd.Timestamp, L: int, cols: list) -> str | None:
    """Trả về lý do cửa sổ L phiên kết thúc tại origin không hợp lệ, hoặc None nếu hợp lệ."""
    pos = df.index.get_loc(origin)
    if pos + 1 < L:
        return f"Chỉ có {pos + 1} phiên dữ liệu đến ngày này, ít hơn L = {L}."
    w = df.iloc[pos - L + 1:pos + 1]
    if w["is_excluded"].to_numpy(bool).any():
        return "Cửa sổ chạm giai đoạn bị loại (giá âm tháng 4/2020), mô hình không nhận cửa sổ này."
    if "gap_before" in w.columns and w["gap_before"].to_numpy(bool)[1:].any():
        return "Cửa sổ chạy qua một đoạn thiếu dữ liệu của nguồn (khoảng cách hơn 5 ngày lịch)."
    missing = [c for c in cols if w[c].isna().any()]
    if missing:
        return f"Cửa sổ thiếu dữ liệu ở: {', '.join(missing)}."
    return None


def next_session_dates(df: pd.DataFrame, origin: pd.Timestamp, horizon: int = T):
    """T phiên sau origin: lấy từ dữ liệu nếu đã có, còn lại suy ra theo lịch làm việc (xấp xỉ)."""
    pos = df.index.get_loc(origin)
    dates = list(df.index[pos + 1:pos + 1 + horizon])
    projected = False
    last = dates[-1] if dates else origin
    while len(dates) < horizon:
        last = last + _CBD
        dates.append(last)
        projected = True
    return pd.DatetimeIndex(dates), projected


# ----------------------------------------------------------------------------
# Tập huấn luyện từ dữ liệu đã cắt
# ----------------------------------------------------------------------------
def build_training(df_trunc: pd.DataFrame, cols: list, L: int, wf: dict, study_start, horizon: int = T):
    ws = make_windows(df_trunc, L, horizon, cols, study_start)
    need = wf["val_samples"] + wf["min_train_samples"]
    if len(ws) < need:
        raise LivePredictError(
            f"Không đủ dữ liệu huấn luyện trước ngày này: có {len(ws)} mẫu, cần ít nhất {need}. "
            "Hãy chọn ngày muộn hơn hoặc L nhỏ hơn.")
    val = np.arange(len(ws) - wf["val_samples"], len(ws))
    first_val = ws.origin[val[0]]
    train = np.where(ws.target_end <= first_val)[0]                 # purge: nhãn train kết thúc trước val
    if len(train) < wf["min_train_samples"]:
        raise LivePredictError(f"Chỉ còn {len(train)} mẫu train sau khi purge, cần ít nhất {wf['min_train_samples']}.")
    return ws, train, val


def data_fingerprint(df_trunc: pd.DataFrame, cols: list) -> str:
    arr = np.round(df_trunc[cols].to_numpy(dtype=float), 6)
    return hashlib.md5(arr.tobytes() + str(len(df_trunc)).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Huấn luyện, lưu, nạp, dự báo (cần torch)
# ----------------------------------------------------------------------------
@dataclass
class Bundle:
    model: object
    scaler: ChannelScaler
    meta: dict


def _import_torch():
    try:
        import torch
        return torch
    except ImportError as e:                                       # pragma: no cover
        raise LivePredictError("Cần cài torch để huấn luyện mô hình: "
                               "pip install torch --index-url https://download.pytorch.org/whl/cpu") from e


def fit_model(df_trunc: pd.DataFrame, spec: dict, L: int, wf: dict, exp: dict, study_start,
              seed: int = 0, device: str = "cpu", horizon: int = T) -> Bundle:
    _import_torch()
    from src.models import build_model
    from src.train import model_inputs, set_seed, train_one

    cols = spec["cols"]
    ws, train, val = build_training(df_trunc, cols, L, wf, study_start, horizon)
    train_end = ws.target_end[train[-1]]
    scaler = ChannelScaler().fit(df_trunc, cols, wf["train_start"], train_end)
    Xs, Ys = scaler.transform(ws.X), scaler.transform_target(ws.Y)

    grid = spec.get("wd_grid", [0.0]) if spec["kind"] == "tuned" else [0.0]
    best = None
    for wd in grid:
        set_seed(seed)                                              # seed trước khi khởi tạo mô hình
        model = build_model(spec["model"], L, horizon, n_channels=len(cols), moving_avg=exp["moving_avg"],
                            level_branch=spec["level"], zero_init=spec.get("zero_init", False))
        model, hist, best_epoch = train_one(
            model, model_inputs(spec["model"], Xs[train]), Ys[train],
            model_inputs(spec["model"], Xs[val]), Ys[val], epochs=exp["epochs"], lr=exp["lr"],
            batch_size=exp["batch_size"], device=device, seed=seed, weight_decay=wd)
        v = float(hist["val"][best_epoch])
        if best is None or v < best["val_loss"]:
            best = {"model": model, "val_loss": v, "wd": float(wd), "best_epoch": int(best_epoch)}
    meta = {"n_train": int(len(train)), "n_val": int(len(val)), "train_end_date": str(train_end.date()),
            "chosen_wd": best["wd"], "best_epoch": best["best_epoch"], "val_loss": best["val_loss"]}
    return Bundle(best["model"], scaler, meta)


def save_bundle(path: Path, bundle: Bundle, spec: dict, L: int, fingerprint: str) -> None:
    torch = _import_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in bundle.model.state_dict().items()},
                "mean": bundle.scaler.mean.tolist(), "std": bundle.scaler.std.tolist(),
                "columns": list(spec["cols"]), "meta": bundle.meta, "fingerprint": fingerprint, "L": int(L)}, path)


def load_bundle(path: Path, spec: dict, L: int, exp: dict, fingerprint: str | None, horizon: int = T) -> Bundle | None:
    """Nạp mô hình đã lưu nếu khớp cấu hình và dữ liệu; ngược lại trả về None (sẽ huấn luyện lại)."""
    if not path.exists():
        return None
    torch = _import_torch()
    from src.models import build_model
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if (fingerprint is not None and ck.get("fingerprint") != fingerprint) or ck.get("L") != L or ck.get("columns") != list(spec["cols"]):
            return None
        model = build_model(spec["model"], L, horizon, n_channels=len(spec["cols"]), moving_avg=exp["moving_avg"],
                            level_branch=spec["level"], zero_init=spec.get("zero_init", False))
        model.load_state_dict(ck["state"])
        scaler = ChannelScaler()
        scaler.columns, scaler.mean, scaler.std = list(spec["cols"]), np.array(ck["mean"]), np.array(ck["std"])
        return Bundle(model, scaler, ck["meta"])
    except Exception:                                              # file hỏng hoặc khác phiên bản: huấn luyện lại
        return None


def forecast_window(bundle: Bundle, df: pd.DataFrame, origin: pd.Timestamp, L: int, spec: dict,
                    device: str = "cpu") -> np.ndarray:
    """Đưa L phiên cuối đến origin qua mô hình; trả về giá dự báo (USD) cho T phiên."""
    from src.train import model_inputs, predict

    pos = df.index.get_loc(origin)
    window = df.iloc[pos - L + 1:pos + 1][spec["cols"]].to_numpy(dtype=float)[None]     # [1, L, C]
    pred_s = predict(bundle.model, model_inputs(spec["model"], bundle.scaler.transform(window)), device)
    return np.exp(bundle.scaler.inverse_target(pred_s)[0])


# ----------------------------------------------------------------------------
# Điểm vào chính
# ----------------------------------------------------------------------------
def predict_day(df: pd.DataFrame, model_name: str, L: int, date, seed: int = 0, exp: dict | None = None,
                wf: dict | None = None, cfg: dict | None = None, cache_dir=None, use_cache: bool = True,
                device: str = "cpu", horizon: int = T) -> dict:
    cfg_, wf_, exp_, _ = load_all_configs() if (exp is None or wf is None or cfg is None) else (cfg, wf, exp, {})
    cfg, wf, exp = cfg or cfg_, wf or wf_, exp or exp_
    catalog = model_catalog(exp)
    if model_name not in catalog:
        raise LivePredictError(f"Mô hình không tồn tại: {model_name}. Có: {', '.join(catalog)}")
    spec = catalog[model_name]
    L = int(L)
    if L < 1:
        raise LivePredictError("L phải là số nguyên dương.")

    origin = resolve_origin(df, date)
    df_trunc = df.loc[:origin]                                      # KHÔNG có dữ liệu sau origin
    if not np.isfinite(df_trunc["wti_log"].iloc[-1]):
        raise LivePredictError("Giá WTI tại ngày này không hợp lệ (không dương).")
    last_price = float(df_trunc["wti"].iloc[-1])
    study_start = cfg["data"]["study_start"]
    t0, from_cache, info = time.time(), False, {}

    if spec["kind"] == "baseline":
        if spec["base"] == "naive":
            forecast = np.repeat(last_price, horizon)
        else:
            mu = estimate_drift(df_trunc, wf["train_start"], origin)
            forecast = last_price * np.exp(mu * np.arange(1, horizon + 1))
            info["drift_per_session"] = float(mu)
        in_cols = ["wti_log"]
    else:
        in_cols = list(spec["cols"])
        problem = window_problem(df_trunc, origin, L, in_cols)
        if problem:
            raise LivePredictError(problem)
        fp = data_fingerprint(df_trunc, in_cols)
        cache_dir = Path(cache_dir) if cache_dir else resolve_path({"paths": {"c": "saved_models/live"}}, "c")
        path = cache_dir / f"{model_name}__L{L}__{origin:%Y%m%d}__s{seed}.pt"
        bundle = load_bundle(path, spec, L, exp, fp, horizon) if use_cache else None
        from_cache = bundle is not None
        if bundle is None:
            bundle = fit_model(df_trunc, spec, L, wf, exp, study_start, seed=seed, device=device, horizon=horizon)
            if use_cache:
                save_bundle(path, bundle, spec, L, fp)
        forecast = forecast_window(bundle, df_trunc, origin, L, spec, device)
        info.update(bundle.meta)

    pos = df.index.get_loc(origin)
    dates, projected = next_session_dates(df, origin, horizon)
    actual = np.full(horizon, np.nan)
    future = df["wti"].iloc[pos + 1:pos + 1 + horizon].to_numpy(dtype=float)
    actual[:len(future)] = future
    win = df_trunc.iloc[-L:] if L <= len(df_trunc) else df_trunc
    return {
        "model": model_name, "kind": spec["kind"], "L": L, "seed": seed, "origin": origin,
        "last_price": last_price, "input_dates": win.index, "input_prices": win["wti"].to_numpy(dtype=float),
        "forecast": np.asarray(forecast, dtype=float), "naive": np.repeat(last_price, horizon),
        "dates": dates, "dates_projected": projected, "actual": actual,
        "from_cache": from_cache, "seconds": time.time() - t0, "info": info,
        "input_columns": in_cols, "data_end": df.index[-1],
    }
