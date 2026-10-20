"""Huấn luyện toàn bộ mô hình một lần, lưu lại, và dùng lại để dự báo (không huấn luyện lại).

Mỗi "bộ mô hình" nằm trong saved_models/asof_YYYYMMDD/ gồm manifest.json và các file {mô hình}__L{L}.pt
(trọng số + scaler + cấu hình). Dự báo = đưa L phiên cuối đến ngày chọn qua mô hình đã lưu.
Trạng thái: TRONG MẪU nếu cả 5 phiên mục tiêu nằm trong dữ liệu đến ngày cắt (đã dùng để huấn luyện/chọn epoch);
MỘT PHẦN nếu ngày cắt nằm giữa; NGOÀI MẪU nếu ngày dự báo từ ngày cắt trở đi.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import estimate_drift
from src.config import PROJECT_ROOT
from src.live_predict import (LivePredictError, data_fingerprint, fit_model, forecast_window, load_bundle,
                              model_catalog, next_session_dates, resolve_origin, save_bundle, window_problem, T)

_CACHE: dict = {}


def store_dir(live: dict | None = None) -> Path:
    return PROJECT_ROOT / (live or {}).get("store", {}).get("dir", "saved_models")


def train_all(df, cfg, wf, exp, live, cutoffs=None, windows=None, seed=None, device="cpu", names=None,
              overwrite=False, epochs=None, log=print) -> list:
    """Huấn luyện mọi mô hình trong danh mục cho từng (cutoff, L) và lưu. Trả về các thư mục bộ mô hình."""
    st = live.get("store", {})
    cutoffs = cutoffs or st.get("cutoffs", ["last"])
    windows = windows or st.get("windows", [5, 30, 120])
    seed = st.get("seed", 0) if seed is None else seed
    catalog = {n: s for n, s in model_catalog(exp).items() if not names or n in names}
    run_exp = dict(exp, epochs=epochs) if epochs else exp
    out = []
    for c in cutoffs:
        origin = df.index[-1] if str(c) == "last" else resolve_origin(df, c)
        d = store_dir(live) / f"asof_{origin:%Y%m%d}"
        d.mkdir(parents=True, exist_ok=True)
        mpath = d / "manifest.json"
        man = json.loads(mpath.read_text(encoding="utf-8")) if (mpath.exists() and not overwrite) else \
            {"cutoff": str(origin.date()), "seed": seed, "windows": [], "models": []}
        man["skipped"], man["tag"] = [], d.name
        trunc = df.loc[:origin]
        man["drift_per_session"] = float(estimate_drift(trunc, wf["train_start"], origin))
        done = {(m["name"], m["L"]) for m in man["models"]}
        for L in windows:
            for name, spec in catalog.items():
                if spec["kind"] == "baseline" or (name, L) in done:
                    continue
                t0 = time.time()
                try:
                    bundle = fit_model(trunc, spec, L, wf, run_exp, cfg["data"]["study_start"], seed=seed, device=device)
                except LivePredictError as e:
                    man["skipped"].append({"name": name, "L": L, "reason": str(e)})
                    log(f"[{d.name}] bỏ qua {name} L={L}: {e}")
                    continue
                fname = f"{name}__L{L}.pt"
                save_bundle(d / fname, bundle, spec, L, data_fingerprint(trunc, spec["cols"]))
                man["models"].append({"name": name, "L": L, "file": fname, "columns": spec["cols"],
                                      **bundle.meta, "seconds": round(time.time() - t0, 1)})
                log(f"[{d.name}] đã lưu {name} L={L} ({time.time() - t0:.1f}s)")
            if L not in man["windows"]:
                man["windows"].append(L)
        mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        out.append(d)
    return out


def list_sets(live: dict | None = None) -> list:
    """Các bộ mô hình đã lưu (mới nhất trước): [{'dir', 'tag', 'cutoff', 'windows', 'models'}...]."""
    sets = []
    for m in sorted(store_dir(live).glob("asof_*/manifest.json"), reverse=True):
        man = json.loads(m.read_text(encoding="utf-8"))
        man["dir"] = m.parent
        sets.append(man)
    return sets


def available_models(man: dict, L: int) -> list:
    base = ["naive", "drift"]
    return base + [m["name"] for m in man["models"] if m["L"] == L]


def predict_saved(df: pd.DataFrame, man: dict, exp: dict, name: str, L: int, date, horizon: int = T) -> dict:
    """Dự báo 5 phiên bằng mô hình ĐÃ LƯU (chỉ chạy suy luận, không huấn luyện)."""
    catalog = model_catalog(exp)
    if name not in catalog:
        raise LivePredictError(f"Mô hình không tồn tại: {name}")
    spec = catalog[name]
    origin = resolve_origin(df, date)
    if not np.isfinite(df["wti_log"].loc[origin]):
        raise LivePredictError("Giá WTI tại ngày này không hợp lệ (không dương).")
    last_price = float(df["wti"].loc[origin])
    cutoff = pd.Timestamp(man["cutoff"])
    info = {}
    if spec["kind"] == "baseline":
        if name == "naive":
            forecast = np.repeat(last_price, horizon)
        else:
            mu = man["drift_per_session"]
            forecast, info = last_price * np.exp(mu * np.arange(1, horizon + 1)), {"drift_per_session": mu}
        cols = ["wti_log"]
    else:
        cols = list(spec["cols"])
        problem = window_problem(df, origin, L, cols)
        if problem:
            raise LivePredictError(problem)
        key = (str(man["dir"]), name, L)
        if key not in _CACHE:
            _CACHE[key] = load_bundle(Path(man["dir"]) / f"{name}__L{L}.pt", spec, L, exp, None, horizon)
        bundle = _CACHE[key]
        if bundle is None:
            raise LivePredictError(f"Chưa có mô hình đã lưu {name} với L={L} trong {man['tag']}.")
        forecast = forecast_window(bundle, df, origin, L, spec)
        info = dict(bundle.meta)
    pos = df.index.get_loc(origin)
    dates, projected = next_session_dates(df, origin, horizon)
    actual = np.full(horizon, np.nan)
    fut = df["wti"].iloc[pos + 1:pos + 1 + horizon].to_numpy(dtype=float)
    actual[:len(fut)] = fut
    win = df.loc[:origin].iloc[-L:]
    target_end = dates[-1] if (projected or pos + horizon >= len(df)) else df.index[pos + horizon]
    status = "trong mẫu" if target_end <= cutoff else ("một phần" if origin < cutoff else "ngoài mẫu")
    return {"model": name, "kind": spec["kind"], "L": L, "origin": origin, "last_price": last_price, "status": status,
            "input_dates": win.index, "input_prices": win["wti"].to_numpy(dtype=float),
            "forecast": np.asarray(forecast, dtype=float), "naive": np.repeat(last_price, horizon),
            "dates": dates, "dates_projected": projected, "actual": actual, "info": info,
            "in_sample": status == "trong mẫu", "cutoff": cutoff, "input_columns": cols, "set": man["tag"]}
