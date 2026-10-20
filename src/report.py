"""Gom kết quả dự báo thành các bảng so sánh A / B / C (không phụ thuộc torch).

Quy ước:
  * Mọi chỉ số tính trên log-giá (sai số log ~ sai số phần trăm), so với log-giá đóng cuối.
  * Mô hình có nhiều seed: chỉ số = trung bình (và độ lệch chuẩn) qua các seed; kiểm định
    Diebold-Mariano dùng dự báo TRUNG BÌNH qua các seed.
  * Chỉ dùng các năm test mà MỌI mô hình đều có kết quả, và tất cả mô hình phải có cùng origin.
  * mean_diff trong bảng DM < 0 nghĩa là mô hình đứng trước có loss nhỏ hơn (tốt hơn) tham chiếu.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import diebold_mariano, load_prediction_file, metrics_table

_PAT = re.compile(r"^(?P<name>.+)__(?P<year>\d{4})__s(?P<seed>\d+)$")

REGIMES = {"2017-2019": [2017, 2018, 2019], "2021-2022": [2021, 2022],
           "2023-2025": [2023, 2024, 2025], "2026": [2026]}


def load_predictions(pred_dir) -> dict:
    """{L: {tên mô hình: {seed: {năm test: kết quả}}}}"""
    out: dict = {}
    for f in sorted(Path(pred_dir).glob("L*/*.npz")):
        m = _PAT.match(f.stem)
        if not m:
            continue
        L = int(f.parent.name[1:])
        out.setdefault(L, {}).setdefault(m["name"], {}).setdefault(int(m["seed"]), {})[int(m["year"])] = \
            load_prediction_file(f)
    return out


def common_years(models: dict) -> list:
    sets = [set(by_year) for seeds in models.values() for by_year in seeds.values()]
    return sorted(set.intersection(*sets)) if sets else []


def pool(by_year: dict, years) -> dict | None:
    ys = [y for y in sorted(by_year) if y in set(years)]
    if not ys:
        return None
    cat = lambda k: np.concatenate([by_year[y][k] for y in ys], axis=0)
    return {"pred": cat("pred_log"), "actual": cat("actual_log"), "last": cat("last_log"),
            "origin": cat("origin"),
            "year": np.concatenate([np.full(len(by_year[y]["origin"]), y) for y in ys])}


def pool_all(models: dict, years) -> dict:
    pooled = {m: {s: pool(by_year, years) for s, by_year in seeds.items()} for m, seeds in models.items()}
    ref = next(iter(next(iter(pooled.values())).values()))
    for m, seeds in pooled.items():
        for p in seeds.values():
            if not np.array_equal(p["origin"], ref["origin"]):
                raise ValueError(f"Mô hình {m} không cùng tập origin với các mô hình khác.")
    return pooled


def summarize(pooled: dict, naive: str = "naive"):
    """Trả về (bảng theo bước tau, bảng trung bình qua tau) cho từng mô hình."""
    naive_pred = pooled[naive][min(pooled[naive])]["pred"]
    rows = []
    for m, seeds in pooled.items():
        for s, p in seeds.items():
            t = metrics_table(p["pred"], p["actual"], p["last"], naive_pred)
            t["model"], t["seed"] = m, s
            rows.append(t)
    allm = pd.concat(rows)
    by_tau = (allm.groupby(["model", "tau"])
              .agg(rmse=("rmse", "mean"), rmse_std=("rmse", "std"), mae=("mae", "mean"),
                   dir_acc=("dir_acc", "mean"), ratio_vs_naive=("rmse_ratio_vs_naive", "mean"))
              .reset_index().fillna({"rmse_std": 0.0}))
    avg = (by_tau.groupby("model").agg(rmse=("rmse", "mean"), mae=("mae", "mean"),
                                       dir_acc=("dir_acc", "mean"), ratio_vs_naive=("ratio_vs_naive", "mean"))
           .reset_index().sort_values("ratio_vs_naive").reset_index(drop=True))
    return by_tau, avg


def default_pairs(names: list, avg: pd.DataFrame) -> list:
    a_models = [m for m in ("linear", "nlinear", "dlinear") if m in names]
    pairs = [(m, "naive") for m in names if m != "naive"]
    if "hybrid_B" in names:
        pairs += [("hybrid_B", a) for a in a_models]
        best_a = avg[avg["model"].isin(a_models)].iloc[0]["model"] if a_models else None
        if best_a:
            pairs.append(("hybrid_B", best_a))
    pairs += [(m, "hybrid_B") for m in names if m.startswith("hybrid_C") and "hybrid_B" in names]
    pairs += [(m, m[:-3]) for m in names if m.endswith("_mv") and m[:-3] in names]
    for m in names:                                    # biến thể cải tiến so với bản gốc của chính nó
        core = m[:-7] if m.endswith("_shrink") else m
        if m.endswith("_shrink") and core in names:
            pairs.append((m, core))
        base = core[:-3] if core.endswith("_zr") else None
        if base and base in names:
            pairs.append((m, base))
    return list(dict.fromkeys(pairs))


def dm_table(pooled: dict, pairs: list, alpha: float = 0.05) -> pd.DataFrame:
    mean_pred = {m: np.mean([p["pred"] for p in seeds.values()], axis=0) for m, seeds in pooled.items()}
    ref = next(iter(next(iter(pooled.values())).values()))
    actual = ref["actual"]
    rows = []
    for m1, m2 in pairs:
        for j in range(actual.shape[1]):
            r = diebold_mariano(mean_pred[m1][:, j] - actual[:, j], mean_pred[m2][:, j] - actual[:, j], h=j + 1)
            if np.isnan(r["p_value"]) or r["p_value"] >= alpha:
                verdict = "không khác biệt"
            else:
                verdict = "mô hình tốt hơn" if r["mean_diff"] < 0 else "tham chiếu tốt hơn"
            rows.append({"model": m1, "reference": m2, "tau": j + 1, **r, "verdict": verdict})
    return pd.DataFrame(rows)


def by_group(pooled: dict, groups: dict, naive: str = "naive") -> pd.DataFrame:
    """RMSE trung bình qua tau và tỉ lệ so với naive, theo nhóm năm (chế độ thị trường)."""
    rows = []
    for label, years in groups.items():
        per_model = {}
        for m, seeds in pooled.items():
            vals = []
            for p in seeds.values():
                mask = np.isin(p["year"], years)
                if mask.sum() == 0:
                    continue
                vals.append(np.sqrt(np.mean((p["pred"][mask] - p["actual"][mask]) ** 2, axis=0)).mean())
            if vals:
                n = int(np.isin(next(iter(seeds.values()))["year"], years).sum())
                per_model[m] = (float(np.mean(vals)), n)
        if naive not in per_model:
            continue
        for m, (rmse, n) in per_model.items():
            rows.append({"group": label, "model": m, "n_samples": n, "rmse": rmse,
                         "ratio_vs_naive": rmse / per_model[naive][0]})
    return pd.DataFrame(rows)
