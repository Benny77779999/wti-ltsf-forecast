"""Định nghĩa danh sách thí nghiệm (không phụ thuộc torch, dễ kiểm thử)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset import TARGET


def all_channel_columns(exp: dict) -> list:
    """Hợp của mọi nhóm biến phụ trợ, giữ thứ tự khai báo."""
    return list(dict.fromkeys(c for cols in exp["feature_groups"].values() for c in cols))


def build_specs(exp: dict, groups: list | None = None) -> list:
    """Mỗi spec: name, kind ('baseline'|'model'), model/base, cols, level."""
    specs = [
        {"name": "naive", "kind": "baseline", "base": "naive", "cols": [TARGET], "level": False},
        {"name": "drift", "kind": "baseline", "base": "drift", "cols": [TARGET], "level": False},
    ]
    for m in ("linear", "nlinear", "dlinear"):                                   # nhóm A
        specs.append({"name": m, "kind": "model", "model": m, "cols": [TARGET], "level": False})
    for m in ("linear", "nlinear", "dlinear"):                                   # nhóm A đa biến (mọi biến phụ trợ)
        specs.append({"name": f"{m}_mv", "kind": "model", "model": m,
                      "cols": [TARGET] + all_channel_columns(exp), "level": False})
    specs.append({"name": "hybrid_B", "kind": "model", "model": "hybrid",         # nhóm B
                  "cols": [TARGET], "level": False})
    group_map = dict(exp["feature_groups"])
    group_map["all"] = all_channel_columns(exp)
    for g, cols in group_map.items():                                            # nhóm C
        if groups and g not in groups:
            continue
        for lvl in exp.get("level_branch_variants", [False]):
            specs.append({"name": f"hybrid_C_{g}{'_lvl' if lvl else ''}", "kind": "model",
                          "model": "hybrid", "cols": [TARGET] + list(cols), "level": bool(lvl)})
    return specs


def build_improve_specs(exp: dict) -> list:
    """Cải tiến vòng 1: khởi tạo trọng số bằng 0 + weight decay chọn trên validation, mỗi mô hình
    cho thêm bản co về naive. Mỗi spec huấn luyện một lần và tạo hai kết quả: `name` và `name_shrink`."""
    cfg = exp["improve1"]
    groups = exp["feature_groups"]
    items = (("nlinear_zr", "nlinear", [TARGET]),
             ("hybrid_B_zr", "hybrid", [TARGET]),
             ("hybrid_C_usd_zr", "hybrid", [TARGET] + list(groups["usd"])))
    return [{"name": n, "kind": "tuned", "model": m, "cols": c, "level": False,
             "wd_grid": list(cfg["wd_grid"]), "zero_init": True, "outputs": [n, n + "_shrink"]}
            for n, m, c in items]


def spec_outputs(spec: dict) -> list:
    return spec.get("outputs", [spec["name"]])


def parse_windows(args_windows, wf: dict) -> list:
    if not args_windows or args_windows == ["main"]:
        return list(wf["main_windows"])
    if args_windows == ["supp"]:
        return list(wf["supplementary_windows"])
    if args_windows == ["all"]:
        return list(wf["main_windows"]) + list(wf["supplementary_windows"])
    return [int(w) for w in args_windows]


def prediction_path(out_dir, L: int, name: str, year: int, seed: int) -> Path:
    return Path(out_dir) / f"L{L}" / f"{name}__{year}__s{seed}.npz"


def fold_table(folds, skipped) -> pd.DataFrame:
    rows = [{"test_year": f.test_year, "status": "run", "n_train": len(f.train), "n_val": len(f.val),
             "n_test": len(f.test), "train_end_date": f.train_end_date} for f in folds]
    rows += [{"test_year": y, "status": f"skipped: {why}"} for y, why in skipped]
    return pd.DataFrame(rows).sort_values("test_year").reset_index(drop=True)
