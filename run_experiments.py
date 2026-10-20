"""Chạy thí nghiệm walk-forward. Mỗi kết quả lưu ra một file .npz; chạy lại sẽ bỏ qua file đã có.

Ví dụ:
    python run_experiments.py --dry-run                      # chỉ in kế hoạch và số lần huấn luyện
    python run_experiments.py --windows 30 --test-years 2023 --seeds 1   # chạy thử nhỏ
    python run_experiments.py                                # toàn bộ L chính (nên chạy qua đêm)
    python run_experiments.py --set improve1 --windows 5 30  # cải tiến vòng 1 (cần đã có kết quả gốc)
"""
import argparse
import time

import pandas as pd

from src.config import PROJECT_ROOT, load_config, resolve_path
from src.dataset import TARGET, add_log_price, make_folds, make_windows, select_channels
from src.experiments import (all_channel_columns, build_improve_specs, build_specs, fold_table,
                             parse_windows, prediction_path, spec_outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="base", choices=["base", "improve1"], help="bộ thí nghiệm")
    ap.add_argument("--windows", nargs="+", default=None, help="main | supp | all | danh sách L")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--test-years", nargs="+", type=int, default=None)
    ap.add_argument("--groups", nargs="+", default=None, help="chỉ chạy các nhóm C này (spread usd ovx fed all)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    wf = load_config(PROJECT_ROOT / "configs" / "walk_forward.yaml")
    exp = load_config(PROJECT_ROOT / "configs" / "experiments.yaml")
    n_seeds = args.seeds or exp["seeds"]
    epochs = args.epochs or exp["epochs"]
    T, study_start = wf["horizon"], cfg["data"]["study_start"]

    df = pd.read_csv(resolve_path(cfg, "processed_dir") / "wti_dataset.csv",
                     parse_dates=["date"], index_col="date")
    df = add_log_price(df)
    specs = build_improve_specs(exp) if args.set == "improve1" else build_specs(exp, args.groups)
    all_cols = [TARGET] + all_channel_columns(exp)       # cùng tập mẫu với bộ thí nghiệm gốc
    out_dir = resolve_path({"paths": {"p": "results/predictions"}}, "p")
    tables = resolve_path(cfg, "tables_dir")

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    plan = []
    for L in parse_windows(args.windows, wf):
        ws_all = make_windows(df, L, T, all_cols, study_start)
        folds, skipped = make_folds(ws_all, wf)
        if args.test_years:
            folds = [f for f in folds if f.test_year in args.test_years]
        if args.set == "base":
            fold_table(folds, skipped).to_csv(tables / f"folds_L{L}.csv", index=False)
        print(f"L={L}: {len(ws_all)} mẫu, {len(folds)} fold chạy, bỏ {[y for y, _ in skipped]}")
        cache = {}
        for fold in folds:
            for spec in specs:
                for seed in ([0] if spec["kind"] == "baseline" else range(n_seeds)):
                    paths = [(name, prediction_path(out_dir, L, name, fold.test_year, seed))
                             for name in spec_outputs(spec)]
                    plan.append((L, ws_all, cache, fold, spec, seed, paths))

    todo = [p for p in plan if not all(path.exists() for _, path in p[-1])]
    print(f"\nKế hoạch: {len(plan)} lần huấn luyện/baseline, đã xong {len(plan) - len(todo)}, "
          f"còn {len(todo)} cần chạy.")
    if args.dry_run or not todo:
        return

    from src.improvements import shrink_result
    from src.train import run_baseline, run_fold, run_fold_tuned, save_predictions   # nạp torch muộn

    start = time.time()
    for i, (L, ws_all, cache, fold, spec, seed, paths) in enumerate(todo, 1):
        key = tuple(spec["cols"])
        if key not in cache:
            cache[key] = select_channels(ws_all, spec["cols"])
        ws = cache[key]
        if spec["kind"] == "baseline":
            results = [run_baseline(df, ws, fold, spec["base"], wf)]
        elif spec["kind"] == "tuned":
            res = run_fold_tuned(df, ws, fold, spec["model"], wf, spec["wd_grid"], seed=seed,
                                 zero_init=spec["zero_init"], epochs=epochs, lr=exp["lr"],
                                 batch_size=exp["batch_size"], moving_avg=exp["moving_avg"], device=device,
                                 level_branch=spec["level"])
            results = [res, shrink_result(res)]
        else:
            results = [run_fold(df, ws, fold, spec["model"], wf, seed=seed, epochs=epochs, lr=exp["lr"],
                                batch_size=exp["batch_size"], moving_avg=exp["moving_avg"], device=device,
                                level_branch=spec["level"])]
        for (_, path), res in zip(paths, results):
            path.parent.mkdir(parents=True, exist_ok=True)
            save_predictions(path, res)
        el = time.time() - start
        print(f"[{i}/{len(todo)}] L={L} fold={fold.test_year} {spec['name']} s{seed} "
              f"(đã {el/60:.1f} phút, còn ~{el/i*(len(todo)-i)/60:.0f} phút)", flush=True)


if __name__ == "__main__":
    main()
