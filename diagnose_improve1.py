"""Chẩn đoán cải tiến vòng 1: weight decay được chọn và hệ số co lambda ở từng fold và seed.

    python diagnose_improve1.py
Chỉ đọc results/predictions/, ghi thêm results/tables/improve1_diagnostics.csv.
"""
import re

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config, resolve_path

PAT = re.compile(r"^(?P<name>.+_zr(?:_shrink)?)__(?P<year>\d{4})__s(?P<seed>\d+)$")


def main():
    rows = []
    for f in sorted((PROJECT_ROOT / "results" / "predictions").glob("L*/*.npz")):
        m = PAT.match(f.stem)
        if not m:
            continue
        z = np.load(f)
        rows.append({"L": int(f.parent.name[1:]), "model": m["name"], "year": int(m["year"]),
                     "seed": int(m["seed"]),
                     "chosen_wd": float(z["chosen_wd"]) if "chosen_wd" in z.files else np.nan,
                     "shrink_lambda": float(z["shrink_lambda"]) if "shrink_lambda" in z.files else np.nan,
                     "best_epoch": int(z["best_epoch"])})
    if not rows:
        raise SystemExit("Chưa có kết quả của vòng 1 (tên mô hình có hậu tố _zr).")
    df = pd.DataFrame(rows)
    out = resolve_path(load_config(), "tables_dir") / "improve1_diagnostics.csv"
    df.to_csv(out, index=False)

    tuned = df[~df["model"].str.endswith("_shrink")]
    print("=== Weight decay được chọn (số lần, mọi fold và seed) ===")
    print(pd.crosstab([tuned["L"], tuned["model"]], tuned["chosen_wd"]).to_string())

    sh = df[df["model"].str.endswith("_shrink")]
    print("\n=== Hệ số co lambda (0 = naive, 1 = giữ nguyên mô hình) ===")
    summ = sh.groupby(["L", "model"])["shrink_lambda"].agg(
        n="count", mean="mean", median="median",
        share_zero=lambda s: float((s < 0.01).mean()), share_one=lambda s: float((s > 0.99).mean()))
    print(summ.round(3).to_string())

    print("\n=== lambda trung bình theo năm test (trung bình qua seed) ===")
    for L, g in sh.groupby("L"):
        print(f"L={L}")
        print(g.pivot_table(index="model", columns="year", values="shrink_lambda", aggfunc="mean").round(2).to_string())
    print(f"\nĐã lưu {out}")


if __name__ == "__main__":
    main()
