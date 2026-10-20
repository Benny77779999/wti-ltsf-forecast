"""Gom kết quả từ results/predictions/ thành các bảng và hình so sánh.

    python run_evaluation.py                    # mọi L tìm thấy, bỏ fold 2020 khỏi bảng tổng
    python run_evaluation.py --windows 30 120
"""
import argparse

from src.config import PROJECT_ROOT, load_config, resolve_path
from src.report import (REGIMES, by_group, common_years, default_pairs, dm_table, load_predictions,
                        pool_all, summarize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", type=int, default=None)
    ap.add_argument("--exclude-years", nargs="*", type=int, default=[2020],
                    help="năm test bị loại khỏi bảng tổng (mặc định 2020: fold một phần, chế độ lẫn lộn)")
    ap.add_argument("--exclude-improvements", action="store_true",
                    help="bỏ các biến thể cải tiến (_zr, _zr_shrink) để tái tạo bảng của vòng 0")
    ap.add_argument("--tag", default="", help="hậu tố thêm vào tên file đầu ra, để không ghi đè (ví dụ _2023_2025)")
    args = ap.parse_args()

    cfg = load_config()
    tables = resolve_path(cfg, "tables_dir")
    figs = resolve_path({"paths": {"f": "results/figures"}}, "f")
    data = load_predictions(PROJECT_ROOT / "results" / "predictions")
    if not data:
        raise SystemExit("Chưa có kết quả trong results/predictions/. Chạy run_experiments.py trước.")

    for L in sorted(data):
        if args.windows and L not in args.windows:
            continue
        models = data[L]
        if args.exclude_improvements:
            models = {m: v for m, v in models.items() if not m.endswith(("_zr", "_zr_shrink"))}
        years_all = common_years(models)
        years = [y for y in years_all if y not in args.exclude_years]
        if "naive" not in models or not years:
            print(f"L={L}: thiếu naive hoặc không có năm chung, bỏ qua.")
            continue
        print(f"\n===== L={L} | năm test dùng cho bảng tổng: {years} (loại {args.exclude_years}) =====")

        pooled = pool_all(models, years)
        by_tau, avg = summarize(pooled)
        pairs = default_pairs(list(pooled), avg)
        dm = dm_table(pooled, pairs)
        by_year = by_group(pool_all(models, years_all), {str(y): [y] for y in years_all})
        regimes = {k: [y for y in v if y in years] for k, v in REGIMES.items()}
        by_regime = by_group(pooled, {k: v for k, v in regimes.items() if v})

        by_tau.to_csv(tables / f"summary_by_tau_L{L}{args.tag}.csv", index=False)
        avg.to_csv(tables / f"summary_avg_L{L}{args.tag}.csv", index=False)
        dm.to_csv(tables / f"dm_tests_L{L}{args.tag}.csv", index=False)
        by_year.to_csv(tables / f"by_year_L{L}{args.tag}.csv", index=False)
        by_regime.to_csv(tables / f"by_regime_L{L}{args.tag}.csv", index=False)

        print("Tỉ lệ RMSE so với naive (trung bình qua tau; < 1 là tốt hơn naive):")
        print(avg[["model", "rmse", "mae", "dir_acc", "ratio_vs_naive"]].round(4).to_string(index=False))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4.5))
            for m, g in by_tau.groupby("model"):
                ax.plot(g["tau"], g["ratio_vs_naive"], marker="o", label=m, linewidth=1.2)
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("Bước dự báo tau (phiên)"); ax.set_ylabel("RMSE / RMSE naive")
            ax.set_title(f"So với naive theo bước dự báo (L={L})"); ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2); fig.tight_layout()
            fig.savefig(figs / f"ratio_by_tau_L{L}{args.tag}.png", dpi=150); plt.close(fig)
        except ImportError:
            pass
    print(f"\nĐã lưu bảng vào {tables}")


if __name__ == "__main__":
    main()
