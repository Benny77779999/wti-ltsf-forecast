"""Chạy toàn bộ pipeline dữ liệu: (tải) -> đọc thô -> căn chỉnh -> kiểm tra -> lưu.

Ví dụ:
    python run_data_pipeline.py --download     # tải bằng FRED API rồi xử lý
    python run_data_pipeline.py                # dùng file thô có sẵn trong data/raw/
"""
import argparse

from src.config import load_config, resolve_path
from src.data_loader import download_all, load_raw
from src.preprocessing import (adf_report, build_dataset, lead_lag_check,
                               missing_business_days, quality_report, save_processed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="tải dữ liệu từ FRED API")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.download:
        download_all(cfg)

    raw = load_raw(cfg)
    processed = build_dataset(raw, cfg, apply_lags=True)
    nolag = build_dataset(raw, cfg, apply_lags=False)  # chỉ để chẩn đoán lệch ngày

    tables = resolve_path(cfg, "tables_dir")
    series_tbl, summary_tbl = quality_report(raw, processed, cfg)
    series_tbl.to_csv(tables / "quality_series.csv", index=False)
    summary_tbl.to_csv(tables / "quality_summary.csv")

    missing_business_days(processed.index).to_csv(tables / "missing_business_days_by_year.csv")

    others = [c for c in ("brent_logret", "spread_diff", "usdcad_logret",
                          "usd_broad_logret", "ovx_diff", "fed_diff") if c in nolag.columns]
    lead_lag_check(nolag, others).to_csv(tables / "lead_lag_check.csv")
    adf_report(processed, [c for c in ("wti", "wti_logret") if c in processed.columns]
               ).to_csv(tables / "adf_report.csv", index=False)

    path = save_processed(processed, cfg)
    print("\n=== Tổng hợp chất lượng ===")
    print(summary_tbl.to_string())
    print(f"\nĐã lưu: {path}")


if __name__ == "__main__":
    main()
