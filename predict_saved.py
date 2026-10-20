"""Dự báo bằng mô hình ĐÃ LƯU (không huấn luyện lại).

    python predict_saved.py --list
    python predict_saved.py --model nlinear --L 30 --date 2026-03-10
    python predict_saved.py --model all --L 30                 # mọi mô hình đã lưu, ngày cuối của dữ liệu
    python predict_saved.py --set asof_20251231 --model hybrid_B --L 120 --date 2026-05-04
"""
import argparse
import sys

import numpy as np
import pandas as pd

from src.live_predict import LivePredictError, load_all_configs, load_frame
from src.model_store import available_models, list_sets, predict_saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--set", default=None, help="tên bộ mô hình (mặc định: mới nhất)")
    ap.add_argument("--model", default="nlinear", help="tên mô hình hoặc 'all'")
    ap.add_argument("--L", type=int, default=30)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (mặc định: ngày cuối của dữ liệu)")
    a = ap.parse_args()
    cfg, wf, exp, live = load_all_configs()
    sets = list_sets(live)
    if a.list or not sets:
        if not sets:
            print("Chưa có mô hình đã lưu. Chạy: python train_all.py")
            sys.exit(0 if a.list else 1)
        for s in sets:
            print(f"{s['tag']}: huấn luyện đến {s['cutoff']}, L = {s['windows']}, {len(s['models'])} mô hình, seed {s['seed']}")
        return
    man = next((s for s in sets if s["tag"] == a.set), None) if a.set else sets[0]
    if man is None:
        sys.exit(f"Không có bộ mô hình {a.set}. Dùng --list.")
    df = load_frame(cfg)
    date = a.date or str(df.index[-1].date())
    names = available_models(man, a.L) if a.model == "all" else [a.model]
    rows, first = [], None
    for n in names:
        try:
            r = predict_saved(df, man, exp, n, a.L, date)
        except LivePredictError as e:
            print(f"{n}: không dự báo được: {e}")
            continue
        first = first or r
        rows.append({"mô hình": n, **{f"t+{k + 1}": v for k, v in enumerate(r["forecast"])}})
    if not first:
        sys.exit(1)
    print(f"Bộ mô hình {man['tag']} (huấn luyện đến {man['cutoff']}); L={a.L}; ngày dự báo {first['origin'].date()}, giá {first['last_price']:.2f} USD")
    note = {"trong mẫu": "TRONG MẪU (5 phiên mục tiêu đã nằm trong dữ liệu huấn luyện/chọn epoch, không dùng để đánh giá)",
            "một phần": "MỘT PHẦN trong mẫu (ngày cắt nằm giữa các phiên mục tiêu)",
            "ngoài mẫu": "NGOÀI MẪU (mô hình chưa thấy các phiên mục tiêu)"}
    print("Trạng thái:", note[first["status"]])
    print("Ngày các phiên:", ", ".join(str(d.date()) for d in first["dates"]) + (" (một phần suy ra theo lịch)" if first["dates_projected"] else ""))
    tbl = pd.DataFrame(rows)
    act = {f"t+{k + 1}": v for k, v in enumerate(first["actual"])}
    tbl = pd.concat([tbl, pd.DataFrame([{"mô hình": "thực tế", **act}])], ignore_index=True)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.2f}", na_rep="chưa có"))


if __name__ == "__main__":
    main()
