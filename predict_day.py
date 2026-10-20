"""Chọn mô hình và dự đoán thử 5 phiên sau một ngày bất kỳ. Mô hình THỰC SỰ chạy (không đọc kết quả có sẵn).

Ví dụ:
    python predict_day.py --list
    python predict_day.py --model nlinear --L 30 --date 2024-06-14
    python predict_day.py --model hybrid_C_usd --L 60 --date 2025-03-03 --seed 1 --no-cache
"""
import argparse
import sys

import numpy as np
import pandas as pd

from src.live_predict import LivePredictError, load_all_configs, load_frame, model_catalog, predict_day
from src.config import resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="liệt kê các mô hình có thể chọn")
    ap.add_argument("--model", default=None)
    ap.add_argument("--L", type=int, default=None, help="số phiên quá khứ đưa vào mô hình")
    ap.add_argument("--date", default=None, help="ngày dự báo, định dạng YYYY-MM-DD (lấy phiên gần nhất trước hoặc đúng ngày)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true", help="luôn huấn luyện lại, không dùng mô hình đã lưu")
    ap.add_argument("--device", default=None, help="cpu | cuda | auto")
    args = ap.parse_args()

    cfg, wf, exp, live = load_all_configs()
    catalog = model_catalog(exp)
    if args.list:
        for name, spec in catalog.items():
            kind = {"baseline": "không huấn luyện", "model": "huấn luyện", "tuned": "huấn luyện + chọn weight decay"}[spec["kind"]]
            print(f"{name:26s} {kind:34s} đầu vào: {', '.join(spec['cols'])}")
        return

    model = args.model or live.get("default_model", "nlinear")
    L = args.L or live.get("default_L", 30)
    seed = args.seed if args.seed is not None else live.get("default_seed", 0)
    device = args.device or live.get("device", "cpu")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        df = load_frame(cfg)
        date = args.date or str(df.index[-1].date())
        cache_dir = resolve_path({"paths": {"c": live.get("cache_dir", "saved_models/live")}}, "c")
        print(f"Đang dự báo: mô hình={model}, L={L}, ngày={date}, seed={seed} ...", flush=True)
        r = predict_day(df, model, L, date, seed=seed, exp=exp, wf=wf, cfg=cfg, cache_dir=cache_dir,
                        use_cache=not args.no_cache, device=device)
    except LivePredictError as e:
        print(f"\nKhông dự báo được: {e}")
        sys.exit(1)

    print("\n=== KẾT QUẢ ===")
    print(f"Mô hình           : {r['model']}  (seed {r['seed']}, {'nạp từ bộ nhớ đã lưu' if r['from_cache'] else 'vừa huấn luyện'}, {r['seconds']:.1f} giây)")
    print(f"Ngày dự báo       : {r['origin'].date()} (phiên cuối của cửa sổ), giá đóng cửa {r['last_price']:.2f} USD")
    if r["kind"] != "baseline":
        print(f"Cửa sổ đưa vào    : {r['L']} phiên, từ {r['input_dates'][0].date()} đến {r['input_dates'][-1].date()}")
        i = r["info"]
        print(f"Huấn luyện bằng   : {i['n_train']} mẫu (validation {i['n_val']}), nhãn cuối {i['train_end_date']}, "
              f"weight decay {i['chosen_wd']}, epoch tốt nhất {i['best_epoch']}")
        print("Dữ liệu sau ngày dự báo KHÔNG được dùng để huấn luyện hay dự báo.")
    rows = []
    for k in range(len(r["forecast"])):
        a = r["actual"][k]
        rows.append({"phiên": f"t+{k + 1}", "ngày": r["dates"][k].date(), "dự báo": r["forecast"][k],
                     "naive": r["naive"][k], "thực tế": a,
                     "sai số %": 100 * (r["forecast"][k] / a - 1) if np.isfinite(a) else np.nan,
                     "sai số naive %": 100 * (r["naive"][k] / a - 1) if np.isfinite(a) else np.nan})
    tbl = pd.DataFrame(rows)
    print("\n" + tbl.to_string(index=False, float_format=lambda v: f"{v:.2f}", na_rep="-"))
    if r["dates_projected"]:
        print("\nLưu ý: một số ngày ở trên được suy ra theo lịch làm việc (xấp xỉ) vì dữ liệu chưa có tới ngày đó.")
    if np.isfinite(r["actual"]).any():
        print("Cột 'thực tế' chỉ được lấy để so sánh SAU khi mô hình đã dự báo.")
    else:
        print("Chưa có giá thực tế cho các phiên này (đây là dự báo cho tương lai theo dữ liệu hiện có).")
    print("Đây là sản phẩm nghiên cứu: kết quả của đồ án cho thấy mô hình không vượt dự báo 'giữ nguyên giá' (naive).")


if __name__ == "__main__":
    main()
