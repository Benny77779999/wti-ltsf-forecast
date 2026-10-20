"""Huấn luyện TOÀN BỘ mô hình trong một lần chạy và lưu vào saved_models/ để dùng lại.

    python train_all.py                      # mọi mô hình, L và mốc cắt theo configs/live.yaml (mục store)
    python train_all.py --quick              # chạy thử nhanh: L=30, 5 epoch
    python train_all.py --windows 30 60 --cutoffs 2024-12-31 last --names nlinear hybrid_B
Chạy lại sẽ bỏ qua mô hình đã lưu (dùng --overwrite để huấn luyện lại).
"""
import argparse
import time

from src.live_predict import load_all_configs, load_frame
from src.model_store import train_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", nargs="+", default=None, help="ngày cắt YYYY-MM-DD hoặc 'last'")
    ap.add_argument("--windows", nargs="+", type=int, default=None)
    ap.add_argument("--names", nargs="+", default=None, help="chỉ huấn luyện các mô hình này")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg, wf, exp, live = load_all_configs()
    if a.quick:
        a.windows, a.epochs = a.windows or [30], a.epochs or 5
    device = a.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_frame(cfg)
    t0 = time.time()
    dirs = train_all(df, cfg, wf, exp, live, cutoffs=a.cutoffs, windows=a.windows, seed=a.seed, device=device,
                     names=a.names, overwrite=a.overwrite, epochs=a.epochs, log=lambda m: print(m, flush=True))
    print(f"\nXong trong {(time.time() - t0) / 60:.1f} phút. Các bộ mô hình đã lưu:")
    for d in dirs:
        print(f"  {d}")
    print("Dùng: streamlit run app/app.py (tab đầu) hoặc python predict_saved.py --list")


if __name__ == "__main__":
    main()
