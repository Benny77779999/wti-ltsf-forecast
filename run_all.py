"""Chạy TOÀN BỘ dự án từ đầu đến cuối trong một lệnh: dữ liệu, thí nghiệm, đánh giá, huấn luyện và lưu mô hình.

    python run_all.py --download             # đầy đủ (khoảng 2-3 giờ trên CPU), cần khoá FRED trong .env
    python run_all.py --quick                # chạy thử nhanh vài phút (dùng dữ liệu đã có hoặc thêm --download)
    python run_all.py --skip-data --skip-experiments   # chỉ huấn luyện và lưu mô hình (dữ liệu đã sẵn sàng)
Mỗi bước có thể chạy lại tiếp nếu bị ngắt. Dừng ngay nếu một bước lỗi.
"""
import argparse
import subprocess
import sys
import time


def run(title, *cmd):
    print(f"\n{'=' * 8} {title} {'=' * 8}\n$ python {' '.join(cmd)}", flush=True)
    t0 = time.time()
    if subprocess.call([sys.executable, *cmd]) != 0:
        sys.exit(f"\nBước '{title}' lỗi. Sửa lỗi rồi chạy lại cùng lệnh (các kết quả đã xong được giữ).")
    print(f"-> xong trong {(time.time() - t0) / 60:.1f} phút", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="tải dữ liệu từ FRED (cần FRED_API_KEY)")
    ap.add_argument("--quick", action="store_true", help="chạy thử nhanh với cấu hình rút gọn")
    ap.add_argument("--supp", action="store_true", help="chạy thêm L=480 (bổ sung)")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-experiments", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()
    q = ["--windows", "30", "--test-years", "2023", "2024", "--seeds", "1", "--epochs", "5"] if a.quick else []

    if not a.skip_data:
        run("1. Dữ liệu", "run_data_pipeline.py", *(["--download"] if a.download else []))
    if not a.skip_experiments:
        run("2. Thí nghiệm walk-forward", "run_experiments.py", *q)
        run("3. Cải tiến vòng 1", "run_experiments.py", "--set", "improve1", *(q or ["--windows", "5", "30"]))
        if a.supp and not a.quick:
            run("3b. L=480 (bổ sung)", "run_experiments.py", "--windows", "supp")
        run("4. Đánh giá", "run_evaluation.py", *(["--windows", "30"] if a.quick else []))
        if not a.quick:
            run("4b. Bảng chỉ mô hình gốc", "run_evaluation.py", "--windows", "5", "30", "--exclude-improvements", "--tag", "_round0")
    if not a.skip_train:
        run("5. Huấn luyện và lưu toàn bộ mô hình", "train_all.py", *(["--quick"] if a.quick else []))
    print("\nXong. Mở ứng dụng:  streamlit run app/app.py")


if __name__ == "__main__":
    main()
