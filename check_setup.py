"""Kiểm tra môi trường và cấu hình trước khi chạy dự án, và gợi ý bước tiếp theo.

    python check_setup.py             # kiểm tra cục bộ, không cần mạng
    python check_setup.py --online    # thêm: thử khoá FRED bằng một yêu cầu nhỏ tới máy chủ FRED
Thoát với mã 1 nếu có mục BẮT BUỘC chưa đạt.
"""
import argparse
import importlib
import importlib.metadata as md
import os
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OK, FIX, OPT = "ĐẠT       ", "CẦN SỬA   ", "TUỲ CHỌN  "
rows = []


def add(status, name, detail="", fix=""):
    rows.append((status, name, detail, fix))


def read_env_key():
    """Đọc FRED_API_KEY từ biến môi trường hoặc file .env (không in giá trị ra màn hình)."""
    if os.getenv("FRED_API_KEY"):
        return os.getenv("FRED_API_KEY").strip(), "biến môi trường"
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"\s*FRED_API_KEY\s*=\s*(.*)\s*$", line)
            if m:
                return m.group(1).strip().strip('"').strip("'"), "file .env"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true", help="kiểm tra khoá FRED bằng một yêu cầu thật")
    args = ap.parse_args()

    # 1. Python
    v = sys.version_info
    add(OK if v >= (3, 10) else FIX, f"Python {v.major}.{v.minor}.{v.micro}",
        "" if v >= (3, 10) else "cần 3.10 trở lên", "Cài Python 3.10+ từ python.org")

    # 2. Thư viện
    needed = {"pandas": "pandas", "numpy": "numpy", "requests": "requests", "yaml": "pyyaml",
              "dotenv": "python-dotenv", "statsmodels": "statsmodels", "scipy": "scipy", "matplotlib": "matplotlib"}
    for mod, dist in needed.items():
        try:
            importlib.import_module(mod)
            add(OK, f"Thư viện {dist}", md.version(dist))
        except Exception:
            add(FIX, f"Thư viện {dist}", "chưa cài", "pip install -r requirements.txt")
    for mod, dist, why in (("torch", "torch", "huấn luyện mô hình và 'Dự đoán thử' (trừ naive/drift)"),
                           ("streamlit", "streamlit", "chạy demo"), ("pytest", "pytest", "chạy kiểm thử")):
        try:
            importlib.import_module(mod)
            add(OK, f"Thư viện {dist}", md.version(dist))
        except Exception:
            add(OPT if mod != "torch" else FIX, f"Thư viện {dist}", f"chưa cài (cần để {why})",
                "pip install torch --index-url https://download.pytorch.org/whl/cpu" if mod == "torch"
                else f"pip install {dist}")

    # 3. Khoá FRED
    key, src = read_env_key()
    if not key or key == "your_key_here":
        add(FIX, "FRED_API_KEY", "chưa điền",
            "Sao chép .env.example thành .env, điền khoá (xem docs/SETUP_CONFIG.md, mục 1)")
    elif not re.fullmatch(r"[a-z0-9]{32}", key):
        add(FIX, "FRED_API_KEY", f"có trong {src} nhưng sai định dạng (cần 32 ký tự chữ thường và số)",
            "Kiểm tra lại khoá trên trang FRED, tránh thừa dấu cách hoặc dấu ngoặc")
    else:
        add(OK, "FRED_API_KEY", f"có trong {src}, đúng định dạng")
        if args.online:
            try:
                import requests
                r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                                 params={"series_id": "DCOILWTICO", "api_key": key, "file_type": "json",
                                         "limit": 1, "sort_order": "desc"}, timeout=20)
                if r.status_code == 200:
                    add(OK, "Kết nối FRED", "khoá hợp lệ, tải thử thành công")
                elif r.status_code in (400, 401, 403):
                    add(FIX, "Kết nối FRED", f"FRED từ chối khoá (HTTP {r.status_code})",
                        "Tạo lại khoá trên trang FRED rồi cập nhật .env")
                else:
                    add(FIX, "Kết nối FRED", f"HTTP {r.status_code}", "Thử lại sau; kiểm tra mạng")
            except Exception as e:
                add(FIX, "Kết nối FRED", f"không kết nối được ({type(e).__name__})",
                    "Kiểm tra internet; nếu qua proxy, đặt HTTPS_PROXY trong .env")
        else:
            add(OPT, "Kết nối FRED", "chưa thử", "Chạy lại với --online để thử khoá thật")

    # 4. Cấu hình
    try:
        import yaml
        cfgs = {n: yaml.safe_load((ROOT / "configs" / n).read_text(encoding="utf-8"))
                for n in ("config.yaml", "walk_forward.yaml", "experiments.yaml", "features.yaml", "live.yaml")}
        d = cfgs["config.yaml"]["data"]
        add(OK, "Các file configs/*.yaml", "đọc được cả 5 file")
        end = d.get("end_date")
        if end is not None:
            try:
                e, s = date.fromisoformat(str(end)), date.fromisoformat(str(d["study_start"]))
                add(OK if e > s else FIX, "end_date", f"{end} (study_start {d['study_start']})",
                    "" if e > s else "end_date phải sau study_start")
            except ValueError:
                add(FIX, "end_date", f"'{end}' không phải ngày YYYY-MM-DD", "Sửa trong configs/config.yaml")
        else:
            add(OK, "end_date", "null: dùng dữ liệu mới nhất FRED có")
        if "improve1" not in cfgs["experiments.yaml"]:
            add(OPT, "experiments.yaml: khối improve1", "chưa có (chỉ cần cho --set improve1)",
                "Thêm khối improve1 như trong README")
    except Exception as e:
        add(FIX, "Các file configs/*.yaml", f"lỗi: {e}", "Khôi phục file từ gói gốc")

    # 5. File tuỳ chọn
    add(OPT, "data/raw/fomc_dates.csv", "có" if (ROOT / "data/raw/fomc_dates.csv").exists() else "chưa có",
        "Chỉ cần nếu muốn dùng biến lịch FOMC (hiện chưa dùng trong thí nghiệm)")

    # 6. Trạng thái dữ liệu và kết quả
    raw = [n for n in ("wti", "brent", "usdcad", "usd_broad", "ovx", "fed_funds") if (ROOT / "data/raw" / f"{n}.csv").exists()]
    processed = (ROOT / "data/processed/wti_dataset.csv").exists() and (ROOT / "data/processed/wti_dataset.csv").stat().st_size > 1000
    n_pred = len(list((ROOT / "results/predictions").glob("L*/*.npz"))) if (ROOT / "results/predictions").exists() else 0
    has_eval = bool(list((ROOT / "results/tables").glob("summary_avg_L*.csv"))) if (ROOT / "results/tables").exists() else False
    add(OK if len(raw) == 6 else OPT, "Dữ liệu thô data/raw", f"{len(raw)}/6 file", "python run_data_pipeline.py --download")
    add(OK if processed else OPT, "Dữ liệu sạch data/processed", "có" if processed else "chưa có", "python run_data_pipeline.py")
    add(OK if n_pred else OPT, "Kết quả results/predictions", f"{n_pred} file", "python run_experiments.py")
    n_saved = len(list((ROOT / "saved_models").glob("asof_*/manifest.json"))) if (ROOT / "saved_models").exists() else 0
    add(OK if n_saved else OPT, "Mô hình đã lưu saved_models", f"{n_saved} bộ", "python train_all.py")
    add(OK if has_eval else OPT, "Bảng results/tables", "có" if has_eval else "chưa có", "python run_evaluation.py")

    # In kết quả
    print("\n=== KIỂM TRA MÔI TRƯỜNG VÀ CẤU HÌNH ===\n")
    for status, name, detail, fix in rows:
        line = f"[{status}] {name}" + (f": {detail}" if detail else "")
        print(line)
        if status != OK and fix:
            print(f"             -> {fix}")
    must = [r for r in rows if r[0] == FIX]
    print("\n" + ("Có %d mục BẮT BUỘC cần sửa (xem các dòng CẦN SỬA ở trên)." % len(must) if must
                  else "Mọi mục bắt buộc đã đạt."))
    nxt = ("python run_data_pipeline.py --download" if len(raw) < 6 else
           "python run_data_pipeline.py" if not processed else
           "python run_experiments.py   (hoặc chạy thử nhanh: python run_experiments.py --windows 30 --test-years 2023 --seeds 1)" if not n_pred else
           "python run_evaluation.py" if not has_eval else "streamlit run app/app.py")
    if not must:
        print(f"Bước tiếp theo gợi ý: {nxt}")
    sys.exit(1 if must else 0)


if __name__ == "__main__":
    main()
