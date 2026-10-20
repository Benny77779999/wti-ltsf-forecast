"""Tải và đọc dữ liệu thô (FRED).

Hai cách có dữ liệu thô, kết quả đều là các file data/raw/<tên>.csv gồm 2 cột `date,value`:
  1. Tải tự động qua FRED API:  python run_data_pipeline.py --download
  2. Tải tay file CSV từ trang FRED (nút Download) rồi đổi tên thành <tên>.csv
     (ví dụ wti.csv, brent.csv). File tải tay có cột `observation_date` cũng được hỗ trợ.

Dữ liệu thô không bao giờ bị sửa; mọi xử lý nằm ở preprocessing.py.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import PROJECT_ROOT, resolve_path

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
METADATA_FILE = "_metadata.json"


def get_api_key() -> str:
    """Lấy khoá FRED API từ biến môi trường FRED_API_KEY (hoặc file .env)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "Chưa có FRED_API_KEY. Sao chép .env.example thành .env và điền khoá, "
            "hoặc tải tay CSV từ trang FRED vào data/raw/."
        )
    return key


def fetch_fred_series(series_id: str, api_key: str, start: str, end: str | None = None,
                      retries: int = 3, timeout: int = 30) -> pd.Series:
    """Tải một chuỗi từ FRED API. Giá trị '.' (thiếu) được đổi thành NaN."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    if end:
        params["observation_end"] = end

    for attempt in range(retries):
        try:
            resp = requests.get(FRED_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    obs = resp.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["value"].sort_index()
    s.name = series_id
    return s


def download_all(cfg: dict, api_key: str | None = None) -> dict:
    """Tải toàn bộ chuỗi khai báo trong cfg và lưu vào data/raw/ kèm metadata."""
    api_key = api_key or get_api_key()
    d = cfg["data"]
    raw_dir = resolve_path(cfg, "raw_dir")
    meta = {}
    for name, series_id in d["fred_series"].items():
        s = fetch_fred_series(series_id, api_key, d["download_start"], d.get("end_date"))
        s.rename("value").rename_axis("date").to_csv(raw_dir / f"{name}.csv")
        meta[name] = {
            "fred_series_id": series_id,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_obs": int(len(s)),
            "n_non_null": int(s.notna().sum()),
            "first_date": str(s.index.min().date()),
            "last_date": str(s.index.max().date()),
        }
        print(f"[OK] {name:10s} ({series_id}): {meta[name]['n_non_null']} quan sát, "
              f"{meta[name]['first_date']} -> {meta[name]['last_date']}")
    with open(raw_dir / METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def read_raw_file(path) -> pd.Series:
    """Đọc một file CSV thô (2 cột: ngày, giá trị) thành Series theo ngày."""
    df = pd.read_csv(path)
    date_col, value_col = df.columns[0], df.columns[1]
    dates = pd.to_datetime(df[date_col])
    values = pd.to_numeric(df[value_col], errors="coerce")  # '.' hoặc rỗng -> NaN
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=path.stem)
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_raw(cfg: dict) -> pd.DataFrame:
    """Đọc mọi chuỗi thô thành một DataFrame (join ngoài theo ngày, CHƯA căn chỉnh)."""
    d = cfg["data"]
    raw_dir = resolve_path(cfg, "raw_dir")
    missing = [n for n in d["fred_series"] if not (raw_dir / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu file thô trong {raw_dir}: {[m + '.csv' for m in missing]}. "
            "Chạy `python run_data_pipeline.py --download` hoặc tải tay từ FRED."
        )
    series = {n: read_raw_file(raw_dir / f"{n}.csv") for n in d["fred_series"]}
    raw = pd.concat(series, axis=1).sort_index()
    raw.index.name = "date"
    raw = raw.loc[raw.index >= pd.Timestamp(d["download_start"])]
    if d.get("end_date"):
        raw = raw.loc[raw.index <= pd.Timestamp(d["end_date"])]
    return raw
