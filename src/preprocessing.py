"""Căn chỉnh, làm trễ, gắn cờ chất lượng và tạo biến phái sinh.

Nguyên tắc chống rò rỉ (xem docs/data_timing_rules.md):
  * Lịch gốc = các ngày có giá WTI hợp lệ.
  * Biến phụ trợ chỉ nhận quan sát gần nhất TRƯỚC (hoặc đúng) ngày đó, không bao giờ dùng giá sau.
  * Quan sát quá cũ (> max_stale_days) bị bỏ thành NaN thay vì điền mãi.
  * Biến chốt giá sau WTI bị làm trễ `lag_rules` phiên giao dịch.
  * Biến phái sinh (return, chênh lệch) tính từ chuỗi ĐÃ làm trễ.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config import resolve_path


# ----------------------------------------------------------------------------
# Căn chỉnh
# ----------------------------------------------------------------------------
def align_to_calendar(series: pd.Series, calendar: pd.DatetimeIndex,
                      max_stale_days: int) -> pd.Series:
    """Lấy, tại mỗi ngày của `calendar`, quan sát hợp lệ gần nhất tại hoặc trước ngày đó."""
    calendar = pd.DatetimeIndex(calendar)
    s = series.dropna()
    if s.empty:
        return pd.Series(np.nan, index=calendar, name=series.name)

    left = pd.DataFrame({"date": calendar.astype("datetime64[ns]")})
    obs_dates = s.index.astype("datetime64[ns]")
    right = pd.DataFrame({"date": obs_dates, "value": s.to_numpy(), "obs_date": obs_dates})
    merged = pd.merge_asof(left.sort_values("date"), right.sort_values("date"),
                           on="date", direction="backward")
    age_days = (merged["date"] - merged["obs_date"]).dt.days
    merged.loc[age_days > max_stale_days, "value"] = np.nan
    return pd.Series(merged["value"].to_numpy(), index=calendar, name=series.name)


def _run_length(s: pd.Series) -> pd.Series:
    """Độ dài đoạn giá lặp liên tiếp mà mỗi dòng thuộc về."""
    groups = (s != s.shift()).cumsum()
    return s.groupby(groups).transform("size")


# ----------------------------------------------------------------------------
# Dựng bộ dữ liệu
# ----------------------------------------------------------------------------
def build_dataset(raw: pd.DataFrame, cfg: dict, apply_lags: bool = True) -> pd.DataFrame:
    """Từ dữ liệu thô sang bộ dữ liệu đã căn chỉnh theo lịch của WTI.

    apply_lags=False chỉ dùng để chẩn đoán lệch ngày (lead_lag_check), không dùng để huấn luyện.
    """
    d = cfg["data"]
    target = d["target"]
    raw = raw.sort_index()

    calendar = raw.index[raw[target].notna()]
    aligned = pd.DataFrame(index=calendar)
    for col in raw.columns:
        stale = 0 if col == target else d["max_stale_days"]
        aligned[col] = align_to_calendar(raw[col], calendar, stale)

    if apply_lags:
        for col, k in d["lag_rules"].items():
            if col == target and k != 0:
                raise ValueError("Không được làm trễ biến mục tiêu.")
            if k and col in aligned.columns:
                aligned[col] = aligned[col].shift(k)

    out = aligned.copy()
    wti = aligned[target]
    pos_price = wti.where(wti > 0)

    # --- Biến phái sinh (tính từ chuỗi đã làm trễ) ---
    out["wti_diff"] = wti.diff()                       # USD/thùng, hợp lệ cả khi giá âm
    out["wti_logret"] = np.log(pos_price).diff()       # NaN nếu một trong hai ngày giá <= 0
    if "brent" in aligned:
        out["brent_logret"] = np.log(aligned["brent"].where(aligned["brent"] > 0)).diff()
        out["spread_brent_wti"] = aligned["brent"] - wti   # chênh lệch mức giá, không log
        out["spread_diff"] = out["spread_brent_wti"].diff()
    if "usdcad" in aligned:
        out["usdcad_logret"] = np.log(aligned["usdcad"]).diff()
    if "usd_broad" in aligned:
        out["usd_broad_logret"] = np.log(aligned["usd_broad"]).diff()
    if "ovx" in aligned:
        out["ovx_diff"] = aligned["ovx"].diff()
    if "fed_funds" in aligned:
        out["fed_diff"] = aligned["fed_funds"].diff()

    # --- Cờ chất lượng và giai đoạn ---
    idx = out.index
    nonpos = wti <= 0
    pct = wti.pct_change().abs()
    out["flag_nonpositive"] = nonpos
    out["flag_jump"] = (pct > d["jump_threshold"]) | nonpos | nonpos.shift(fill_value=False)
    out["wti_run_length"] = _run_length(wti)
    out["flag_stale_run"] = out["wti_run_length"] >= d["repeat_run_threshold"]

    ex = d["exclusion_window"]
    out["is_excluded"] = (idx >= pd.Timestamp(ex["start"])) & (idx <= pd.Timestamp(ex["end"]))
    out["is_buffer"] = idx < pd.Timestamp(d["study_start"])
    # gap_before = True nếu phiên liền trước cách quá max_gap_days ngày lịch (đoạn thiếu dữ liệu)
    gap_days = pd.Series(idx, index=idx).diff().dt.days
    out["gap_before"] = (gap_days > d.get("max_gap_days", 5)).to_numpy()

    out.index.name = "date"
    return out


# ----------------------------------------------------------------------------
# Báo cáo chất lượng (dùng cho chương 3)
# ----------------------------------------------------------------------------
def quality_report(raw: pd.DataFrame, processed: pd.DataFrame, cfg: dict):
    """Trả về (bảng theo chuỗi, bảng tổng hợp) cho phần 'trước và sau tiền xử lý'."""
    cal = processed.index
    rows = []
    for col in raw.columns:
        s = raw[col]
        rows.append({
            "series": col,
            "fred_id": cfg["data"]["fred_series"].get(col),
            "raw_non_null": int(s.notna().sum()),
            "raw_first": s.first_valid_index(),
            "raw_last": s.last_valid_index(),
            "missing_on_wti_calendar_before": int(s.reindex(cal).isna().sum()),
            "missing_after_align_and_lag": int(processed[col].isna().sum()),
        })
    series_table = pd.DataFrame(rows)

    usable = ~processed["is_excluded"] & ~processed["is_buffer"]
    summary = pd.Series({
        "n_trading_days": len(processed),
        "first_date": processed.index.min(),
        "last_date": processed.index.max(),
        "n_wti_nonpositive": int(processed["flag_nonpositive"].sum()),
        "n_jump_flags": int(processed["flag_jump"].sum()),
        "n_stale_run_days": int(processed["flag_stale_run"].sum()),
        "n_gaps_over_threshold": int(processed["gap_before"].sum()),
        "n_excluded_days": int(processed["is_excluded"].sum()),
        "n_buffer_days": int(processed["is_buffer"].sum()),
        "n_usable_study_days": int(usable.sum()),
    }, name="value").to_frame()
    return series_table, summary


def missing_business_days(index) -> pd.Series:
    """Số ngày làm việc kỳ vọng (trừ lễ liên bang Mỹ và Thứ Sáu Tuần Thánh) không có trong dữ liệu,
    theo năm. Chỉ mang tính tham khảo vì lịch nghỉ của nguồn có thể khác chút ít."""
    from pandas.tseries.holiday import GoodFriday, USFederalHolidayCalendar

    idx = pd.DatetimeIndex(index)
    hol = pd.DatetimeIndex(USFederalHolidayCalendar().holidays(idx.min(), idx.max()))
    hol = hol.union(pd.DatetimeIndex(GoodFriday.dates(idx.min(), idx.max())))
    expected = pd.bdate_range(idx.min(), idx.max(), freq="C", holidays=list(hol))
    missing = expected.difference(idx)
    years = range(idx.min().year, idx.max().year + 1)
    counts = pd.Series(1, index=missing).groupby(missing.year).sum() if len(missing) else pd.Series(dtype=int)
    return counts.reindex(years, fill_value=0).rename("missing_business_days")


def lead_lag_check(df_nolag: pd.DataFrame, cols: list[str], target: str = "wti_logret",
                   max_lag: int = 2) -> pd.DataFrame:
    """Tương quan giữa target tại t và biến x tại t+k, k = -max_lag..+max_lag.

    Chạy trên dữ liệu CHƯA làm trễ (build_dataset(apply_lags=False)).
    k = 0: đồng thời. Tương quan lớn ở k != 0 gợi ý lệch ngày giữa hai chuỗi cần xem lại.
    Chỉ là công cụ chẩn đoán, không phải bằng chứng nhân quả.
    """
    use = df_nolag.loc[~df_nolag["is_excluded"] & ~df_nolag["is_buffer"]]
    table = {}
    for col in cols:
        table[col] = {k: use[target].corr(use[col].shift(-k))
                      for k in range(-max_lag, max_lag + 1)}
    out = pd.DataFrame(table).T
    out.columns = [f"x(t{k:+d})" for k in out.columns]
    return out


def adf_report(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Kiểm định ADF (H0: có nghiệm đơn vị) trên giai đoạn nghiên cứu, bỏ giai đoạn loại."""
    from statsmodels.tsa.stattools import adfuller

    use = df.loc[~df["is_excluded"] & ~df["is_buffer"]]
    rows = []
    for col in cols:
        x = use[col].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            res = adfuller(x, autolag="AIC")
        stat, pval = res[0], res[1]
        rows.append({"column": col, "n": len(x), "adf_stat": stat, "p_value": pval,
                     "stationary_at_5pct": bool(pval < 0.05)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Từ điển dữ liệu và lưu file
# ----------------------------------------------------------------------------
_DESCRIPTIONS = {
    "wti": "Giá spot WTI Cushing (USD/thùng) - biến mục tiêu",
    "brent": "Giá spot Brent (USD/thùng)",
    "usdcad": "Tỷ giá CAD trên mỗi USD",
    "usd_broad": "Chỉ số USD broad của Fed",
    "ovx": "Biến động ngụ ý của dầu (OVX)",
    "fed_funds": "Lãi suất FED hiệu dụng (%)",
    "wti_diff": "Thay đổi giá WTI so với phiên trước (USD)",
    "wti_logret": "Log-return WTI (NaN khi giá <= 0)",
    "brent_logret": "Log-return Brent",
    "spread_brent_wti": "Brent - WTI (USD)",
    "spread_diff": "Thay đổi spread Brent-WTI so với phiên trước",
    "usdcad_logret": "Log-return USD/CAD",
    "usd_broad_logret": "Log-return chỉ số USD broad",
    "ovx_diff": "Thay đổi OVX so với phiên trước",
    "fed_diff": "Thay đổi lãi suất FED so với phiên trước",
    "flag_nonpositive": "Giá WTI <= 0",
    "flag_jump": "Biến động 1 ngày vượt ngưỡng hoặc liên quan giá không dương",
    "wti_run_length": "Độ dài đoạn giá WTI lặp liên tiếp",
    "flag_stale_run": "Đoạn lặp giá >= ngưỡng (nghi lỗi nguồn)",
    "is_excluded": "Thuộc giai đoạn bị loại (giá âm 2020)",
    "is_buffer": "Thuộc giai đoạn đệm, trước study_start",
    "gap_before": "Phiên liền trước cách quá max_gap_days ngày lịch (đoạn thiếu dữ liệu)",
}


def make_data_dictionary(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    rows = []
    for col in df.columns:
        base = col if col in d["fred_series"] else None
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "fred_id": d["fred_series"].get(base) if base else "",
            "lag_phien": d["lag_rules"].get(base, "") if base else "",
            "description": _DESCRIPTIONS.get(col, ""),
        })
    return pd.DataFrame(rows)


def save_processed(df: pd.DataFrame, cfg: dict, name: str = "wti_dataset.csv"):
    out_dir = resolve_path(cfg, "processed_dir")
    df.to_csv(out_dir / name)
    make_data_dictionary(df, cfg).to_csv(out_dir / "data_dictionary.csv", index=False)
    return out_dir / name
