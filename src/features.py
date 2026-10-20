"""Biến lịch (biết trước): thứ trong tuần, ngày công bố tồn kho EIA, ngày FOMC, đáo hạn CL.

Các quy tắc dưới đây là XẤP XỈ theo hiểu biết của tác giả, cần đối chiếu với lịch chính thức:
  * Lịch nghỉ lấy từ lịch liên bang Mỹ (NYMEX có vài ngày nghỉ khác, ví dụ Thứ Sáu Tuần Thánh).
  * EIA: thứ Tư; nếu thứ Tư là ngày nghỉ liên bang thì dời sang thứ Năm. Các trường hợp
    thứ Hai nghỉ lễ chưa xử lý riêng, cần đối chiếu lịch phát hành của EIA.
  * Đáo hạn CL: 3 ngày làm việc trước ngày 25 của tháng liền trước tháng giao hàng; nếu ngày 25
    không phải ngày làm việc thì tính từ ngày làm việc liền trước đó.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from src.config import PROJECT_ROOT, load_config

_HOL = USFederalHolidayCalendar().holidays(start="2000-01-01", end="2035-12-31")
_HOL_DATES = {h.date() for h in _HOL}
_HOL_D64 = np.array([h.date() for h in _HOL], dtype="datetime64[D]")
_CBD = CustomBusinessDay(holidays=_HOL)


def _expiry(year: int, month: int) -> pd.Timestamp:
    """Ngày ngừng giao dịch của hợp đồng đáo hạn trong tháng `month` (ngày 25 của tháng đó)."""
    d = _CBD.rollback(pd.Timestamp(year, month, 25))
    return d - 3 * _CBD


def days_to_expiry(index) -> pd.Series:
    """Số ngày làm việc từ mỗi ngày tới ngày đáo hạn gần nhất chưa qua."""
    out = []
    for t in pd.DatetimeIndex(index):
        e = _expiry(t.year, t.month)
        if t.normalize() > e:
            y, m = (t.year + 1, 1) if t.month == 12 else (t.year, t.month + 1)
            e = _expiry(y, m)
        out.append(int(np.busday_count(t.date(), e.date(), holidays=_HOL_D64)))
    return pd.Series(out, index=pd.DatetimeIndex(index), name="days_to_expiry")


def _is_eia_day(t: pd.Timestamp) -> bool:
    if t.weekday() not in (2, 3):
        return False
    wed = t.normalize() - pd.Timedelta(days=t.weekday() - 2)
    release = wed + pd.Timedelta(days=1) if wed.date() in _HOL_DATES else wed
    return t.normalize() == release


def load_fomc_dates(path) -> pd.DatetimeIndex | None:
    path = Path(path)
    if not path.exists():
        return None
    return pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path)["date"])).normalize()


def build_calendar_features(index, cfg: dict | None = None) -> pd.DataFrame:
    """Trả về DataFrame biến lịch theo `index` (các giá trị đều đã được biết trước)."""
    cfg = cfg or load_config(PROJECT_ROOT / "configs" / "features.yaml")
    idx = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=idx)
    for k in range(5):
        out[f"dow_{k}"] = (idx.weekday == k).astype(float)
    out["is_eia_day"] = [float(_is_eia_day(t)) for t in idx]
    dte = days_to_expiry(idx)
    out["days_to_expiry"] = dte.astype(float)
    out["near_expiry"] = (dte <= cfg["near_expiry_days"]).astype(float)

    fomc = load_fomc_dates(PROJECT_ROOT / cfg["fomc_dates_file"])
    if fomc is None:
        warnings.warn("Chưa có file ngày FOMC, bỏ biến is_fomc_day.", stacklevel=2)
    else:
        out["is_fomc_day"] = idx.normalize().isin(fomc).astype(float)
    return out


def add_calendar_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    return df.join(build_calendar_features(df.index, cfg))
