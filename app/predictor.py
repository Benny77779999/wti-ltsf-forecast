"""Đọc các dự báo đã lưu (results/predictions) để dựng demo. Không cần torch.

Mỗi file L{L}/{tên}__{năm}__s{seed}.npz chứa dự báo cho mọi origin của năm test đó.
Với mô hình có nhiều seed, demo dùng dự báo TRUNG BÌNH qua các seed.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_PAT = re.compile(r"^(?P<name>.+)__(?P<year>\d{4})__s(?P<seed>\d+)$")

LABELS = {
    "naive": "Naive (giữ giá đóng cuối)", "drift": "Random walk có drift",
    "linear": "Linear", "nlinear": "NLinear", "dlinear": "DLinear", "hybrid_B": "Lai NLinear+DLinear (B)",
}
ORDER = ["naive", "drift", "linear", "nlinear", "dlinear", "hybrid_B"]


def label(name: str) -> str:
    if name.endswith("_shrink"):
        return label(name[:-7]) + " + co về naive"
    if name.endswith("_mv"):
        return label(name[:-3]) + " đa biến"
    if name.endswith("_zr"):
        return label(name[:-3]) + " (khởi tạo 0 + ridge)"
    if name in LABELS:
        return LABELS[name]
    if name.startswith("hybrid_C_"):
        rest = name[len("hybrid_C_"):]
        lvl = rest.endswith("_lvl")
        group = rest[:-4] if lvl else rest
        return f"Lai đa biến (C): {group}" + (" + nhánh mức" if lvl else "")
    return name


class Forecasts:
    def __init__(self, pred_dir):
        self.dir = Path(pred_dir)
        self._cache: dict = {}

    def windows(self) -> list:
        return sorted(int(p.name[1:]) for p in self.dir.glob("L*") if p.is_dir() and p.name[1:].isdigit())

    def models(self, L: int) -> list:
        names = {m["name"] for f in (self.dir / f"L{L}").glob("*.npz") if (m := _PAT.match(f.stem))}
        known = [n for n in ORDER if n in names]
        return known + sorted(n for n in names if n not in ORDER)

    def years(self, L: int) -> list:
        return sorted({int(m["year"]) for f in (self.dir / f"L{L}").glob("naive__*.npz")
                       if (m := _PAT.match(f.stem))})

    def _load(self, L: int, name: str, year: int):
        key = (L, name, year)
        if key not in self._cache:
            files = sorted((self.dir / f"L{L}").glob(f"{name}__{year}__s*.npz"))
            if not files:
                self._cache[key] = None
            else:
                arrs = [np.load(f) for f in files]
                self._cache[key] = {
                    "origin": arrs[0]["origin"].astype("datetime64[ns]"),
                    "pred_log": np.mean([a["pred_log"] for a in arrs], axis=0),
                    "actual_log": arrs[0]["actual_log"], "last_log": arrs[0]["last_log"],
                    "n_seeds": len(files),
                }
        return self._cache[key]

    def origins(self, L: int, year: int) -> pd.DatetimeIndex:
        r = self._load(L, "naive", year)
        return pd.DatetimeIndex([]) if r is None else pd.DatetimeIndex(r["origin"])

    def forecast_at(self, L: int, name: str, origin) -> dict | None:
        origin = pd.Timestamp(origin)
        r = self._load(L, name, origin.year)
        if r is None:
            return None
        hit = np.where(r["origin"] == np.datetime64(origin, "ns"))[0]
        if len(hit) == 0:
            return None
        i = hit[0]
        return {"pred": np.exp(r["pred_log"][i]), "actual": np.exp(r["actual_log"][i]),
                "last": float(np.exp(r["last_log"][i])), "n_seeds": r["n_seeds"]}

    def forecast_table(self, L: int, origin, names: list) -> pd.DataFrame:
        """Mỗi mô hình một dòng: giá dự báo tau=1..T (USD), sai số % trung bình và sai số % ở tau cuối."""
        rows = []
        for n in names:
            f = self.forecast_at(L, n, origin)
            if f is None:
                continue
            err = f["pred"] / f["actual"] - 1.0
            row = {"model": label(n), "_key": n}
            row.update({f"t+{j + 1}": float(p) for j, p in enumerate(f["pred"])})
            row["sai số TB (%)"] = float(100 * np.mean(np.abs(err)))
            row["sai số t+5 (%)"] = float(100 * err[-1])
            rows.append(row)
        return pd.DataFrame(rows)


def history_window(df: pd.DataFrame, origin, n: int = 30) -> pd.Series:
    s = df["wti"].loc[:pd.Timestamp(origin)]
    return s.iloc[-n:]


def future_dates(df: pd.DataFrame, origin, T: int = 5) -> pd.DatetimeIndex:
    pos = df.index.get_loc(pd.Timestamp(origin))
    return df.index[pos + 1:pos + 1 + T]
