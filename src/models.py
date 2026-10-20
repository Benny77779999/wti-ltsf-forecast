"""Các mô hình LTSF-Linear: Linear, NLinear, DLinear (nhóm A) và Hybrid (nhóm B, C).

Quy ước đầu vào/đầu ra:
  * Linear, NLinear, DLinear: x [B, L]    -> [B, T]  (đúng như tài liệu)
  * Hybrid:                   x [B, L] hoặc [B, L, C] -> [B, T]; kênh 0 là chuỗi cần dự báo.

Hybrid = NLinear + DLinear: trừ giá cuối của từng kênh, tách xu hướng/phần dư bằng trung bình
trượt, mỗi nhánh một lớp tuyến tính, cộng hai nhánh rồi cộng lại giá cuối của kênh 0.
  * n_channels = 1  -> nhóm B (lai đơn biến)
  * n_channels > 1  -> nhóm C (lai đa biến, trộn kênh: mỗi nhánh nhận L*C giá trị)
  * level_branch=True (chỉ có tác dụng khi n_channels > 1): thêm một lớp tuyến tính nhận MỨC hiện tại
    (giá trị cuối, đã chuẩn hoá) của các kênh phụ. Lý do: bước trừ giá cuối chỉ giữ lại diễn biến
    gần đây của kênh phụ và làm mất mức của nó (ví dụ mức spread Brent-WTI, mức OVX).
    Mức được cắt về [-level_clip, level_clip] để một biến bị ngoại suy (ví dụ lãi suất FED tăng
    mạnh sau giai đoạn train) không làm mô hình mất ổn định.

Ghi chú về thứ tự "trừ mốc rồi phân rã" và "phân rã rồi trừ mốc": trung bình trượt với
replicate padding là phép tịnh tiến-tương thích, MA(x - c) = MA(x) - c, nên phần dư
x - MA(x) không đổi và hai thứ tự cho cùng kết quả (xem tests/test_models.py).

Khác biệt có chủ đích so với bản trong tài liệu: cửa sổ trung bình trượt phải LẺ và không bị
cắt xuống seq_len - 1. Với seq_len = 5, m = 5 vẫn hợp lệ (replicate padding 2 phía), nên không
cần nội suy xấp xỉ như trường hợp m = 4 trong tài liệu.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_(*layers) -> None:
    """Đặt trọng số và bias của các lớp về 0: mô hình bắt đầu đúng bằng dự báo naive (giá cuối)."""
    for layer in layers:
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)


class MovingAverage(nn.Module):
    """Trung bình trượt dọc trục thời gian, replicate padding. x: [B, L, C] -> [B, L, C]."""

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("moving_avg phải là số lẻ dương.")
        self.k = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.k - 1) // 2
        if pad > 0:
            x = torch.cat([x[:, :1, :].repeat(1, pad, 1), x, x[:, -1:, :].repeat(1, pad, 1)], dim=1)
        return F.avg_pool1d(x.transpose(1, 2), kernel_size=self.k, stride=1).transpose(1, 2)


class Linear(nn.Module):
    """x: [B, L] hoặc [B, L, C]. Với C > 1 (đa biến) trộn kênh: lớp tuyến tính nhận L*C giá trị."""

    def __init__(self, seq_len: int, pred_len: int = 5, n_channels: int = 1):
        super().__init__()
        self.seq_len, self.pred_len, self.n_channels = seq_len, pred_len, n_channels
        self.linear = nn.Linear(seq_len * n_channels, pred_len)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))


class NLinear(nn.Module):
    """Trừ giá cuối của TỪNG kênh rồi dự báo; cộng lại giá cuối của kênh 0 (kênh mục tiêu)."""

    def __init__(self, seq_len: int, pred_len: int = 5, zero_init: bool = False, n_channels: int = 1):
        super().__init__()
        self.seq_len, self.pred_len, self.n_channels = seq_len, pred_len, n_channels
        self.linear = nn.Linear(seq_len * n_channels, pred_len)
        if zero_init:
            _zero_(self.linear)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        last = x[:, -1:, :]
        return self.linear((x - last).reshape(x.shape[0], -1)) + last[:, 0, 0:1]


class DLinear(nn.Module):
    """Tách xu hướng/phần dư của từng kênh bằng trung bình trượt, mỗi phần một lớp tuyến tính."""

    def __init__(self, seq_len: int, pred_len: int = 5, moving_avg: int = 5, n_channels: int = 1):
        super().__init__()
        self.seq_len, self.pred_len, self.n_channels = seq_len, pred_len, n_channels
        self.ma = MovingAverage(moving_avg)
        self.linear_trend = nn.Linear(seq_len * n_channels, pred_len)
        self.linear_seasonal = nn.Linear(seq_len * n_channels, pred_len)

    def decompose(self, x):
        two_d = x.dim() == 2
        if two_d:
            x = x.unsqueeze(-1)
        trend = self.ma(x)
        seasonal = x - trend
        return (trend.squeeze(-1), seasonal.squeeze(-1)) if two_d else (trend, seasonal)

    def forward(self, x):
        trend, seasonal = self.decompose(x)
        b = x.shape[0]
        return self.linear_trend(trend.reshape(b, -1)) + self.linear_seasonal(seasonal.reshape(b, -1))


class Hybrid(nn.Module):
    def __init__(self, seq_len: int, pred_len: int = 5, n_channels: int = 1, moving_avg: int = 5,
                 level_branch: bool = False, level_clip: float = 10.0, zero_init: bool = False):
        super().__init__()
        self.seq_len, self.pred_len, self.n_channels = seq_len, pred_len, n_channels
        self.ma = MovingAverage(moving_avg)
        self.linear_trend = nn.Linear(seq_len * n_channels, pred_len)
        self.linear_seasonal = nn.Linear(seq_len * n_channels, pred_len)
        self.level_branch = bool(level_branch and n_channels > 1)
        self.level_clip = level_clip
        if self.level_branch:
            self.linear_level = nn.Linear(n_channels - 1, pred_len)
        if zero_init:
            _zero_(self.linear_trend, self.linear_seasonal, *([self.linear_level] if self.level_branch else []))

    def decompose(self, x):
        """x: [B, L, C] -> (trend, seasonal) của x đã trừ giá cuối từng kênh."""
        xc = x - x[:, -1:, :]
        trend = self.ma(xc)
        return trend, xc - trend

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.shape[-1] != self.n_channels:
            raise ValueError(f"Cần {self.n_channels} kênh, nhận {x.shape[-1]}.")
        trend, seasonal = self.decompose(x)
        b = x.shape[0]
        out = self.linear_trend(trend.reshape(b, -1)) + self.linear_seasonal(seasonal.reshape(b, -1))
        if self.level_branch:
            level = torch.clamp(x[:, -1, 1:], -self.level_clip, self.level_clip)
            out = out + self.linear_level(level)
        return out + x[:, -1, 0:1]


def build_model(name: str, seq_len: int, pred_len: int = 5, n_channels: int = 1,
                moving_avg: int = 5, level_branch: bool = False, zero_init: bool = False) -> nn.Module:
    name = name.lower()
    if zero_init and name not in ("nlinear", "hybrid"):
        raise ValueError("zero_init chỉ áp dụng cho nlinear và hybrid (hai mô hình có bước trừ giá cuối).")
    if name == "linear":
        return Linear(seq_len, pred_len, n_channels)
    if name == "nlinear":
        return NLinear(seq_len, pred_len, zero_init=zero_init, n_channels=n_channels)
    if name == "dlinear":
        return DLinear(seq_len, pred_len, moving_avg, n_channels)
    if name == "hybrid":
        return Hybrid(seq_len, pred_len, n_channels, moving_avg, level_branch=level_branch,
                      zero_init=zero_init)
    raise ValueError(f"Mô hình không hợp lệ: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
