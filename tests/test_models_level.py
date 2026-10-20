import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models import Hybrid, build_model, count_parameters


def test_level_branch_parameter_count_and_noop_for_single_channel():
    base = count_parameters(Hybrid(30, 5, n_channels=4))
    lvl = count_parameters(Hybrid(30, 5, n_channels=4, level_branch=True))
    assert lvl - base == 3 * 5 + 5                                    # Linear(C-1 -> T)
    assert count_parameters(Hybrid(30, 5, n_channels=1, level_branch=True)) == \
        count_parameters(Hybrid(30, 5, n_channels=1))                 # 1 kênh: không có nhánh


def test_level_branch_manual_value_and_target_shift_invariance():
    m = Hybrid(5, 3, n_channels=3, moving_avg=3, level_branch=True)
    for lin in (m.linear_trend, m.linear_seasonal):
        lin.weight.data.zero_(); lin.bias.data.zero_()
    m.linear_level.weight.data = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    m.linear_level.bias.data = torch.tensor([0.5, 0.0, 0.0])
    x = torch.randn(2, 5, 3)
    expected = x[:, -1, 0:1] + torch.stack([x[:, -1, 1] + 0.5, 2 * x[:, -1, 2], x[:, -1, 1] + x[:, -1, 2]], dim=1)
    torch.testing.assert_close(m(x), expected, atol=1e-5, rtol=0)
    x2 = x.clone(); x2[:, :, 0] += 40.0                                # dịch kênh mục tiêu
    torch.testing.assert_close(m(x2), m(x) + 40.0, atol=1e-4, rtol=0)


def test_level_clip_bounds_extrapolation():
    m = build_model("hybrid", 10, 5, n_channels=2, level_branch=True)
    x = torch.zeros(1, 10, 2)
    x[:, :, 1] = 50.0                      # kênh phụ đứng yên ở mức 50 (diễn biến = 0, chỉ mức khác)
    y_big = m(x)
    x[:, :, 1] = 1e6
    torch.testing.assert_close(m(x), y_big, atol=1e-5, rtol=0)   # cả hai mức đều bị cắt về 10
    x[:, :, 1] = 3.0                       # trong ngưỡng cắt thì mức vẫn ảnh hưởng
    assert not torch.allclose(m(x), y_big, atol=1e-6)
