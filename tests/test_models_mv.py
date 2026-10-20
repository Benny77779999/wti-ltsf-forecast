import pytest

torch = pytest.importorskip("torch")

from src.models import DLinear, Linear, NLinear, build_model, count_parameters


def test_multichannel_shapes_and_parameter_counts():
    x = torch.randn(4, 30, 3)
    for name, extra in (("linear", 1), ("nlinear", 1), ("dlinear", 2)):
        m = build_model(name, 30, 5, n_channels=3)
        assert m(x).shape == (4, 5)
        assert count_parameters(m) == extra * (5 * 30 * 3 + 5)


def test_nlinear_mv_recenters_every_channel_and_shifts_with_target():
    m = NLinear(30, 5, n_channels=3)
    x = torch.randn(2, 30, 3)
    y = m(x)
    x2 = x.clone(); x2[:, :, 1:] += 100.0                    # đổi mức kênh phụ: không ảnh hưởng
    torch.testing.assert_close(m(x2), y, atol=1e-3, rtol=0)
    x3 = x.clone(); x3[:, :, 0] += 7.0                       # dịch kênh mục tiêu: dự báo dịch theo
    torch.testing.assert_close(m(x3), y + 7.0, atol=1e-4, rtol=0)


def test_single_channel_inputs_still_work_for_2d_and_3d():
    x = torch.randn(3, 20)
    for cls in (Linear, NLinear, DLinear):
        m = cls(20, 5)
        torch.testing.assert_close(m(x), m(x.unsqueeze(-1)))
