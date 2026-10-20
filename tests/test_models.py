"""Kiểm tra mô hình, gồm đối chiếu với kết quả tính tay của tài liệu (Mục 8)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models import DLinear, Hybrid, Linear, MovingAverage, NLinear, build_model, count_parameters

DATA = torch.tensor([[100.0, 102.0, 98.0, 105.0, 103.0]])


def test_linear_matches_document():
    m = Linear(5, 3)
    m.linear.weight.data = torch.tensor([[0.2, 0.3, -0.1, 0.4, 0.2], [0.1, 0.2, 0.3, 0.3, 0.1],
                                         [0.0, 0.1, 0.2, 0.4, 0.3]])
    m.linear.bias.data = torch.tensor([1.0, 2.0, 3.0])
    np.testing.assert_allclose(m(DATA).detach().numpy()[0], [104.4, 103.6, 105.7], atol=1e-4)


def test_nlinear_matches_document():
    m = NLinear(5, 3)
    m.linear.weight.data = torch.tensor([[0.1, 0.2, 0.1, 0.3, 0.3], [0.2, 0.1, 0.3, 0.2, 0.2],
                                         [0.3, 0.2, 0.2, 0.2, 0.1]])
    m.linear.bias.data = torch.tensor([0.5, 1.0, 2.0])
    np.testing.assert_allclose(m(DATA).detach().numpy()[0], [103.1, 102.2, 103.3], atol=1e-4)


def test_dlinear_matches_document():
    m = DLinear(5, 3, moving_avg=3)
    m.linear_trend.weight.data = torch.tensor([[0.3, 0.3, 0.2, 0.1, 0.1], [0.2, 0.3, 0.3, 0.1, 0.1],
                                               [0.1, 0.2, 0.3, 0.3, 0.1]])
    m.linear_trend.bias.data = torch.tensor([0.5, 1.0, 1.5])
    m.linear_seasonal.weight.data = torch.tensor([[0.1, -0.2, 0.3, -0.1, 0.0], [0.0, 0.1, -0.2, 0.2, 0.1],
                                                  [-0.1, 0.0, 0.1, 0.1, 0.2]])
    m.linear_seasonal.bias.data = torch.tensor([0.2, 0.3, 0.1])
    trend, _ = m.decompose(DATA)
    np.testing.assert_allclose(trend.detach().numpy()[0], [100.667, 100.0, 101.667, 102.0, 103.667], atol=1e-3)
    np.testing.assert_allclose(m(DATA).detach().numpy()[0], [99.933, 103.967, 103.0], atol=1e-3)


def test_moving_average_is_shift_equivariant():
    ma, x = MovingAverage(5), torch.randn(3, 30, 2)
    torch.testing.assert_close(ma(x + 7.0), ma(x) + 7.0)


def test_hybrid_invariant_to_additive_level_shift():
    torch.manual_seed(0)
    m, x = Hybrid(30, 5, n_channels=1), torch.randn(4, 30)
    torch.testing.assert_close(m(x + 50.0), m(x) + 50.0, atol=1e-3, rtol=0)
    d = DLinear(30, 5)                        # DLinear thì KHÔNG bất biến với dịch mức
    assert not torch.allclose(d(x + 50.0), d(x) + 50.0, atol=1e-3)


def test_decompose_then_recenter_equals_recenter_then_decompose():
    torch.manual_seed(1)
    m, x = Hybrid(20, 5, n_channels=3), torch.randn(2, 20, 3)
    trend, seasonal = m.decompose(x)                     # trừ mốc rồi phân rã
    t2 = m.ma(x)                                         # phân rã rồi trừ mốc
    torch.testing.assert_close(trend, t2 - x[:, -1:, :], atol=1e-5, rtol=0)
    torch.testing.assert_close(seasonal, x - t2, atol=1e-5, rtol=0)


def test_hybrid_univariate_matches_manual_computation():
    m = Hybrid(5, 3, n_channels=1, moving_avg=3)
    for lin in (m.linear_trend, m.linear_seasonal):
        lin.weight.data.zero_(); lin.bias.data.zero_()
    out = m(DATA)
    np.testing.assert_allclose(out.detach().numpy()[0], [103.0, 103.0, 103.0], atol=1e-5)  # = giá cuối


def test_hybrid_multichannel_shapes_and_parameter_counts():
    x = torch.randn(6, 120, 4)
    m = build_model("hybrid", 120, 5, n_channels=4)
    assert m(x).shape == (6, 5)
    assert count_parameters(m) == 2 * (5 * 120 * 4 + 5)
    assert count_parameters(build_model("linear", 120, 5)) == 5 * 120 + 5
    assert count_parameters(build_model("dlinear", 120, 5)) == 2 * (5 * 120 + 5)
    with pytest.raises(ValueError):
        m(torch.randn(2, 120, 3))
    with pytest.raises(ValueError):
        MovingAverage(4)


def test_all_models_can_overfit_tiny_batch():
    torch.manual_seed(0)
    x = torch.randn(16, 30).cumsum(dim=1)
    y = x[:, -1:] + 0.1 * torch.randn(16, 5)
    for name in ("linear", "nlinear", "dlinear", "hybrid"):
        m = build_model(name, 30, 5)
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        first = None
        for _ in range(300):
            opt.zero_grad(); loss = ((m(x) - y) ** 2).mean(); loss.backward(); opt.step()
            first = first if first is not None else loss.item()
        assert loss.item() < first
