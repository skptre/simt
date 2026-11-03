import torch

from simt import RollingMean


def test_rolling_mean_matches_direct_average():
    g = torch.Generator().manual_seed(0)
    W, N, T = 7, 5, 40
    xs = torch.randn(T, N, generator=g, dtype=torch.float64)
    rm = RollingMean(W, N, "cpu", torch.float64)
    for t in range(T):
        rm.push(xs[t])
        expected = xs[max(0, t - W + 1):t + 1].mean(0)        # average of last W (or fewer) values
        assert torch.allclose(rm.mean(), expected)
    assert rm.full()


def test_peek_mean_equals_mean_after_push():
    g = torch.Generator().manual_seed(1)
    rm = RollingMean(4, 3, "cpu", torch.float64)
    for t in range(10):
        x = torch.randn(3, generator=g, dtype=torch.float64)
        peeked = rm.peek_mean(x)                              # predict without pushing...
        rm.push(x)
        assert torch.allclose(peeked, rm.mean())              # ...must equal the real result after pushing