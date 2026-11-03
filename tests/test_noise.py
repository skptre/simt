import torch
import math
from simt import ar1

def make(seed=0, device="cpu", dtype=torch.float64):
    g = torch.Generator(device=device).manual_seed(seed)
    return ar1(n_runs=20000, n_steps=200, sigma=2.0, tau_s=600.0, dt=30.0, generator=g, device=device, dtype=dtype)

def test_shape():
    x = make()
    assert x.shape == (200, 20000)

def test_reproducibility():
    assert torch.equal(make(seed=1), make(seed=1))

def test_spread_is_sigma():
    x = make()
    std_per_step = x.std(dim=1)
    assert torch.allclose(std_per_step, torch.full_like(std_per_step, 2.0), atol=0.05)

def test_memory_after_tau():
    x = make()
    lag = 20
    corr = (x[:-lag] * x[lag:]).mean() / x.var()
    assert abs(corr.item() - math.exp(-1)) < 0.02

def test_ar1_step_matches_spread_and_memory():
    from simt.noise import ar1_step
    g = torch.Generator().manual_seed(0)
    sigma, tau, dt = 2.0, 600.0, 30.0
    x = sigma * torch.randn(50_000, generator=g, dtype=torch.float64)   # start at the right spread
    start = x.clone()
    for _ in range(20):                                                 # 20 steps = 600 s = one tau
        x = ar1_step(x, sigma, tau, dt, g)
    assert abs(x.std().item() - sigma) / sigma < 0.02                   # spread stays sigma
    corr = (start * x).mean().item() / sigma ** 2
    assert abs(corr - math.exp(-1)) < 0.02                              # memory: 37% after one tau