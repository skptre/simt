import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

@cuda_only
def test_particle_kernel_matches_engine():
    from kernels import run_particle_kernel
    g = torch.Generator(device="cuda").manual_seed(0)
    n, dt, n_steps = 100_000, 0.5, 1000
    x0 = torch.randn(n, generator=g, device="cuda", dtype=torch.float64)
    v = 2.0 + 0.3 * torch.randn(n, generator=g, device="cuda", dtype=torch.float64)
    out = run_particle_kernel(x0, v, dt, n_steps)
    assert torch.allclose(out, x0 + v * dt * n_steps)