import pytest
import torch
from cv import ConstantVelocity
from engine import run


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

@cuda_only
@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])

def test_kf_kernel_consistent(dims, dtype):
    from kernels import make_kf_kernel
    sc = ConstantVelocity(dims=dimns)
    out = run_kf_kernel(sc, n_runs=20_000, n_steps=50, dtype=dtype)
    assert abs(out["nees"].mean().item() - sc.D) / sc.D < 0.05

@cuda_only
def test_kf_kernel_matches_pytorch_statistics():
    from kernels import run_kf_kernel
    sc = ConstantVelocity(dims=2)
    ker = run_kf_kernel(sc, n_runs=50_000, n_steps=50, dtype=torch.float64)
    eng, _ = run(sc, n_runs=50_000, n_steps=50, device="cuda", dtype=torch.float64)
    k, e = ker["pos_err"].mean().item(), eng["pos_err"].mean().item()
    assert abs(k - e) / e < 0.03

@cuda_only
def test_particle_kernel_matches_engine():
    from kernels import run_particle_kernel
    g = torch.Generator(device="cuda").manual_seed(0)
    n, dt, n_steps = 100_000, 0.5, 1000
    x0 = torch.randn(n, generator=g, device="cuda", dtype=torch.float64)
    v = 2.0 + 0.3 * torch.randn(n, generator=g, device="cuda", dtype=torch.float64)
    out = run_particle_kernel(x0, v, dt, n_steps)
    assert torch.allclose(out, x0 + v * dt * n_steps)