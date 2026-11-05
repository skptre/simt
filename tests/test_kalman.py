import pytest
import torch
from simt import run, kf_update, kf_update_scalar
from examples.cv import ConstantVelocity

@pytest.mark.parametrize("dims", [1, 2, 3])

def test_filter_is_consistent(dims):
    sc = ConstantVelocity(dims=dims)
    results, _ = run(sc, n_runs=20_000, n_steps=50, dtype=torch.float64)
    mean_nees = results["nees"].mean().item()
    assert abs(mean_nees - sc.D) / sc.D < 0.05

def test_filter_beats_raw_sensor():
    sc = ConstantVelocity(dims=2)
    results, _ = run(sc, n_runs=20_000, n_steps=50, dtype=torch.float64)
    raw_sensor_err = (results["r"] * (2**0.5)).mean()

    assert results["pos_err"].mean() < raw_sensor_err

def _setup(n=1000, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    A = torch.randn(n, d, d, generator=g, dtype=torch.float64)
    P = A @ A.mT + 0.1 * torch.eye(d, dtype=torch.float64)          # random valid covariances
    H = torch.randn(d, generator=g, dtype=torch.float64)
    z = torch.randn(n, generator=g, dtype=torch.float64)
    return g, x, P, H, z


def test_scalar_matches_matrix_update():
    _, x, P, H, z = _setup()
    r2 = 0.3
    xs, Ps = kf_update_scalar(x, P, H, z, x @ H, r2)
    R = torch.full((x.shape[0], 1, 1), r2, dtype=torch.float64)
    xm, Pm = kf_update(x, P, z[:, None], H[None, :], R)             # same update, matrix form
    assert torch.allclose(xs, xm) and torch.allclose(Ps, Pm)


def test_joseph_equals_standard():
    _, x, P, H, z = _setup()
    _, P1 = kf_update_scalar(x, P, H, z, x @ H, 0.3)
    _, P2 = kf_update_scalar(x, P, H, z, x @ H, 0.3, joseph=True)
    assert torch.allclose(P1, P2)


def test_mask_nan_and_gate_skip_runs():
    _, x, P, H, z = _setup(n=4)
    z = z.clone()
    z[1] = float("nan")                                              # gap written as NaN
    z[2] = 1e6                                                       # huge outlier
    mask = torch.tensor([True, True, True, False])                   # run 3 skipped by mask
    x2, P2, info = kf_update_scalar(x, P, H, z, x @ H, 0.3, mask=mask, gate=9.0, return_info=True)
    assert info["accepted"].tolist() == [True, False, False, False]
    assert not torch.equal(x2[0], x[0])                              # run 0 updated
    for i in (1, 2, 3):                                              # others untouched, no NaN
        assert torch.equal(x2[i], x[i]) and torch.equal(P2[i], P[i])


def test_nis_averages_one_when_model_is_right():
    g, x, P, H, _ = _setup(n=50_000)
    L = torch.linalg.cholesky(P)
    x_true = x + (L @ torch.randn(x.shape[0], x.shape[1], 1, generator=g, dtype=torch.float64)).squeeze(-1)
    r2 = 0.3
    z = x_true @ H + r2 ** 0.5 * torch.randn(x.shape[0], generator=g, dtype=torch.float64)
    _, _, info = kf_update_scalar(x, P, H, z, x @ H, r2, return_info=True)
    assert abs(info["nis"].mean().item() - 1.0) < 0.03