import pytest
import torch
from engine import run
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