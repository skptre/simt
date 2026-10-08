import torch
from engine import run
from particle import Particle

SC = Particle(dt=0.5, v_mean=2.0, v_std=0.3, x0_std=1.0)
N_STEPS = 100
TOTAL_TIME = N_STEPS * SC.dt


def test_every_run_matches_exact_answer():
    results, _ = run(SC, n_runs=10_000, n_steps=N_STEPS, dtype=torch.float64)
    expected = results["x0"] + results["v"] * TOTAL_TIME
    assert torch.allclose(results["final_x"], expected)


def test_statistics_match_theory():
    results, _ = run(SC, n_runs=200_000, n_steps=N_STEPS, dtype=torch.float64)
    final = results["final_x"]
    expected_mean = SC.v_mean * TOTAL_TIME
    expected_std = (SC.x0_std**2 + (SC.v_std * TOTAL_TIME)**2) ** 0.5
    assert abs(final.mean().item() - expected_mean) < 0.2
    assert abs(final.std().item() - expected_std) / expected_std < 0.01


def test_chunking_keeps_every_run():
    results, _ = run(SC, n_runs=250, n_steps=5, chunk_size=100)
    assert results["final_x"].shape == (250,)
