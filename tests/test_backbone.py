import csv
import math

import torch

from examples.particle import Particle
from simt import (RunningStats, bootstrap_ci, grid, load_results, percentile, run, save_results,
                  summarize, sweep, one_at_a_time, paired_bootstrap_ci)


def test_running_stats_known_values():
    s = RunningStats(2, "cpu", torch.float64)
    s.add(torch.tensor([1.0, 3.0], dtype=torch.float64))
    s.add(torch.tensor([3.0, -5.0], dtype=torch.float64))
    out = s.summary("err")
    assert torch.allclose(out["err_bias"], torch.tensor([2.0, -1.0], dtype=torch.float64))
    assert torch.allclose(out["err_std"], torch.tensor([1.0, 4.0], dtype=torch.float64))
    assert torch.allclose(out["err_mae"], torch.tensor([2.0, 4.0], dtype=torch.float64))
    assert torch.allclose(out["err_rms"], torch.tensor([math.sqrt(5), math.sqrt(17)], dtype=torch.float64))


def test_running_stats_mask_and_empty():
    s = RunningStats(3, "cpu", torch.float64)
    x = torch.tensor([1.0, 2.0, float("nan")], dtype=torch.float64)
    s.add(x, mask=torch.tensor([True, False, False]))
    s.add(torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64), mask=torch.tensor([True, True, False]))
    out = s.summary("e")
    assert out["e_bias"][0].item() == 2.0
    assert out["e_bias"][1].item() == 4.0
    assert math.isnan(out["e_bias"][2].item())


def test_percentile_matches_torch_quantile():
    x = torch.randn(1001, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    for q in (0, 10, 50, 90, 100):
        assert math.isclose(percentile(x, q).item(), torch.quantile(x, q / 100).item(), rel_tol=1e-12)


def test_summarize_ignores_nan():
    out = summarize(torch.tensor([1.0, 2.0, 3.0, float("nan")]))
    assert out["mean"] == 2.0 and out["p50"] == 2.0


def test_bootstrap_ci_contains_true_median():
    x = torch.randn(20_000, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    lo, hi = bootstrap_ci(x, q=50, n_boot=500)
    assert lo < 0.0 < hi
    assert hi - lo < 0.06


def test_results_names_and_default_n_steps():
    sc = Particle(dt=0.5)
    sc.n_steps = 10
    results, _ = run(sc, n_runs=50, dtype=torch.float64)
    assert results.param_names == ["v", "x0"]
    assert results.metric_names == ["final_x"]
    expected = results["x0"] + results["v"] * 0.5 * 10
    assert torch.allclose(results["final_x"], expected)


def test_save_load_roundtrip(tmp_path):
    results, _ = run(Particle(), n_runs=20, n_steps=3, dtype=torch.float64)
    path = tmp_path / "r.npz"
    save_results(results, str(path), meta={"v_mean": 1.0})
    loaded, meta = load_results(str(path))
    assert meta == {"v_mean": 1.0}
    assert loaded.param_names == results.param_names
    assert loaded.metric_names == results.metric_names
    for k in results:
        assert torch.equal(loaded[k], results[k])


def test_grid_is_cartesian_product():
    g = grid(a=[1, 2, 3], b=["x", "y"])
    assert len(g) == 6
    assert {"a": 3, "b": "y"} in g


def test_sweep_summaries_and_outputs(tmp_path):
    csv_path = tmp_path / "s.csv"
    rows = sweep(lambda v_mean: Particle(dt=1.0, v_mean=v_mean, v_std=0.01, x0_std=0.01),
                 grid(v_mean=[1.0, 2.0]), n_runs=2000, n_steps=10, dtype=torch.float64,
                 csv_path=str(csv_path), save_dir=str(tmp_path / "runs"))
    assert len(rows) == 2
    for row in rows:
        assert abs(row["final_x_p50"] - row["v_mean"] * 10) < 0.05
    with open(csv_path) as f:
        assert len(list(csv.DictReader(f))) == 2
    loaded, meta = load_results(str(tmp_path / "runs" / "config_001.npz"))
    assert meta == {"v_mean": 2.0}
    assert loaded["final_x"].shape == (2000,)

def test_sweep_resume_skips_finished_configs(tmp_path):
    csv_path = str(tmp_path / "s.csv")
    calls = []

    def make(v_mean):
        calls.append(v_mean) # record which configs actually ran
        return Particle(dt=1.0, v_mean=v_mean, v_std=0.01, x0_std=0.01)

    sweep(make, grid(v_mean=[1.0, 2.0]), n_runs=100, n_steps=5, csv_path=csv_path)
    calls.clear()
    rows = sweep(make, grid(v_mean=[1.0, 2.0, 3.0]), n_runs=100, n_steps=5,
                 csv_path=csv_path, resume=True)
    assert calls == [3.0] # only the new config ran
    assert len(rows) == 3 # old 2 + new 1 in the CSV
    calls.clear()
    sweep(make, grid(v_mean=[1.0]), n_runs=200, n_steps=5, csv_path=csv_path, resume=True)
    assert calls == [1.0] # different n_runs -> not "done"

def test_one_at_a_time_varies_one_setting():
    configs = one_at_a_time({"a": 1, "b": 2}, a=[0, 1, 2], b=[2, 3])
    assert configs == [{"a": 1, "b": 2}, {"a": 0, "b": 2}, {"a": 2, "b": 2}, {"a": 1, "b": 3}]


def test_paired_ci_is_tight_when_runs_are_paired():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(20_000, generator=g, dtype=torch.float64)           
    b = a + 0.5 + 0.01 * torch.randn(20_000, generator=g, dtype=torch.float64)   
    est, lo, hi = paired_bootstrap_ci(a, b, n_boot=500)
    assert lo < 0.5 < hi and abs(est - 0.5) < 0.01
    assert hi - lo < 0.01                                                
    lo_a, hi_a = bootstrap_ci(a, n_boot=500)
    assert (hi - lo) < (hi_a - lo_a) / 5  

def test_recording_first_runs_every_kth_step(tmp_path):
    results, _ = run(Particle(dt=0.5), n_runs=50, n_steps=10, dtype=torch.float64,
                     record_runs=3, record_every=2)
    s = results.series
    assert s["step"].tolist() == [0, 2, 4, 6, 8]
    assert s["x"].shape == (5, 3)                                       # 5 recorded steps x 3 runs
    for row, k in enumerate(s["step"].tolist()):
        expected = results["x0"][:3] + results["v"][:3] * 0.5 * (k + 1)  # position after step k
        assert torch.allclose(s["x"][row], expected)
    save_results(results, str(tmp_path / "r.npz"))
    loaded, _ = load_results(str(tmp_path / "r.npz"))
    assert torch.equal(loaded.series["x"], s["x"])                                        
