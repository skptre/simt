import time
import torch

class Scenario:
    def sample(self, n_runs, generator, device, dtype):
        # Draw each run's random settings ("dispersions"): noise levels, gains, offsets.
        # Returns dict {name: tensor [N]} - one value per run.
        # Kept separately because later the SURROGATE learns: settings -> outcome.
        raise NotImplementedError

    def init_state(self, params, generator):
        # Starting state for every run. Returns starting state
        # Each run's start can depend on its own params (broadcasting).
        raise NotImplementedError
    
    def step(self, state, k, params, generator):
        # Advance ALL runs by one timestep k. Returns new state, same shape.
        # This is the sequential part - the engine calls it in a loop over time.
        raise NotImplementedError
    
    def score(self, state, params):
        # After the last step: per-run outcome metrics, e.g. final error, success flag.
        # Returns dict {name: tensor [N]}.
        raise NotImplementedError


class Results(dict):
    def __init__(self, data, param_names, metric_names):
        super().__init__(data)
        self.param_names = list(param_names)
        self.metric_names = list(metric_names)

    def params(self):
        return {k: self[k] for k in self.param_names}

    def metrics(self):
        return {k: self[k] for k in self.metric_names}


def run(scenario, n_runs, n_steps=None, seed=0, device="cpu", dtype=torch.float32, chunk_size=100_000,
        progress=False):
    if n_steps is None:
        n_steps = getattr(scenario, "n_steps", None)
        if n_steps is None:
            raise ValueError("pass n_steps or give the scenario an n_steps attribute")

    generator = torch.Generator(device=device).manual_seed(seed)
    n_chunks = (n_runs + chunk_size - 1) // chunk_size
    report_every = max(1, n_steps // 10)

    chunks = []
    param_names, metric_names = [], []
    start = time.perf_counter()

    for c, first in enumerate(range(0, n_runs, chunk_size)):
        n = min(chunk_size, n_runs - first)

        params = scenario.sample(n, generator, device, dtype)
        state = scenario.init_state(params, generator)

        for k in range(n_steps):
            state = scenario.step(state, k, params, generator)
            if progress and (k + 1) % report_every == 0:
                done = (c * n_steps + k + 1) / (n_chunks * n_steps)
                spent = time.perf_counter() - start
                print(f"  chunk {c + 1}/{n_chunks}  step {k + 1}/{n_steps}  "
                      f"{spent:6.1f} s elapsed  ~{spent / done - spent:6.1f} s left", flush=True)
        metrics = scenario.score(state, params)
        param_names, metric_names = list(params), list(metrics)
        chunks.append({name: v.cpu() for name, v in {**params, **metrics}.items()})

    elapsed = time.perf_counter() - start

    data = {name: torch.cat([chunk[name] for chunk in chunks]) for name in chunks[0]}
    return Results(data, param_names, metric_names), elapsed