import time
import torch

class Scenario:
    def sample(self, n_runs, generator, device, dtype):
        # Draw each run's random settings ("dispersions"): noise levels, gains, offsets.
        # Returns dict {name: tensor [N]} - one value per run.
        # Kept separately because later the SURROGATE learns: settings -> outcome.
        raise NotImplementedError

    def init_state(self, params):
        # Starting state for every run. Returns starting state
        # Each run's start can depend on its own params (broadcasting).
        raise NotImplementedError
    
    def step(self, state, k, params):
        # Advance ALL runs by one timestep k. Returns new state, same shape.
        # This is the sequential part - the engine calls it in a loop over time.
        raise NotImplementedError
    
    def score(self, state, params):
        # After the last step: per-run outcome metrics, e.g. final error, success flag.
        # Returns dict {name: tensor [N]}.
        raise NotImplementedError

def run(scenario, n_runs, n_steps, seed=0, device="cpu", dtype=torch.float32, chunk_size=100_000):
    generator = torch.Generator(device=device).manual_seed(seed)

    chunks = []
    start = time.perf_counter()

    for first in range(0, n_runs, chunk_size):
        n = min(chunk_size, n_runs - first)

        params = scenario.sample(n, generator, device, dtype)
        state = scenario.init_state(params)

        for k in range(n_steps):
            state = scenario.step(state, k, params)
        metrics = scenario.score(state, params)
        chunks.append({name: v.cpu() for name, v in {**params, **metrics}.items()})

    elapsed = time.perf_counter() - start

    results = {name: torch.cat([chunk[name] for chunk in chunks]) for name in chunks[0]}
    return results, elapsed