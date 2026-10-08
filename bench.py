import time
import torch
from noise import ar1
from engine import run
from particle import Particle

def sync(device):
    if device == "cuda":
        torch.cuda.synchronize(device)

def time_ar1(n_runs, n_steps, device, dtype=torch.float32, repeats=5):
    g = torch.Generator(device=device).manual_seed(0)
    args = (n_runs, n_steps, 1.0, 600.0, 30.0, g, device, dtype)
    ar1(*args)
    sync(device)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        ar1(*args)
        sync(device)
        times.append(time.perf_counter() - start)
    
    times.sort()
    return times[len(times) // 2]

def time_engine(scenario, n_runs, n_steps, device, reapts=5, chunk_size=1_000_000):
    run(scenario, n_runs, n_steps, device=device, chunk_size=chunk_size)
    times = sorted(run(scenario, n_runs, n_steps, device=device, chunk_size=chunk_size)[1] for _ in range(repeats))
    return times[len(times) // 2]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("---AR1---")
    for n_runs in [1, 10, 100, 1_000, 10_000, 100_000]:
        t = time_ar1(n_runs, 1000, device)
        print(f"{device} N={n_runs:>8} {t*1000:8.1f} ms {n_runs/t:12.0f} runs/s")
    
    print("---ENGINE: Particle---")
    sc = Particle()
    for n_runs in [1, 100, 10_000, 1_000_000, 10_000_000]:
        t = time_engine(sc, n_runs, 1000, device)
        print(f"{device}  N={n_runs:>10}  {t*1000:8.1f} ms  {n_runs/t:14.0f} runs/s")

