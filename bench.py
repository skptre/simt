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

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_steps = 1000
    for n_runs in [1, 10, 100, 1000, 10000, 100000]:
        t = time_ar1(n_runs, n_steps, device)

        print(f"{device} N={n_runs:>8} {t*1000:8.1f} ms {n_runs/t:12.0f} runs/s")