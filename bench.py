import time
import torch
from noise import ar1
from engine import run
from particle import Particle
import csv 
import sys
from cv import ConstantVelocity


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

def time_engine(scenario, n_runs, n_steps, device, repeats=5, chunk_size=1_000_000):
    run(scenario, n_runs, n_steps, device=device, chunk_size=chunk_size)
    times = sorted(run(scenario, n_runs, n_steps, device=device, chunk_size=chunk_size)[1] for _ in range(repeats))
    return times[len(times) // 2]

def bench_kf(device, n_steps=100):
    max_n = 1_000_000 if device == "cuda" else 100_000
    rows = []

    for dims in [1, 2, 3]:
        sc = ConstantVelocity(dims=dims)
        for n_runs in [1_000, 10_000, 100_000, 1_000_000]:
            if n_runs > max_n:
                continue
            t = time_engine(sc, n_runs, n_steps, device)
            rows.append({"device": device, "D": sc.D, "n_runs": n_runs, "n_steps": n_steps, "seconds": t, "run_per_s": n_runs / t})
            print(f"{device}  D={sc.D}  N={n_runs:>9}  {t*1000:9.1f} ms  {n_runs/t:14.0f} runs/s")
    return rows

def time_fn(fn, device, repeats=5):
    fn()
    sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]

def particle_kernel_full(n_runs, n_steps, dt):
    from kernels import run_particle_kernel
    g = torch.Generator(device="cuda").manual_seed(0)
    x0 = 0.5 * torch.randn(n_runs, generator=g, device="cuda")
    v = 1.0 + 0.1 * torch.randn(n_runs, generator=g, device="cuda")
    out = run_particle_kernel(x0, v, dt, n_steps)
    return {"v": v.cpu(), "x0": x0.cpu(), "final_x": out.cpu()}

def bench_kf_kernel(n_steps=100):
    from kernels import run_kf_kernel
    rows = []

    for dims in [1, 2, 3]:
        sc = ConstantVelocity(dims=dims)
        for n_runs in [1_000, 10_000, 100_000, 1_000_000]:
            t_eng = time_engine(sc, n_runs, n_steps, "cuda")
            t_ker = time_fn(lambda: run_kf_kernel(sc, n_runs, n_steps), "cuda")
            rows.append({"D": sc.D, "n_runs": n_runs, "n_steps": n_steps, "engine_s": t_eng, "kernel_s": t_ker, "speedup": t_eng / t_ker})
            print(f"D={sc.D}  N={n_runs:>9}  engine={t_eng*1000:9.1f} ms  kernel={t_ker*1000:8.1f} ms  speedup={t_eng/t_ker:6.1f}x")
    return rows

def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("ar1", "all"):
        print("---AR1---")
        for n_runs in [1, 10, 100, 1_000, 10_000, 100_000]:
            t = time_ar1(n_runs, 1000, device)
            print(f"{device} N={n_runs:>10} {t*1000:8.1f} ms {n_runs/t:14.0f} runs/s")
    
    if which in ("particle", "all"):
        print("---ENGINE: Particle---")
        sc = Particle()
        for n_runs in [1, 100, 10_000, 1_000_000, 10_000_000]:
            t = time_engine(sc, n_runs, 1000, device)
            print(f"{device}  N={n_runs:>10}  {t*1000:8.1f} ms  {n_runs/t:14.0f} runs/s")

    if which in ("kf", "all"):
        print("---Kalman filter (constant velocity)---")
        rows = bench_kf(device)
        save_csv(rows, f"bench_kf_{device}.csv")
    
    if which in ("pkernel", "all") and device == "cuda":
        print("---Particle: PyTorch engine vs fused kernel---")
        sc = Particle()
        for n_runs in [1, 100, 10_000, 1_000_000, 10_000_000]:
            t_eng = time_engine(sc, n_runs, 1000, device)
            t_ker = time_fn(lambda: particle_kernel_full(n_runs, 1000, sc.dt), device)
            print(f"N={n_runs:>10}  engine={t_eng*1000:8.1f} ms  kernel={t_ker*1000:8.1f} ms  speedup={t_eng/t_ker:6.1f}x")

    if which in ("kfkernel", "all") and device == "cuda":
        print("---Kalman filter: PyTorch engine vs fused kernel---")
        rows = bench_kf_kernel()
        save_csv(rows, "bench_kf_kernel.csv")

