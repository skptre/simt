import torch
import math

def ar1(n_runs, n_steps, sigma, tau_s, dt, generator, device, dtype):
    # n_runs: N, number of independent runs
    # n_steps: T, number of timesteps
    # sigma: wiggle size
    # tau_s: how long wiggle lasts, seconds
    # dt: timestep, seconds
    # generator: torch's random number generator
    # device: device to run on (CPU or CUDA)
    # dtype: torch.float32 or torch.float64 for precision
    a = math.exp(-dt/tau_s)
    z = torch.randn((n_steps, n_runs), generator=generator, device=device, dtype=dtype)
    kick_scale = sigma * math.sqrt(1 - a**2)
    kicks = kick_scale * z
    x = torch.empty_like(z)
    x[0] = sigma * z[0]
    for k in range(1, n_steps):
        x[k] = a * x[k-1] + kicks[k]
    return x

def ar1_step(x, sigma, tau_s, dt, generator):
    # x: current noise value for every run, any shape (e.g. [N] or [N, 7])
    # returns the next value: a * x + kick, same AR(1) rule as ar1()

    a = math.exp(-dt/tau_s)
    kick = sigma * math.sqrt(1 - a * a)
    z = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return a * x + kick * z