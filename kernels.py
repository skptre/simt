import torch
from numba import cuda

@cuda.jit

def particle_kernel(x0, v, dt, n_steps, out):
    i = cuda.grid(1)
    if i < x0.shape[0]:
        x = x0[i]
        vi = v[i]

        for k in range(n_steps):
            x += vi * dt
        out[i] = x

def run_particle_kernel(x0, v, dt, n_steps, threads=256):
    out = torch.empty_like(x0)
    blocks = (x0.shape[0] + threads - 1) // threads
    particle_kernel[blocks, threads](cuda.as_cuda_array(x0), cuda.as_cuda_array(v), dt, n_steps, cuda.as_cuda_array(out))
    return out