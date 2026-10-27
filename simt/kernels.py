import math

import numpy as np
import torch
from numba import cuda, float32, float64
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float32,
    xoroshiro128p_normal_float64,
)


@cuda.jit
def particle_kernel(x0, v, dt, n_steps, out):
    i = cuda.grid(1)  # this thread's run index
    if i < x0.shape[0]:
        x = x0[i]  # position, kept in a register
        vi = v[i]  # this run's speed
        for k in range(n_steps):
            x += vi * dt
        out[i] = x


def run_particle_kernel(x0, v, dt, n_steps, threads=256):
    np_t = np.float64 if x0.dtype == torch.float64 else np.float32  # scalar type matching the data
    out = torch.empty_like(x0)  # output: final position per run
    blocks = (x0.shape[0] + threads - 1) // threads  # ceil(N / threads)
    particle_kernel[blocks, threads](
        cuda.as_cuda_array(x0), cuda.as_cuda_array(v), np_t(dt), n_steps, cuda.as_cuda_array(out)
    )
    return out


@cuda.jit(device=True)
def chol_solve(S, B, X, m, ncols):
    for j in range(m):
        s = S[j, j]  # running sum for the current entry
        for k in range(j):
            s -= S[j, k] * S[j, k]
        S[j, j] = math.sqrt(s)
        for i in range(j + 1, m):
            s = S[i, j]
            for k in range(j):
                s -= S[i, k] * S[j, k]
            S[i, j] = s / S[j, j]
    for c in range(ncols):
        for i in range(m):
            s = B[i, c]
            for k in range(i):
                s -= S[i, k] * X[k, c]
            X[i, c] = s / S[i, i]
        for i in range(m - 1, -1, -1):
            s = X[i, c]
            for k in range(i + 1, m):
                s -= S[k, i] * X[k, c]
            X[i, c] = s / S[i, i]


_kf_cache = {}  # compiled kernels keyed by (dims, use64)
_state_cache = {}  # pristine RNG states keyed by (n_runs, seed)


def get_states(n_runs, seed):
    key = (n_runs, seed)
    if key not in _state_cache:
        _state_cache[key] = create_xoroshiro128p_states(n_runs, seed=seed)
    src = _state_cache[key]
    states = cuda.device_array_like(src)  # fresh GPU buffer, same shape/type
    states.copy_to_device(src)
    return states


def make_kf_kernel(dims, use64):
    key = (dims, use64)
    if key in _kf_cache:
        return _kf_cache[key]

    D = 2 * dims  # state size: position + velocity per axis
    M = dims  # measurement size: positions only
    ft = float64 if use64 else float32  # number type used everywhere in the kernel
    normal = xoroshiro128p_normal_float64 if use64 else xoroshiro128p_normal_float32  # per-thread random normal draw

    @cuda.jit
    def kf_kernel(r, F, Lq, Q, H, p0, n_steps, states, pos_err, nees):
        i = cuda.grid(1)  # this thread's run index
        if i >= r.shape[0]:
            return

        zero = ft(0.0)  # typed zero, avoids silent float64 promotion
        ri = r[i]  # this run's sensor noise std

        xt = cuda.local.array(D, ft)  # truth state
        xh = cuda.local.array(D, ft)  # filter estimate
        tv = cuda.local.array(D, ft)  # temporary vector
        w = cuda.local.array(D, ft)  # random draws
        P = cuda.local.array((D, D), ft)  # covariance
        T = cuda.local.array((D, D), ft)  # temporary matrix
        y = cuda.local.array(M, ft)  # innovation
        S = cuda.local.array((M, M), ft)  # innovation covariance
        B = cuda.local.array((M, D), ft)  # H P
        X = cuda.local.array((M, D), ft)  # K transposed
        E = cuda.local.array((D, 1), ft)  # final error, for NEES
        U = cuda.local.array((D, 1), ft)  # P^-1 E, for NEES

        sp = math.sqrt(p0)  # initial truth std
        for a in range(D):
            xt[a] = sp * normal(states, i)
            xh[a] = zero
            for b in range(D):
                P[a, b] = p0 if a == b else zero

        for k in range(n_steps):
            for a in range(D):
                w[a] = normal(states, i)
            for a in range(D):
                s = zero
                for b in range(D):
                    s += F[a, b] * xt[b]
                for b in range(a + 1):
                    s += Lq[a, b] * w[b]
                tv[a] = s
            for a in range(D):
                xt[a] = tv[a]

            for a in range(D):
                s = zero
                for b in range(D):
                    s += F[a, b] * xh[b]
                tv[a] = s
            for a in range(D):
                xh[a] = tv[a]

            for a in range(D):
                for b in range(D):
                    s = zero
                    for c in range(D):
                        s += F[a, c] * P[c, b]
                    T[a, b] = s
            for a in range(D):
                for b in range(D):
                    s = Q[a, b]
                    for c in range(D):
                        s += T[a, c] * F[b, c]
                    P[a, b] = s

            for m in range(M):
                s = zero
                for c in range(D):
                    s += H[m, c] * xt[c]
                z = s + ri * normal(states, i)  # noisy measurement
                s = zero
                for c in range(D):
                    s += H[m, c] * xh[c]
                y[m] = z - s

            for m in range(M):
                for b in range(D):
                    s = zero
                    for c in range(D):
                        s += H[m, c] * P[c, b]
                    B[m, b] = s
            for m in range(M):
                for n in range(M):
                    s = zero
                    for c in range(D):
                        s += B[m, c] * H[n, c]
                    if m == n:
                        s += ri * ri
                    S[m, n] = s

            chol_solve(S, B, X, M, D)

            for a in range(D):
                s = zero
                for m in range(M):
                    s += X[m, a] * y[m]
                xh[a] += s
            for a in range(D):
                for b in range(D):
                    s = zero
                    for m in range(M):
                        s += X[m, a] * B[m, b]
                    P[a, b] -= s

        for a in range(D):
            E[a, 0] = xt[a] - xh[a]
            for b in range(D):
                T[a, b] = P[a, b]
        chol_solve(T, E, U, D, 1)
        s = zero
        for a in range(D):
            s += E[a, 0] * U[a, 0]
        nees[i] = s
        s = zero
        for a in range(M):
            s += E[a, 0] * E[a, 0]
        pos_err[i] = math.sqrt(s)

    _kf_cache[key] = kf_kernel
    return kf_kernel


def run_kf_kernel(sc, n_runs, n_steps=100, seed=0, dtype=torch.float32, threads=128):
    use64 = dtype == torch.float64  # True = double precision kernel
    np_t = np.float64 if use64 else np.float32  # matching numpy scalar type for kernel arguments

    sc._build("cuda", dtype)
    g = torch.Generator(device="cuda").manual_seed(seed)  # seeded generator for per-run params
    u = torch.rand(n_runs, generator=g, device="cuda", dtype=dtype)  # uniform [0, 1) per run
    r = sc.r_min + (sc.r_max - sc.r_min) * u  # per-run sensor noise std

    states = get_states(n_runs, seed)  # one random stream per thread (setup cached, fresh copy per call)
    pos_err = torch.empty(n_runs, device="cuda", dtype=dtype)  # output: final position error per run
    nees = torch.empty_like(pos_err)  # output: final NEES per run

    kernel = make_kf_kernel(sc.dims, use64)  # compiled kernel for this D and precision
    blocks = (n_runs + threads - 1) // threads  # ceil(n_runs / threads)
    mats = [cuda.as_cuda_array(m.contiguous()) for m in (sc.F, sc.Lq, sc.Q, sc.H)]  # model matrices, zero-copy views
    kernel[blocks, threads](
        cuda.as_cuda_array(r), *mats, np_t(sc.p0), n_steps, states,
        cuda.as_cuda_array(pos_err), cuda.as_cuda_array(nees),
    )
    return {"r": r.cpu(), "pos_err": pos_err.cpu(), "nees": nees.cpu()}
