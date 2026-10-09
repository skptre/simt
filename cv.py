import torch
from engine import Scenario
from kalman import kf_predict, kf_update

class ConstantVelocity(Scenario):
    
    def __init__(self, dims=2, dt=1.0, q=0.01, r_min=0.5, r_max=2.0, p0=10.0):
        self.dims, self.dt, self.q = dims, dt, q
        self.r_min, self.r_max, self.p0 = r_min, r_max, p0
        self.D = 2*dims
        self.M = dims
    
    def _build(self, device, dtype):
        d, dt, q = self.dims, self.dt, self.q
        I = torch.eye(d, device=device, dtype=dtype)
        Z = torch.zeros(d, d, device=device, dtype=dtype)

        self.F = torch.cat([torch.cat([I, dt * I], 1), torch.cat([Z, I], 1)], 0)
        self.Q = q * torch.cat([torch.cat([dt**3 / 3 * I, dt**2 / 2 * I], 1),
                                torch.cat([dt**2 / 2 * I, dt * I], 1)], 0)

        self.Lq = torch.linalg.cholesky(self.Q)
        self.H = torch.cat([I, Z], 1)
        self.P0 = self.p0 * torch.eye(self.D, device=device, dtype=dtype)

    def sample(self, n_runs, generator, device, dtype):
        self._build(device, dtype)
        u = torch.rand(n_runs, generator=generator, device=device, dtype=dtype)
        return {"r": self.r_min + (self.r_max - self.r_min) * u}

    def init_state(self, params, generator):
        r = params["r"]
        n = r.shape[0]
        x_true = (self.p0 ** 0.5) * torch.randn(n, self.D, generator=generator, device=r.device, dtype=r.dtype)
        x_hat = torch.zeros_like(x_true)
        P = self.P0.expand(n, -1, -1).clone()
        return x_true, x_hat, P

    def step(self, state, k, params, generator):
        x_true, x_hat, P = state
        r = params["r"]
        n = r.shape[0]
        w = torch.randn(n, self.D, generator=generator, device=r.device, dtype=r.dtype) @ self.Lq.T
        x_true = x_true @ self.F.T + w

        v = r[:, None] * torch.randn(n, self.M, generator=generator, device=r.device, dtype=r.dtype)
        z = x_true @ self.H.T + v

        R = (r**2)[:, None, None] * torch.eye(self.M, device=r.device, dtype=r.dtype)
        x_hat, P = kf_predict(x_hat, P, self.F, self.Q)
        x_hat, P = kf_update(x_hat, P, z, self.H, R)
        return x_true, x_hat, P
    
    def score(self, state, params):
        x_true, x_hat, P = state
        e = x_true - x_hat

        nees = (e[:, None, :] @ torch.linalg.solve(P, e[:, :, None])).squeeze(-1).squeeze(-1)
        return {"pos_err": e[:, :self.dims].norm(dim=1), "nees": nees}



