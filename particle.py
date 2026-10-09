import torch
from engine import Scenario


class Particle(Scenario):

    def __init__(self, dt=1.0, v_mean=1.0, v_std=0.1, x0_std=0.5):
        self.dt = dt
        self.v_mean = v_mean
        self.v_std = v_std
        self.x0_std = x0_std

    def sample(self, n_runs, generator, device, dtype):
        v = self.v_mean + self.v_std * torch.randn(n_runs, generator=generator, device=device, dtype=dtype)
        x0 = self.x0_std * torch.randn(n_runs, generator=generator, device=device, dtype=dtype)
        return {"v": v, "x0": x0}

    def init_state(self, params, generator):
        return params["x0"]

    def step(self, state, k, params, generator):
        return state + params["v"] * self.dt

    def score(self, state, params):
        return {"final_x": state}
