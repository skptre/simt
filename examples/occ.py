import math
import os

import numpy as np
import torch

from engine import Scenario
from kalman import kf_update_scalar

AMU = 1.66053906660e-27
MU = 3.986004418e14
RE = 6371008.8
WE = 7.2921159e-5
DT = 30.0
NDAY = 4
T0 = np.datetime64("2024-05-09T00:00")
SPECIES = ("N2", "O2", "O", "He", "H", "Ar", "N")
MASS = (28.0134, 31.9988, 15.9994, 4.002602, 1.00794, 39.948, 14.0067)
GROUP = (1, 1, 0, 2, 2, 1, 0)
R_TRUE = (1.0, 0.90, 0.58, 0.16, 0.42, 1.45, 0.55)
R_FILT = (1.0, 0.90, 0.55, 0.16, 0.42, 1.45, 0.55)
O_IDX = SPECIES.index("O")
STORM_HOURS = (39.0, 72.0)


def ap3_profile(lag, cap):
    n = (NDAY + 3) * 8
    hrs = (np.arange(n) - 24) * 3.0
    ap = np.full(n, 7.0)
    for i, h in enumerate(hrs):
        if 39 <= h < 42:
            ap[i] = 94
        elif 42 <= h < 45:
            ap[i] = 300
        elif 45 <= h < 57:
            ap[i] = 400
        elif 57 <= h < 66:
            ap[i] = 207
        elif 66 <= h < 78:
            ap[i] = 80
        elif 78 <= h < 90:
            ap[i] = 32
        elif h >= 90:
            ap[i] = 15
    if lag:
        ap = np.concatenate([np.full(lag, 7.0), ap[:-lag]])
    if cap is not None:
        ap = np.minimum(ap, cap)
    return hrs, ap


def ap_array(t, lag, cap):
    hrs, ap = ap3_profile(lag, cap)
    idx = np.searchsorted(hrs, t / 3600.0, side="right") - 1
    out = np.zeros((len(t), 7))
    for k, i in enumerate(idx):
        day0 = int((i // 8) * 8)
        out[k, 0] = ap[day0:day0 + 8].mean()
        out[k, 1:5] = ap[i], ap[i - 1], ap[i - 2], ap[i - 3]
        out[k, 5] = ap[i - 11:i - 3].mean()
        out[k, 6] = ap[i - 19:i - 11].mean()
    return out


def orbit(t, alt_km, raan0, inc_deg=51.6):
    r = RE + alt_km * 1e3
    n = math.sqrt(MU / r ** 3)
    u = n * t
    inc = math.radians(inc_deg)
    lat = np.degrees(np.arcsin(np.sin(inc) * np.sin(u)))
    lon_in = np.arctan2(np.cos(inc) * np.sin(u), np.cos(u)) + raan0
    lon = np.degrees((lon_in - WE * t + np.pi) % (2 * np.pi) - np.pi)
    return lat, lon, 2 * math.pi / n


def run_msis(dates, lat, lon, alt_km, f107, f107a, aps):
    from pymsis import msis
    n = len(dates)
    o = msis.calculate(dates, lon, lat, np.full(n, float(alt_km)), np.full(n, f107), np.full(n, f107a),
                       aps, geomagnetic_activity=-1, version=2.1)
    o = np.nan_to_num(o, nan=0.0)
    cols = {"N2": o[:, 1], "O2": o[:, 2], "O": o[:, 3] + o[:, 8], "He": o[:, 4], "H": o[:, 5],
            "Ar": o[:, 6], "N": o[:, 7]}
    return np.stack([cols[s] for s in SPECIES], axis=1)


def build_tables(alt_km, n_phases, cache_dir="cache"):
    path = os.path.join(cache_dir, f"occ_msis_{alt_km}km_{n_phases}ph.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["truth"], d["prior"], int(d["nper"])
    t = np.arange(0, NDAY * 86400, DT)
    dates = T0 + (t * 1e3).astype("timedelta64[ms]")
    ap_truth = ap_array(t, 0, None)
    ap_prior = ap_array(t, 2, 100.0)
    truth = np.zeros((n_phases, len(t), len(SPECIES)))
    prior = np.zeros_like(truth)
    period = 0.0
    for j in range(n_phases):
        lat, lon, period = orbit(t, alt_km, 2 * math.pi * j / n_phases)
        truth[j] = run_msis(dates, lat, lon, alt_km, 220.0, 170.0, ap_truth)
        prior[j] = run_msis(dates, lat, lon, alt_km, 200.0, 170.0, ap_prior)
    nper = int(round(period / DT))
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, truth=truth, prior=prior, nper=nper)
    return truth, prior, nper


class OCC(Scenario):

    def __init__(self, alt_km=300, n_phases=32, cache_dir="cache",
                 use_gauge=True, use_o=True, use_drag=True,
                 r_true=R_TRUE, r_filt=R_FILT,
                 gauge_gain_sd=0.08, gauge_drift_sd=0.02, gauge_noise=0.01,
                 o_gain_sd=0.12, o_drift_mean=0.03, o_drift_sd=0.01, o_noise=0.05,
                 cd_sd=0.07, drag_noise=0.03,
                 p0_sd=(0.5, 0.5, 0.7, 0.10, 0.15, 0.10),
                 q_sd=(0.02, 0.02, 0.01, 0.0005, 0.0008, 0.0002),
                 r_gauge=0.012, r_o=0.05, r_drag=0.035,
                 common_sd=0.05, common_tau=600.0, species_sd=0.02, species_tau=900.0):
        self.alt_km = alt_km
        self.n_phases = n_phases
        self.use_gauge, self.use_o, self.use_drag = use_gauge, use_o, use_drag
        self.r_true, self.r_filt = r_true, r_filt
        self.gauge_gain_sd, self.gauge_drift_sd, self.gauge_noise = gauge_gain_sd, gauge_drift_sd, gauge_noise
        self.o_gain_sd, self.o_drift_mean, self.o_drift_sd, self.o_noise = o_gain_sd, o_drift_mean, o_drift_sd, o_noise
        self.cd_sd, self.drag_noise = cd_sd, drag_noise
        self.p0_sd, self.q_sd = p0_sd, q_sd
        self.r_gauge, self.r_o, self.r_drag = r_gauge, r_o, r_drag
        self.a_common = math.exp(-DT / common_tau)
        self.kick_common = common_sd * math.sqrt(1 - self.a_common ** 2)
        self.a_species = math.exp(-DT / species_tau)
        self.kick_species = species_sd * math.sqrt(1 - self.a_species ** 2)
        self.truth_np, self.prior_np, self.nper = build_tables(alt_km, n_phases, cache_dir)
        self.n_steps = self.truth_np.shape[1]
        self._key = None

    def _build(self, device, dtype):
        key = (str(device), dtype)
        if self._key == key:
            return
        self._key = key

        def f(a):
            return torch.as_tensor(np.asarray(a), device=device, dtype=dtype)

        cum = np.cumsum(self.prior_np, axis=1)
        shifted = np.zeros_like(cum)
        shifted[:, self.nper:] = cum[:, :-self.nper]
        self.truth = f(self.truth_np)
        self.prior = f(self.prior_np)
        self.prior_win = f((cum - shifted) / self.nper)
        self.mass = f(MASS)
        self.r_true_t = f(self.r_true)
        self.r_filt_t = f(self.r_filt)
        self.gidx = torch.as_tensor(GROUP, device=device)
        self.G = f(np.eye(3)[list(GROUP)])
        self.P0 = torch.diag(f(self.p0_sd) ** 2)
        self.Qd = torch.diag(f(self.q_sd) ** 2)

    def sample(self, n_runs, generator, device, dtype):
        self._build(device, dtype)

        def normal(mean, sd):
            return mean + sd * torch.randn(n_runs, generator=generator, device=device, dtype=dtype)

        return {
            "phase": torch.randint(0, self.n_phases, (n_runs,), generator=generator, device=device),
            "gauge_gain": normal(0.0, self.gauge_gain_sd),
            "gauge_drift": normal(0.0, self.gauge_drift_sd) / 86400,
            "o_gain": normal(0.0, self.o_gain_sd),
            "o_drift": -normal(self.o_drift_mean, self.o_drift_sd).abs() / 86400,
            "cd_bias": normal(0.0, self.cd_sd),
        }

    def init_state(self, params, generator):
        ref = params["gauge_gain"]
        n = ref.shape[0]

        def zeros(*shape):
            return torch.zeros(*shape, device=ref.device, dtype=ref.dtype)

        return {
            "common": zeros(n),
            "species": zeros(n, len(SPECIES)),
            "x": zeros(n, 6),
            "P": self.P0.expand(n, 6, 6).clone(),
            "truth_ring": zeros(self.nper, n),
            "truth_sum": zeros(n),
            "est_ring": zeros(self.nper, n),
            "est_sum": zeros(n),
            "err_prior": zeros(n),
            "err_gauge": zeros(n),
            "err_fused": zeros(n),
            "n_storm": 0,
        }

    def _group_H(self, terms, col):
        tot = terms.sum(1)
        H = torch.zeros(terms.shape[0], 6, device=terms.device, dtype=terms.dtype)
        H[:, :3] = (terms / tot[:, None]) @ self.G
        H[:, col] = 1.0
        return H, tot

    def step(self, s, k, p, generator):
        x, P = s["x"], s["P"]
        n, dev, dt = x.shape[0], x.device, x.dtype

        def randn(*shape):
            return torch.randn(*shape, generator=generator, device=dev, dtype=dt)

        if k > 0:
            s["common"] = self.a_common * s["common"] + self.kick_common * randn(n)
            s["species"] = self.a_species * s["species"] + self.kick_species * randn(n, len(SPECIES))

        tk = k * DT
        n_true = self.truth[p["phase"], k] * torch.exp(s["common"][:, None] + s["species"])
        n_prior = self.prior[p["phase"], k]
        rho_true = (n_true * self.mass).sum(1) * AMU
        rho_prior = (n_prior * self.mass).sum(1) * AMU

        slot = k % self.nper
        s["truth_sum"] = s["truth_sum"] + rho_true - s["truth_ring"][slot]
        s["truth_ring"][slot] = rho_true

        z_gauge = (torch.log((n_true * self.r_true_t).sum(1)) + p["gauge_gain"] + p["gauge_drift"] * tk
                   + self.gauge_noise * randn(n))
        z_o = torch.log(n_true[:, O_IDX]) + p["o_gain"] + p["o_drift"] * tk + self.o_noise * randn(n)

        n_tot_prior = n_prior.sum(1)
        r_bar = (n_prior * self.r_filt_t).sum(1) / n_tot_prior
        rho_gauge = torch.exp(z_gauge) / r_bar * rho_prior / n_tot_prior

        P = P + self.Qd
        if self.use_gauge:
            e = torch.exp(x[:, self.gidx])
            H, tot = self._group_H(self.r_filt_t * n_prior * e, 3)
            x, P = kf_update_scalar(x, P, H, z_gauge, torch.log(tot) + x[:, 3], self.r_gauge ** 2)
        if self.use_o:
            H = torch.zeros(n, 6, device=dev, dtype=dt)
            H[:, 0] = 1.0
            H[:, 4] = 1.0
            x, P = kf_update_scalar(x, P, H, z_o, torch.log(n_prior[:, O_IDX]) + x[:, 0] + x[:, 4],
                                    self.r_o ** 2)
        if self.use_drag and (k + 1) % self.nper == 0:
            z_drag = torch.log(s["truth_sum"] / self.nper) + p["cd_bias"] + self.drag_noise * randn(n)
            e = torch.exp(x[:, self.gidx])
            H, _ = self._group_H(self.mass * self.prior_win[p["phase"], k] * e, 5)
            rho_now = (n_prior * e * self.mass).sum(1) * AMU
            h = torch.log((s["est_sum"] - s["est_ring"][slot] + rho_now) / self.nper) + x[:, 5]
            x, P = kf_update_scalar(x, P, H, z_drag, h, self.r_drag ** 2)

        rho_est = (n_prior * torch.exp(x[:, self.gidx]) * self.mass).sum(1) * AMU
        s["est_sum"] = s["est_sum"] + rho_est - s["est_ring"][slot]
        s["est_ring"][slot] = rho_est
        s["x"], s["P"] = x, P

        if STORM_HOURS[0] <= tk / 3600 < STORM_HOURS[1]:
            s["err_prior"] += (rho_prior / rho_true - 1).abs()
            s["err_gauge"] += (rho_gauge / rho_true - 1).abs()
            s["err_fused"] += (rho_est / rho_true - 1).abs()
            s["n_storm"] += 1
        return s

    def score(self, s, params):
        n = max(s["n_storm"], 1)
        return {
            "storm_err_prior": s["err_prior"] / n,
            "storm_err_gauge": s["err_gauge"] / n,
            "storm_err_fused": s["err_fused"] / n,
        }
