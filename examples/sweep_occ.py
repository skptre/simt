import csv
import os
import sys

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import run
from examples.occ import OCC

GAUGE_ERRORS = (0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40)
P0_DEFAULT = (0.5, 0.5, 0.7, 0.10, 0.15, 0.10)
FILTER_RATIO = 0.10 / 0.08
TARGET = 20.0

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else (10_000 if device == "cuda" else 200)
    alt = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    rows = []

    for use_o in (True, False):
        for g in GAUGE_ERRORS:
            p0 = list(P0_DEFAULT)
            p0[3] = FILTER_RATIO * g
            sc = OCC(alt_km=alt, use_o=use_o, gauge_gain_sd=g, p0_sd=tuple(p0))
            results, elapsed = run(sc, n_runs, sc.n_steps, device=device, dtype=torch.float64)
            fused = results["storm_err_fused"] * 100
            row = {
                "alt_km": alt, "use_o": use_o, "gauge_err_pct": g * 100, "n_runs": n_runs, "sim_s": elapsed,
                "fused_median": fused.median().item(),
                "fused_p10": fused.quantile(0.1).item(),
                "fused_p90": fused.quantile(0.9).item(),
                "gauge_only_median": (results["storm_err_gauge"] * 100).median().item(),
                "prior_median": (results["storm_err_prior"] * 100).median().item(),
            }
            rows.append(row)
            print(f"O sensor {'on ' if use_o else 'off'}  gauge +-{g*100:4.0f}%  fused median {row['fused_median']:5.2f}%  "
                  f"(P10 {row['fused_p10']:4.1f}, P90 {row['fused_p90']:4.1f})  gauge-only {row['gauge_only_median']:5.1f}%  "
                  f"{elapsed:5.1f} s")

    os.makedirs("data", exist_ok=True)
    stem = f"data/occ_sweep_gauge_{alt}km_{device}"
    with open(f"{stem}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for use_o, color, label in ((True, "#2a78d6", "Fusion: gauge + O sensor + drag"),
                                (False, "#e8743b", "Fusion: gauge + drag (no O sensor)")):
        sel = [r for r in rows if r["use_o"] == use_o]
        x = [r["gauge_err_pct"] for r in sel]
        ax.fill_between(x, [r["fused_p10"] for r in sel], [r["fused_p90"] for r in sel], color=color, alpha=0.15)
        ax.plot(x, [r["fused_median"] for r in sel], "o-", color=color, label=label)
    base = [r for r in rows if r["use_o"]]
    ax.plot([r["gauge_err_pct"] for r in base], [r["gauge_only_median"] for r in base], "s--", color="#888888",
            label="Gauge alone (naive)")
    ax.axhline(base[0]["prior_median"], color="#555555", linestyle=":", label="Empirical model alone")
    ax.axhline(TARGET, color="black", linewidth=1, label="OCC 20% target")
    ax.set_xlabel("Gauge calibration error, 1σ (%)")
    ax.set_ylabel("Storm-time density error (%)")
    ax.set_title(f"Fused accuracy vs gauge calibration, {alt} km ({n_runs:,} runs per point, band = P10–P90)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150)
