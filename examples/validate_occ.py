import csv
import os
import platform
import sys
import time

import torch

from engine import run
from examples.occ import OCC

REFERENCE = {
    300: {"prior": (25.91, 25.63, 26.08), "gauge": (6.68, 5.69, 8.14), "fused": (4.63, 2.50, 5.60)},
    450: {"prior": (35.97, 35.79, 36.27), "gauge": (7.52, 6.31, 8.32), "fused": (4.59, 2.48, 5.90)},
}
NAMES = ("prior", "gauge", "fused")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else (10_000 if device == "cuda" else 20)
    rows = []

    for alt in (300, 450):
        start = time.perf_counter()
        sc = OCC(alt_km=alt)
        t_tables = time.perf_counter() - start
        results, elapsed = run(sc, n_runs, sc.n_steps, device=device, dtype=torch.float64)
        print(f"{alt} km  N={n_runs}  tables {t_tables:.1f} s  sim {elapsed:.1f} s  {n_runs / elapsed:.1f} runs/s")
        hardware = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()
        row = {"device": device, "hardware": hardware, "dtype": "float64", "alt_km": alt, "n_runs": n_runs,
               "n_phases": sc.n_phases, "n_steps": sc.n_steps, "tables_s": t_tables, "sim_s": elapsed,
               "sim_s_per_run": elapsed / n_runs}
        for name in NAMES:
            ref, lo, hi = REFERENCE[alt][name]
            err = results[f"storm_err_{name}"] * 100
            med, p10, p90 = err.median().item(), err.quantile(0.1).item(), err.quantile(0.9).item()
            ok = "PASS" if lo <= med <= hi else "CHECK"
            print(f"  {name:<6} median {med:5.1f}%  (P10 {p10:4.1f}, P90 {p90:4.1f})   "
                  f"reference {ref:5.1f}% [95% CI {lo:.1f}-{hi:.1f}]  {ok}")
            row.update({f"{name}_median": med, f"{name}_p10": p10, f"{name}_p90": p90,
                        f"{name}_ref": ref, f"{name}_ref_lo": lo, f"{name}_ref_hi": hi})
        rows.append(row)

    os.makedirs("data", exist_ok=True)
    with open(f"data/occ_validation_{device}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
