import csv
import os
import platform
import sys
import time

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "occ_fusion_osse")
    sys.path.insert(0, os.path.abspath(src))
    import fusion_osse

    rows = []
    for alt in (300, 450):
        fusion_osse.simulate(alt, seed=1, configs=["fused_gauge_O_drag"])
        for seed in range(2, 7):
            start = time.perf_counter()
            fusion_osse.simulate(alt, seed=seed, configs=["fused_gauge_O_drag"])
            t = time.perf_counter() - start
            rows.append({"code": "original fusion_osse.py", "hardware": platform.processor(),
                         "config": "fused_gauge_O_drag", "alt_km": alt, "seed": seed, "seconds_per_run": t})
            print(f"{alt} km  seed {seed}  {t:.2f} s")

    os.makedirs("data", exist_ok=True)
    with open("data/occ_original_timing.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
