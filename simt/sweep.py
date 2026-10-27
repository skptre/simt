import csv
import itertools
import os
import time

import torch

from simt.engine import run
from simt.io import save_results
from simt.stats import summarize


def grid(**axes):
    names = list(axes)
    return [dict(zip(names, values)) for values in itertools.product(*axes.values())]


def write_csv(rows, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def sweep(make_scenario, configs, n_runs, n_steps=None, seed=0, device="cpu", dtype=torch.float32,
          chunk_size=100_000, percentiles=(10, 50, 90), csv_path=None, save_dir=None):
    rows = []
    start = time.perf_counter()

    for i, cfg in enumerate(configs):
        scenario = make_scenario(**cfg)
        results, seconds = run(scenario, n_runs, n_steps, seed=seed, device=device, dtype=dtype,
                               chunk_size=chunk_size)

        row = {**cfg, "n_runs": n_runs, "seconds": seconds}
        for name in results.metric_names:
            for stat, value in summarize(results[name], percentiles).items():
                row[f"{name}_{stat}"] = value
        rows.append(row)

        if save_dir:
            save_results(results, os.path.join(save_dir, f"config_{i:03d}.npz"), meta=cfg)
        if csv_path:
            write_csv(rows, csv_path)

        spent = time.perf_counter() - start
        left = spent / (i + 1) * (len(configs) - i - 1)
        print(f"[{i + 1}/{len(configs)}] {cfg}  {seconds:.1f} s  (~{left / 60:.1f} min left)", flush=True)

    return rows
