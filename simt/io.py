import json
import os

import numpy as np
import torch

from simt.engine import Results

INFO_KEY = "__info__"


def save_results(results, path, meta=None):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    info = {
        "param_names": getattr(results, "param_names", []),
        "metric_names": getattr(results, "metric_names", []),
        "meta": meta or {},
    }
    arrays = {name: value.numpy() for name, value in results.items()}
    np.savez_compressed(path, **{INFO_KEY: np.array(json.dumps(info, default=str))}, **arrays)


def load_results(path):
    with np.load(path) as d:
        info = json.loads(str(d[INFO_KEY]))
        data = {name: torch.from_numpy(d[name]) for name in d.files if name != INFO_KEY}
    return Results(data, info["param_names"], info["metric_names"]), info["meta"]
