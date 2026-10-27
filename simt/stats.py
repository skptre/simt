import torch


def percentile(x, q, dim=-1):
    s = x.sort(dim=dim).values
    n = s.shape[dim]
    pos = q / 100 * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s.select(dim, lo) * (1 - frac) + s.select(dim, hi) * frac


def _clean(x):
    x = x.flatten().double()
    return x[~torch.isnan(x)]


def summarize(x, percentiles=(10, 50, 90)):
    x = _clean(x)
    if x.numel() == 0:
        return {"mean": float("nan"), **{f"p{q}": float("nan") for q in percentiles}}
    out = {"mean": x.mean().item()}
    for q in percentiles:
        out[f"p{q}"] = percentile(x, q).item()
    return out


def bootstrap_ci(x, q=50, n_boot=1000, level=0.95, seed=0):
    x = _clean(x)
    n = x.numel()
    if n == 0:
        return float("nan"), float("nan")
    g = torch.Generator().manual_seed(seed)
    batch = max(1, min(100, 20_000_000 // n))
    stats = []
    for first in range(0, n_boot, batch):
        b = min(batch, n_boot - first)
        idx = torch.randint(0, n, (b, n), generator=g)
        stats.append(percentile(x[idx], q, dim=1))
    stats = torch.cat(stats)
    tail = (1 - level) / 2 * 100
    return percentile(stats, tail).item(), percentile(stats, 100 - tail).item()
