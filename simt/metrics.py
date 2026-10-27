import torch


class RunningStats:

    def __init__(self, n, device, dtype):
        def zeros():
            return torch.zeros(n, device=device, dtype=dtype)

        self.count = zeros()
        self.total = zeros()
        self.total_sq = zeros()
        self.total_abs = zeros()

    def add(self, x, mask=None):
        if mask is None:
            self.count += 1
        else:
            x = torch.where(mask, x, torch.zeros_like(x))
            self.count += mask.to(x.dtype)
        self.total += x
        self.total_sq += x * x
        self.total_abs += x.abs()

    def summary(self, prefix):
        n = self.count.clamp(min=1)
        mean = self.total / n
        mean_sq = self.total_sq / n
        values = {
            "bias": mean,
            "std": (mean_sq - mean * mean).clamp(min=0).sqrt(),
            "mae": self.total_abs / n,
            "rms": mean_sq.sqrt(),
        }
        empty = self.count == 0
        nan = torch.full_like(mean, float("nan"))
        return {f"{prefix}_{k}": torch.where(empty, nan, v) for k, v in values.items()}
