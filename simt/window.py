import torch

class RollingMean:
    def __init__(self, window, n, device, dtype):
        self.window = window    # W: how many recent values to average
        self.buf = torch.zeros(window, n, device=device, dtype=dtype)   # ring buffer: last W values, every run gets one
        self.total = torch.zeros(n, device=device, dtype=dtype)   # running sum of what's in the buffer
        self.count = 0   # values added so far (stops growing at W)
        self.pos = 0   # slot the NEXT value goes into (= oldest)


    def push(self, x):
        self.total += x - self.buf[self.pos]
        self.buf[self.pos] = x
        self.pos = (self.pos + 1) % self.window
        self.count = min(self.count + 1, self.window)
        if self.pos == 0:
            self.total = self.buf.sum(0)
        
    def mean(self):
        return self.total / max(self.count, 1)

    def peek_mean(self, x):
        if self.count < self.window:
            return (self.total + x) / (self.count + 1)
        return (self.total - self.buf[self.pos] + x) / self.window

    def full(self):
        return self.count == self.window