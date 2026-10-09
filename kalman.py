import torch

def kf_predict(x, P, F, Q):
    # PREDICT: push the estimate forward one step using the motion model.
    # x: [N, D] each run's state estimate
    # P: [N, D, D] each run's uncertainty about x (covariance matrix)
    # F: [D, D] motion model, shared by all runs ("x_next = F x")
    # Q: [D, D] how much randomness the motion adds per step
    
    x = x @ F.T
    P = F @ P @ F.T + Q
    return x, P

def kf_update(x, P, z, H, R):
    # UPDATE: correct the estimate using a new measurement.
    # z: [N, M] each run's measurement
    # H: [M, D] what the sensor sees 
    # R: [N, M, M] each run's SENSOR noise; different sensor quality per run

    y = z - x @ H.T
    S = H @ P @ H.T + R
    K = torch.linalg.solve(S, H @ P).mT
    x = x + (K @ y.unsqueeze(-1)).squeeze(-1)

    I = torch.eye(P.shape[-1], device=P.device, dtype=P.dtype)
    P = (I - K @ H) @ P
    return x, P

def kf_update_scalar(x, P, H, z, h, r2):
    PH = (P @ H.unsqueeze(-1)).squeeze(-1)
    S = (H * PH).sum(1) + r2
    K = PH / S[:, None]
    x = x + K * (z - h)[:, None]
    P = P - K[:, :, None] * PH[:, None, :]
    return x, P
