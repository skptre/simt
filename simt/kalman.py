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

def kf_update(x, P, z, H, R, joseph=False):
    # UPDATE: correct the estimate using a new measurement.
    # z: [N, M] each run's measurement
    # H: [M, D] what the sensor sees 
    # R: [N, M, M] each run's SENSOR noise; different sensor quality per run

    y = z - x @ H.T
    S = H @ P @ H.T + R
    K = torch.linalg.solve(S, H @ P).mT
    x = x + (K @ y.unsqueeze(-1)).squeeze(-1)

    I = torch.eye(P.shape[-1], device=P.device, dtype=P.dtype)
    if joseph:
        A = I - K @ H
        P = A @ P @ A.mT + K @ R @ K.mT
    else:
        P = (I - K @ H) @ P
    return x, P

def kf_update_scalar(x, P, H, z, h, r2, mask=None, gate=None, joseph=False, return_info=False):
    if H.dim() == 1:
        H = H.expand(x.shape[0], -1)
    
    PH = (P @ H.unsqueeze(-1)).squeeze(-1)
    S = (H * PH).sum(1) + r2
    nu = z - h
    nis = nu ** 2 / S

    accept = torch.isfinite(nu)
    if mask is not None:
        accept = accept & mask
    if gate is not None:
        accept = accept & (nis < gate)

    K = PH / S[:, None] * accept[:, None].to(PH.dtype)
    nu = torch.where(accept, nu, torch.zeros_like(nu))
    x = x + K * nu[:, None]

    if joseph:
        I = torch.eye(P.shape[-1], device=P.device, dtype=P.dtype)
        A =  I - K[:, :, None] * H[:, None, :]

        r2_t = torch.as_tensor(r2, device=P.device, dtype=P.dtype)
        if r2_t.dim() == 1:
            r2_t = r2_t[:, None, None]
        P = A @ P @ A.mT + r2_t * (K[:, :, None] * K[:, None, :])
    else:
        P = P - K[:, :, None] * PH[:, None, :]

    
    if return_info:
        return x, P, {"nis": nis, "accepted": accept}

    return x, P
