from __future__ import annotations
import numpy as np

def gaussian(t: np.ndarray, amp: float, t0: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

def d_gaussian_dt(t: np.ndarray, amp: float, t0: float, sigma: float) -> np.ndarray:
    g = gaussian(t, amp, t0, sigma)
    return g * (-(t - t0) / (sigma**2))

def drag_pulse(t: np.ndarray, amp: float, t0: float, sigma: float, alpha: float):
    """
    Returns (Omega_x(t), Omega_y(t)) for DRAG:
      Omega_x = gaussian
      Omega_y = alpha * d/dt(gaussian)
    """
    ox = gaussian(t, amp, t0, sigma)
    oy = alpha * d_gaussian_dt(t, amp, t0, sigma)
    return ox, oy
