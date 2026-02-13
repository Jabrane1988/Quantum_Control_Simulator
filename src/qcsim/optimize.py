from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from .pulses import drag_pulse
from .dynamics import simulate_density_matrix
from .fidelity import ketbra, state_fidelity

def objective_x_pi_over_2(params, *, T: float, n_steps: int, delta: float, T1, T2):
    amp, sigma, alpha = params
    ts = np.linspace(0, T, n_steps)
    t0 = T / 2

    ox, oy = drag_pulse(ts, amp=amp, t0=t0, sigma=sigma, alpha=alpha)
    rhos = simulate_density_matrix(ts=ts, ox=ox, oy=oy, delta=delta, T1=T1, T2=T2)

    # Target: X(pi/2) applied to |0> gives |+> (up to phase)
    psi_plus = (1/np.sqrt(2)) * np.array([1, 1], dtype=complex)
    target = ketbra(psi_plus)

    final_rho = rhos[-1]
    F = state_fidelity(final_rho, target)
    return 1.0 - F  # minimize

def optimize_drag_for_x_pi_over_2(*, T=40.0, n_steps=2000, delta=0.0, T1=None, T2=None):
    x0 = np.array([0.2, T/8, 0.0])  # initial guess: (amp, sigma, alpha)
    bounds = [(0.0, 5.0), (T/50, T/2), (-5.0, 5.0)]
    res = minimize(
        objective_x_pi_over_2,
        x0=x0,
        bounds=bounds,
        args=(),
        kwargs={"T": T, "n_steps": n_steps, "delta": delta, "T1": T1, "T2": T2},
        method="L-BFGS-B",
    )
    return res
