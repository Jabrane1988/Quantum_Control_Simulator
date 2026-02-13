from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

from .constants import sx, sy, sz
from .noise import lindblad_ops

def hamiltonian(delta: float, ox: float, oy: float) -> np.ndarray:
    # H = 0.5*(delta*sz + ox*sx + oy*sy)
    return 0.5 * (delta * sz + ox * sx + oy * sy)

def master_rhs(t: float, rho_flat: np.ndarray, *, ts: np.ndarray, ox: np.ndarray, oy: np.ndarray,
               delta: float, T1: float | None, T2: float | None) -> np.ndarray:
    rho = rho_flat.reshape((2, 2))

    # interpolate controls
    ox_t = np.interp(t, ts, ox)
    oy_t = np.interp(t, ts, oy)
    H = hamiltonian(delta, ox_t, oy_t)

    # unitary part
    drho = -1j * (H @ rho - rho @ H)

    # dissipator
    for L in lindblad_ops(T1, T2):
        LdL = L.conj().T @ L
        drho += (L @ rho @ L.conj().T) - 0.5 * (LdL @ rho + rho @ LdL)

    return drho.reshape(-1)

def simulate_density_matrix(*, ts: np.ndarray, ox: np.ndarray, oy: np.ndarray,
                            delta: float = 0.0, T1: float | None = None, T2: float | None = None,
                            rho0: np.ndarray | None = None) -> np.ndarray:
    if rho0 is None:
        # start in |0><0|
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)

    sol = solve_ivp(
        fun=lambda t, y: master_rhs(t, y, ts=ts, ox=ox, oy=oy, delta=delta, T1=T1, T2=T2),
        t_span=(ts[0], ts[-1]),
        y0=rho0.reshape(-1),
        t_eval=ts,
        rtol=1e-7,
        atol=1e-9,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    rhos = sol.y.T.reshape((-1, 2, 2))
    return rhos
