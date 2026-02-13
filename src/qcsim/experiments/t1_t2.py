from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qcsim.dynamics import simulate_density_matrix
from qcsim.fidelity import ketbra, state_fidelity


def _ket0():
    return np.array([1, 0], dtype=complex)


def _ket1():
    return np.array([0, 1], dtype=complex)


def _ket_plus():
    return (1 / np.sqrt(2)) * np.array([1, 1], dtype=complex)


def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C


def t1_decay_demo(T1: float = 50.0, t_max: float = 200.0, n_steps: int = 1200):
    """
    T1 experiment:
      - No drive (Ωx=Ωy=0)
      - Initial state |1><1|
      - Observe P(|1>) vs time, fit to exp decay ~ exp(-t/T1)
    """
    ts = np.linspace(0.0, t_max, n_steps)
    ox = np.zeros_like(ts)
    oy = np.zeros_like(ts)

    rho1 = ketbra(_ket1())

    rhos = simulate_density_matrix(
        ts=ts, ox=ox, oy=oy, delta=0.0, T1=T1, T2=None, rho0=rho1
    )
    p1 = np.real(rhos[:, 1, 1])

    # Fit: p1(t) ≈ A exp(-t/T1) + C
    popt, _ = curve_fit(exp_decay, ts, p1, p0=[1.0, T1, 0.0], maxfev=10000)
    A_hat, T1_hat, C_hat = popt
    fit = exp_decay(ts, *popt)

    plt.figure()
    plt.plot(ts, p1, label="Simulated P(|1>)")
    plt.plot(ts, fit, "--", label=f"Fit: T1≈{T1_hat:.2f}")
    plt.xlabel("Time")
    plt.ylabel("P(|1>)")
    plt.title("T1 Relaxation (no drive)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"[T1] True T1={T1}, fitted T1≈{T1_hat:.4f} (A={A_hat:.3f}, C={C_hat:.3f})")


def t2_dephasing_demo(T1: float | None = None, T2: float = 60.0, delta: float = 0.2,
                      t_max: float = 200.0, n_steps: int = 1400):
    """
    Simple Ramsey-style dephasing experiment (free evolution only):
      - No drive (Ωx=Ωy=0)
      - Initial state |+><+| (superposition)
      - Evolve under detuning Δ and dephasing T2
      - Measure overlap with |+> as a proxy for coherence decay.

    This is a clean way to visualize T2 without implementing explicit π/2 pulses.
    """
    ts = np.linspace(0.0, t_max, n_steps)
    ox = np.zeros_like(ts)
    oy = np.zeros_like(ts)

    rho_plus = ketbra(_ket_plus())
    target_plus = rho_plus

    rhos = simulate_density_matrix(
        ts=ts, ox=ox, oy=oy, delta=delta, T1=T1, T2=T2, rho0=rho_plus
    )

    # Proxy measurement: fidelity with |+><+|
    F_plus = np.array([state_fidelity(rho, target_plus) for rho in rhos])

    # Fit with damped cosine envelope is possible, but keep v1 simple:
    # show coherence decay envelope behavior
    plt.figure()
    plt.plot(ts, F_plus, label="Fidelity to |+>")
    plt.xlabel("Time")
    plt.ylabel("F(|+>)")
    plt.title(f"T2 Dephasing (free evolution), T2={T2}, Δ={delta}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"[T2] Ran free-evolution dephasing demo with T2={T2}, delta={delta}, T1={T1}")


def run():
    # You can tune these values
    t1_decay_demo(T1=50.0, t_max=200.0, n_steps=1200)
    t2_dephasing_demo(T1=50.0, T2=60.0, delta=0.25, t_max=200.0, n_steps=1400)


if __name__ == "__main__":
    run()
