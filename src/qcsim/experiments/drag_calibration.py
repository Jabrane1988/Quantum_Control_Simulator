from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from qcsim.pulses import drag_pulse
from qcsim.dynamics import simulate_density_matrix
from qcsim.fidelity import ketbra, state_fidelity


def drag_alpha_sweep(
    *,
    T: float = 40.0,
    n_steps: int = 2000,
    amp: float = 0.35,
    sigma: float | None = None,
    delta: float = 0.0,
    T1: float | None = None,
    T2: float | None = None,
    alphas: np.ndarray | None = None,
):
    """
    Sweep DRAG alpha and compute fidelity of target X(pi/2)|0> ≈ |+>.
    This mimics a calibration procedure to reduce leakage/phase errors (simplified for 2-level model).
    """
    if sigma is None:
        sigma = T / 8

    if alphas is None:
        alphas = np.linspace(-2.0, 2.0, 41)

    ts = np.linspace(0.0, T, n_steps)
    t0 = T / 2

    # Target state for X(pi/2) applied to |0> is |+> up to global phase
    psi_plus = (1 / np.sqrt(2)) * np.array([1, 1], dtype=complex)
    target = ketbra(psi_plus)

    fidelities = []
    for a in alphas:
        ox, oy = drag_pulse(ts, amp=amp, t0=t0, sigma=sigma, alpha=a)
        rhos = simulate_density_matrix(ts=ts, ox=ox, oy=oy, delta=delta, T1=T1, T2=T2)
        F = state_fidelity(rhos[-1], target)
        fidelities.append(F)

    fidelities = np.array(fidelities)
    best_idx = int(np.argmax(fidelities))
    best_alpha = float(alphas[best_idx])
    best_F = float(fidelities[best_idx])

    # Plot fidelity vs alpha
    plt.figure()
    plt.plot(alphas, fidelities, marker="o")
    plt.axvline(best_alpha, linestyle="--", label=f"best α={best_alpha:.3f}, F={best_F:.4f}")
    plt.xlabel("DRAG alpha")
    plt.ylabel("Target fidelity (|+>)")
    plt.title(f"DRAG alpha sweep (amp={amp}, sigma={sigma}, Δ={delta}, T1={T1}, T2={T2})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the best pulse envelopes
    ox_best, oy_best = drag_pulse(ts, amp=amp, t0=t0, sigma=sigma, alpha=best_alpha)
    plt.figure()
    plt.plot(ts, ox_best, label="Ωx(t) Gaussian")
    plt.plot(ts, oy_best, label="Ωy(t) α·d/dt Gaussian")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Best DRAG pulse envelopes")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"[DRAG] Best alpha={best_alpha:.6f}, best fidelity={best_F:.6f}")
    return best_alpha, best_F


def run():
    # No noise, small detuning (try detuning to see DRAG utility)
    drag_alpha_sweep(T=40.0, n_steps=2000, amp=0.35, delta=0.25, T1=None, T2=None)

    # With decoherence
    drag_alpha_sweep(T=40.0, n_steps=2000, amp=0.35, delta=0.25, T1=80.0, T2=60.0)


if __name__ == "__main__":
    run()
