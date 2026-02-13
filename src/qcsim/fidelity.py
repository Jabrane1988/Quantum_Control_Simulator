from __future__ import annotations
import numpy as np

def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    For pure sigma = |psi><psi|, fidelity reduces to Tr(rho*sigma).
    We'll use this for v1 experiments (prepare known pure states).
    """
    return float(np.real(np.trace(rho @ sigma)))

def ketbra(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T
