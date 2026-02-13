from __future__ import annotations
import numpy as np
from .constants import sm, sz

def lindblad_ops(T1: float | None, T2: float | None):
    """
    Returns Lindblad operators for T1 relaxation and Tphi dephasing.
    If T1/T2 are None, those channels are disabled.
    """
    Ls = []

    if T1 is not None and T1 > 0:
        gamma1 = 1.0 / T1
        Ls.append(np.sqrt(gamma1) * sm)

    if T2 is not None and T2 > 0:
        # 1/Tphi = 1/T2 - 1/(2T1)
        gamma2 = 1.0 / T2
        gamma_phi = gamma2
        if T1 is not None and T1 > 0:
            gamma_phi = max(0.0, gamma2 - 0.5 * (1.0 / T1))
        if gamma_phi > 0:
            Ls.append(np.sqrt(gamma_phi) * sz)

    return Ls
