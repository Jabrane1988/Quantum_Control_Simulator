import numpy as np
import matplotlib.pyplot as plt

from qcsim.pulses import gaussian
from qcsim.dynamics import simulate_density_matrix

def run():
    T = 40.0
    ts = np.linspace(0, T, 2000)
    t0 = T/2
    sigma = T/8

    amps = np.linspace(0.0, 1.0, 25)
    p1 = []

    for amp in amps:
        ox = gaussian(ts, amp=amp, t0=t0, sigma=sigma)
        oy = np.zeros_like(ox)
        rhos = simulate_density_matrix(ts=ts, ox=ox, oy=oy, delta=0.0, T1=None, T2=None)
        rhoT = rhos[-1]
        p_excited = np.real(rhoT[1, 1])
        p1.append(p_excited)

    plt.figure()
    plt.plot(amps, p1, marker="o")
    plt.xlabel("Pulse amplitude")
    plt.ylabel("P(|1>) at final time")
    plt.title("Rabi experiment (Gaussian drive, no noise)")
    plt.show()

if __name__ == "__main__":
    run()
