import numpy as np
from qcsim.dynamics import simulate_density_matrix
from qcsim.pulses import gaussian

def test_unitary_limit_trace_preserved_no_noise():
    T = 10.0
    ts = np.linspace(0.0, T, 800)
    t0 = T / 2
    sigma = T / 6

    # Some drive, but no noise => unitary evolution should preserve trace
    ox = gaussian(ts, amp=0.7, t0=t0, sigma=sigma)
    oy = np.zeros_like(ox)

    rhos = simulate_density_matrix(ts=ts, ox=ox, oy=oy, delta=0.3, T1=None, T2=None)

    traces = np.real(np.trace(rhos, axis1=1, axis2=2))
    assert np.allclose(traces, 1.0, rtol=1e-8, atol=1e-8)

def test_unitary_limit_rho_is_hermitian_no_noise():
    T = 8.0
    ts = np.linspace(0.0, T, 600)
    t0 = T / 2
    sigma = T / 5

    ox = gaussian(ts, amp=0.6, t0=t0, sigma=sigma)
    oy = gaussian(ts, amp=0.2, t0=t0, sigma=sigma)  # add Y quadrature too

    rhos = simulate_density_matrix(ts=ts, ox=ox, oy=oy, delta=0.1, T1=None, T2=None)

    for rho in rhos[::50]:
        assert np.allclose(rho, rho.conj().T, rtol=1e-10, atol=1e-10)

def test_populations_are_valid_probabilities():
    T = 10.0
    ts = np.linspace(0.0, T, 800)
    t0 = T / 2
    sigma = T / 6

    ox = gaussian(ts, amp=0.9, t0=t0, sigma=sigma)
    oy = np.zeros_like(ox)

    rhos = simulate_density_matrix(ts=ts, ox=ox, oy=oy, delta=0.0, T1=None, T2=None)

    p0 = np.real(rhos[:, 0, 0])
    p1 = np.real(rhos[:, 1, 1])

    # probabilities should be in [0,1] up to small numerical tolerance
    assert np.all(p0 > -1e-8)
    assert np.all(p1 > -1e-8)
    assert np.all(p0 < 1.0 + 1e-8)
    assert np.all(p1 < 1.0 + 1e-8)

    # p0 + p1 = 1
    assert np.allclose(p0 + p1, 1.0, rtol=1e-8, atol=1e-8)
