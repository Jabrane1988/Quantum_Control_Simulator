import numpy as np
from qcsim.pulses import gaussian, d_gaussian_dt, drag_pulse

def test_gaussian_peaks_at_t0():
    ts = np.linspace(0.0, 10.0, 2001)
    amp = 0.7
    t0 = 5.0
    sigma = 1.2

    g = gaussian(ts, amp=amp, t0=t0, sigma=sigma)
    idx_peak = int(np.argmax(g))

    # Peak should be very close to t0 (grid resolution dependent)
    assert abs(ts[idx_peak] - t0) < (ts[1] - ts[0]) * 2

    # Peak height should be close to amp
    assert np.isclose(g[idx_peak], amp, rtol=1e-3, atol=1e-4)

def test_gaussian_derivative_zero_at_center():
    ts = np.linspace(0.0, 10.0, 2001)
    amp = 1.0
    t0 = 5.0
    sigma = 1.0

    dg = d_gaussian_dt(ts, amp=amp, t0=t0, sigma=sigma)
    # closest index to t0
    idx0 = int(np.argmin(np.abs(ts - t0)))

    # derivative around center should be ~0
    assert abs(dg[idx0]) < 1e-6

def test_drag_pulse_shapes_and_relationship():
    ts = np.linspace(0.0, 10.0, 1001)
    amp = 0.5
    t0 = 5.0
    sigma = 1.0
    alpha = -0.8

    ox, oy = drag_pulse(ts, amp=amp, t0=t0, sigma=sigma, alpha=alpha)

    assert ox.shape == ts.shape
    assert oy.shape == ts.shape

    # DRAG: oy should be alpha * derivative(ox) exactly (up to numerical precision)
    dg = d_gaussian_dt(ts, amp=amp, t0=t0, sigma=sigma)
    assert np.allclose(oy, alpha * dg, rtol=1e-10, atol=1e-12)

def test_drag_quadrature_integral_is_close_to_zero():
    """
    For a symmetric Gaussian derivative, integral over a symmetric window should be ~0.
    Not exact due to finite window, but should be small.
    """
    ts = np.linspace(0.0, 10.0, 4001)
    amp = 1.0
    t0 = 5.0
    sigma = 1.0
    alpha = 1.0

    _, oy = drag_pulse(ts, amp=amp, t0=t0, sigma=sigma, alpha=alpha)

    integral = np.trapezoid(oy, ts)
    assert abs(integral) < 1e-4
