# Quantum Control Systems Simulator (QCSim)

A mini **quantum hardware control simulator** inspired by real quantum control stacks.

This project simulates a driven qubit under shaped microwave pulses and realistic decoherence, and includes small experiments and calibration workflows similar to what is used in real labs.

---

## 🚀 Features

### Pulse shaping
- Gaussian envelope pulses
- DRAG pulses (Gaussian + derivative quadrature)

### Noise + decoherence
- T1 relaxation
- T2 dephasing
- Lindblad master equation simulation

### Experiments
- Rabi experiment (population vs pulse amplitude)
- T1 decay experiment (|1⟩ → |0⟩)
- T2 / dephasing experiment (Ramsey-style free evolution)
- DRAG calibration sweep (find best alpha)

### Optimization
- Parameter optimization to maximize fidelity of an Xπ/2 gate (optional extension)

---

## 🧠 Physical Model

### Driven qubit Hamiltonian (rotating frame)

\[
H(t) = \frac{1}{2}\left(\Delta \sigma_z + \Omega_x(t)\sigma_x + \Omega_y(t)\sigma_y\right)
\]

Where:
- Δ is detuning
- Ωx(t), Ωy(t) are the control envelopes

### Decoherence (Lindblad master equation)

\[
\dot{\rho} = -i[H(t), \rho] + \sum_k \left(L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\}\right)
\]

Noise channels:
- Relaxation: \(T_1\)
- Dephasing: \(T_2\)

---

## 📦 Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .


Dependencies:
Python 3.12+
NumPy
SciPy
Matplotlib
Pytest


Running Experiments
Run T1 + T2 experiments
python -m qcsim.experiments.t1_t2

Run DRAG calibration sweep
python -m qcsim.experiments.drag_calibration

(Optional) Rabi experiment
python -m qcsim.experiments.rabi

🧪 Testing

Run all tests:
pytest -q


The test suite includes:

pulse shape validation (Gaussian + DRAG)

unitary-limit physics sanity checks (trace preservation, Hermiticity)

probability validity checks