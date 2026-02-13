import numpy as np

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

# Ladder operators
sp = np.array([[0, 1], [0, 0]], dtype=complex)
sm = np.array([[0, 0], [1, 0]], dtype=complex)

I2 = np.eye(2, dtype=complex)
