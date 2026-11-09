"""Exact fidelity kernel of Qiskit's ZZ feature map, computed with numpy (no qiskit).

The circuit is that of ``qiskit.circuit.library.zz_feature_map`` (and
``qsvm_baseline.build_feature_map`` with one qubit per feature): per repetition, H on
every qubit, P(2 x_i) on qubit i, then for every entangled pair (i, j)
CX(i, j) P(2 (pi - x_i)(pi - x_j)) CX(i, j). The phase layer is diagonal in the
computational basis: a basis state b gets the phase

    sum_i 2 x_i b_i + sum_(i,j) 2 (pi - x_i)(pi - x_j) (b_i XOR b_j),

so |phi(x)> = (D(x) H^n)^reps |0...0> and K(x, y) = |<phi(x)|phi(y)>|^2. This is the
statevector kernel of the QSVM scripts, evaluated exactly; it is feasible for the small
qubit counts used here (2^n amplitudes per sample).
"""

from __future__ import annotations

import numpy as np

ENTANGLEMENTS = ("linear", "circular", "full")
MAX_QUBITS = 16


def entanglement_pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    """Qubit pairs of the ZZ layer, in the order of Qiskit's entanglement strings."""
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if entanglement == "circular":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        return ([(n_qubits - 1, 0)] + pairs) if n_qubits > 2 else pairs
    if entanglement == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    raise ValueError(f"Unknown entanglement {entanglement!r}; use one of {ENTANGLEMENTS}.")


def _hadamard_all(states: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply H on every qubit to a batch of statevectors (fast Walsh-Hadamard transform)."""
    out = states.reshape((states.shape[0],) + (2,) * n_qubits).astype(complex)
    for axis in range(1, n_qubits + 1):
        a = np.take(out, 0, axis=axis)
        b = np.take(out, 1, axis=axis)
        out = np.stack([(a + b) / np.sqrt(2.0), (a - b) / np.sqrt(2.0)], axis=axis)
    return out.reshape(states.shape[0], -1)


def zz_states(X: np.ndarray, reps: int = 1, entanglement: str = "linear") -> np.ndarray:
    """Statevectors |phi(x)> of the ZZ feature map, one row per sample (qubit i = feature i)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n_samples, n_qubits = X.shape
    if not 1 <= n_qubits <= MAX_QUBITS:
        raise ValueError(f"zz_states supports 1..{MAX_QUBITS} qubits, got {n_qubits}.")
    if reps < 1:
        raise ValueError("reps must be at least 1.")
    dim = 2 ** n_qubits
    # bits[k, i] = value of qubit i in basis state k (qubit 0 is the least significant bit).
    bits = (np.arange(dim)[:, None] >> np.arange(n_qubits)[None, :]) & 1
    phase = 2.0 * X @ bits.T
    for i, j in entanglement_pairs(n_qubits, entanglement):
        parity = bits[:, i] ^ bits[:, j]
        phase = phase + np.outer(2.0 * (np.pi - X[:, i]) * (np.pi - X[:, j]), parity)
    diagonal = np.exp(1j * phase)
    states = np.zeros((n_samples, dim), dtype=complex)
    states[:, 0] = 1.0
    for _ in range(reps):
        states = diagonal * _hadamard_all(states, n_qubits)
    return states


def zz_kernel(A: np.ndarray, B: np.ndarray | None = None, reps: int = 1,
              entanglement: str = "linear") -> np.ndarray:
    """K[a, b] = |<phi(A_a)|phi(B_b)>|^2; B=None gives the symmetric training matrix."""
    states_a = zz_states(A, reps, entanglement)
    states_b = states_a if B is None else zz_states(B, reps, entanglement)
    return np.abs(states_a.conj() @ states_b.T) ** 2


def zz_kernel_factory(reps: int = 1, entanglement: str = "linear"):
    """Kernel function K(A, B=None) with fixed reps and entanglement."""
    if entanglement not in ENTANGLEMENTS:
        raise ValueError(f"Unknown entanglement {entanglement!r}; use one of {ENTANGLEMENTS}.")

    def kernel(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        return zz_kernel(A, B, reps=reps, entanglement=entanglement)

    return kernel
