import numpy as np
from scipy.linalg import expm


def ghz_state(n):

    d = 2 ** n

    psi = np.zeros(d, dtype=complex)

    psi[0] = 1 / np.sqrt(2)
    psi[-1] = 1 / np.sqrt(2)

    return np.outer(psi, psi.conj())


def noisy_ghz_state(n, p):

    rho_ghz = ghz_state(n)

    d = 2 ** n

    return (1 - p) * rho_ghz + p * np.eye(d) / d


def transverse_field_ising_hamiltonian(
    n,
    J=1.0,
    h=1.0,
):
    """
    Open-boundary transverse-field Ising Hamiltonian

        H = -J Σ Z_i Z_{i+1} - h Σ X_i
    """

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    d = 2 ** n

    H = np.zeros((d, d), dtype=complex)

    # ZZ terms
    for i in range(n - 1):

        ops = []

        for j in range(n):

            if j == i or j == i + 1:
                ops.append(sz)
            else:
                ops.append(I)

        term = ops[0]

        for op in ops[1:]:
            term = np.kron(term, op)

        H -= J * term

    # X terms
    for i in range(n):

        ops = []

        for j in range(n):

            if j == i:
                ops.append(sx)
            else:
                ops.append(I)

        term = ops[0]

        for op in ops[1:]:
            term = np.kron(term, op)

        H -= h * term

    return H


def thermal_state(
    beta,
    H,
):
    """
    Thermal state

        rho = exp(-beta H) / Z
    """

    rho = expm(-beta * H)

    Z = np.trace(rho)

    return rho / Z


def thermal_ising_state(
    n,
    beta,
    J=1.0,
    h=1.0,
):
    """
    Convenience wrapper for TFIM thermal states.
    """

    H = transverse_field_ising_hamiltonian(
        n=n,
        J=J,
        h=h,
    )

    return thermal_state(beta, H)