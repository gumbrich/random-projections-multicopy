"""Minimal utilities for the thermal Ising example in Fig. 2.

The public parameter q_keep denotes the number of retained coherent qubits.
"""

import itertools
import numpy as np
from scipy.linalg import expm


def transverse_field_ising_hamiltonian(n, J=1.0, h=1.0):
    """Open-boundary transverse-field Ising Hamiltonian.

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    eye = np.eye(2, dtype=complex)

    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(n - 1):
        ops = [sz if j in (i, i + 1) else eye for j in range(n)]
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H -= J * term

    for i in range(n):
        ops = [sx if j == i else eye for j in range(n)]
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H -= h * term

    return H


def thermal_state(beta, H):
    """Thermal state rho = exp(-beta H) / tr(exp(-beta H))."""
    rho = expm(-beta * H)
    return rho / np.trace(rho)


def haar_isometry(d, m, rng):
    """Sample a Haar-random d x m isometry."""
    X = rng.normal(size=(d, m)) + 1j * rng.normal(size=(d, m))
    Q, R = np.linalg.qr(X)
    phases = np.diag(R)
    phases = phases / np.abs(phases)
    Q = Q * phases.conj()
    return Q[:, :m]


def projected_moment_for_isometry(rho, Q, K):
    """Return tr(Q^dag rho Q) and tr((Q^dag rho Q / t)^K)."""
    sigma = Q.conj().T @ rho @ Q
    sigma = 0.5 * (sigma + sigma.conj().T)
    t = float(np.real(np.trace(sigma)))

    if t <= 0:
        return 0.0, 0.0

    sigma_norm = sigma / t
    moment_norm = float(np.real(np.trace(np.linalg.matrix_power(sigma_norm, K))))
    moment_norm = float(np.clip(moment_norm, -1.0, 1.0))
    return t, moment_norm


def compose(p, q):
    return tuple(p[i] for i in q)


def inverse_perm(p):
    inv = [None] * len(p)
    for i, pi in enumerate(p):
        inv[pi] = i
    return tuple(inv)


def permutations(K):
    return list(itertools.permutations(range(K)))


def cycle_lengths(p):
    K = len(p)
    seen = [False] * K
    lengths = []

    for i in range(K):
        if seen[i]:
            continue
        j = i
        length = 0
        while not seen[j]:
            seen[j] = True
            length += 1
            j = p[j]
        lengths.append(length)

    return tuple(sorted(lengths, reverse=True))


def num_cycles(p):
    return len(cycle_lengths(p))


def cycle_trace_product(perm, moments):
    return np.prod([moments[l] for l in cycle_lengths(perm)])


def twirl_coefficients_gram(K, m, d, rcond=1e-12):
    perms = permutations(K)
    nperm = len(perms)
    G = np.zeros((nperm, nperm), dtype=float)
    b = np.zeros(nperm, dtype=float)

    for i, pi in enumerate(perms):
        inv_pi = inverse_perm(pi)
        b[i] = m ** num_cycles(pi)

        for j, tau in enumerate(perms):
            rel = compose(inv_pi, tau)
            G[i, j] = d ** num_cycles(rel)

    coeff = np.linalg.pinv(G, rcond=rcond) @ b
    return dict(zip(perms, coeff))


def true_exp_value_from_moments(K, m, d, moments):
    coeffs = twirl_coefficients_gram(K, m, d)
    pi_cycle = tuple(list(range(1, K)) + [0])

    total = 0.0
    for tau, c_tau in coeffs.items():
        sigma = compose(tau, pi_cycle)
        total += c_tau * cycle_trace_product(sigma, moments)

    return float(np.real_if_close(total))


def integer_partitions_parts_at_least_2(total, min_part=2):
    if total == 0:
        yield ()
        return

    for first in range(min_part, total + 1):
        for rest in integer_partitions_parts_at_least_2(total - first, first):
            yield (first,) + rest


def moment_monomials(K):
    monos = [()]

    for weight in range(2, K + 1):
        parts = list(integer_partitions_parts_at_least_2(weight))
        for mu in parts:
            mu = tuple(sorted(mu))
            if mu != (weight,):
                monos.append(mu)
        monos.append((weight,))

    unique = []
    seen = set()
    for mu in monos:
        if sum(mu) <= K and mu not in seen:
            unique.append(mu)
            seen.add(mu)

    return unique


def monomial_value(mu, moments):
    val = 1.0
    for r in mu:
        val *= moments[r]
    return val


def get_recovery_coefficients(max_K, m, d):
    """Coefficients for recovering p_2, ..., p_max_K from projected moments."""
    coeffs = {}

    for K in range(2, max_K + 1):
        monos = moment_monomials(K)
        nmono = len(monos)
        A = np.zeros((nmono, nmono), dtype=float)
        y = np.zeros(nmono, dtype=float)

        rng = np.random.default_rng(12345 + K)
        probe_moments = []

        mom0 = {1: 1.0}
        for r in range(2, K + 1):
            mom0[r] = 0.0
        probe_moments.append(mom0)

        while len(probe_moments) < nmono:
            mom = {1: 1.0}
            for r in range(2, K + 1):
                mom[r] = float(rng.uniform(-0.8, 0.9))
            probe_moments.append(mom)

        for i, moments in enumerate(probe_moments):
            y[i] = true_exp_value_from_moments(K, m, d, moments)
            for j, mu in enumerate(monos):
                A[i, j] = monomial_value(mu, moments)

        sol = np.linalg.solve(A, y)
        coeffs[K] = {mu: float(cj) for mu, cj in zip(monos, sol)}

    return coeffs


def recover_moment_from_E(K_target, E_K, recovered, coeffs_K):
    numerator = E_K

    for mu, coeff in coeffs_K.items():
        if mu == (K_target,):
            continue
        if K_target in mu:
            raise ValueError(f"Unexpected nonlinear dependence on p_{K_target}: {mu}")
        numerator -= coeff * monomial_value(mu, recovered)

    return numerator / coeffs_K[(K_target,)]


def recover_all_moments_from_E(E, coeffs, K_max):
    recovered = {1: 1.0}

    for K in range(2, K_max + 1):
        recovered[K] = recover_moment_from_E(K, E[K], recovered, coeffs[K])

    return recovered
