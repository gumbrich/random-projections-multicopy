"""Generate the thermal Ising data used for one panel of Fig. 2.

Edit the constants below and run

    python generate_data.py

The script saves a CSV and a pickle file in ./data/.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from ising_protocol import (
    get_recovery_coefficients,
    haar_isometry,
    projected_moment_for_isometry,
    recover_all_moments_from_E,
    thermal_state,
    transverse_field_ising_hamiltonian,
)

# ------------------------- user-facing settings -------------------------
N_QUBITS = 5 # number of qubits in the Ising chain. The paper uses n=5.
J = 1.0 # coupling strength. The paper uses J=1.0.
H_FIELD = 1.0 # transverse field strength. The paper uses h=1.0.
K_TARGET = 3 # target moment. The paper uses K_TARGET=3.

# Choose one panel of Fig. 2.
# The paper uses N_TOTAL = 5_000 and N_TOTAL = 100_000.
N_TOTAL = 5000 # number of copies. It is considered as the total copies used for p2 and p3 each. The total number of state copies is 2*n_exec + 3*n_exec = 5*n_exec.

# Number of retained coherent qubits. Fig. 2 uses q = 1 and q = 3.
Q_KEEP_LIST = (1, 3) # kept qubits. The paper uses q_keep = 1 and q_keep = 3.

# Number of independent repetitions for quantiles. The paper uses B = 500.
B = 100

BETA_GRID = np.linspace(0.0, 3.0, 13) # different beta for the thermal state
SEED = 2026
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
# ------------------------------------------------------------------------


def simulate_one_recovered_p3(rho, q_keep, n_exec, rng, coeffs_cache):
    """One finite-sampling estimate of p3 = tr(rho^3).

    q_keep is the number of retained coherent qubits.
    n_exec is the number of protocol executions used for p2 and p3 each.
    Since p3 is reconstructed hierarchically from p2 and p3, the total number
    of state copies is 2*n_exec + 3*n_exec = 5*n_exec.
    """
    d = rho.shape[0]
    n = int(np.log2(d))

    if not (0 <= q_keep <= n):
        raise ValueError("q_keep must satisfy 0 <= q_keep <= n.")

    E = {2: 0.0, 3: 0.0}

    # Fully coherent swap-test limit.
    if q_keep == n:
        for K in (2, 3):
            pK = float(np.real(np.trace(np.linalg.matrix_power(rho, K))))
            p_swap0 = 0.5 * (1.0 + pK)
            z = rng.binomial(1, p_swap0, size=n_exec)
            E[K] = np.mean(2.0 * z - 1.0)
        return E[3]

    m = 2**q_keep
    key = (3, m, d)
    if key not in coeffs_cache:
        coeffs_cache[key] = get_recovery_coefficients(3, m, d)
    coeffs = coeffs_cache[key]

    for _ in range(n_exec):
        Q = haar_isometry(d, m, rng)

        for K in (2, 3):
            t, moment_norm = projected_moment_for_isometry(rho, Q, K)
            p_success = np.clip(t**K, 0.0, 1.0)
            success = rng.binomial(1, p_success)

            if success == 1:
                p_swap0 = np.clip(0.5 * (1.0 + moment_norm), 0.0, 1.0)
                swap0 = rng.binomial(1, p_swap0)
            else:
                swap0 = 0

            E[K] += 2.0 * swap0 - success

    for K in (2, 3):
        E[K] /= n_exec

    recovered = recover_all_moments_from_E(E, coeffs, 3)
    return recovered[3]


def generate_data():
    rng = np.random.default_rng(SEED)
    coeffs_cache = {}
    rows = []

    if N_TOTAL % 5 != 0:
        raise ValueError("For K_TARGET=3, N_TOTAL should be divisible by 5.")

    n_exec = N_TOTAL // 5
    H = transverse_field_ising_hamiltonian(n=N_QUBITS, J=J, h=H_FIELD)

    for beta in BETA_GRID:
        rho = thermal_state(beta, H)
        exact = float(np.real(np.trace(np.linalg.matrix_power(rho, K_TARGET))))
        print(f"beta={beta:.3f}, exact p{K_TARGET}={exact:.6f}")

        for q_keep in Q_KEEP_LIST:
            estimates = np.empty(B, dtype=float)

            for b in range(B):
                estimates[b] = simulate_one_recovered_p3(
                    rho=rho,
                    q_keep=q_keep,
                    n_exec=n_exec,
                    rng=rng,
                    coeffs_cache=coeffs_cache,
                )

            rows.append(
                {
                    "beta": beta,
                    "q_keep": q_keep,
                    "N_total": N_TOTAL,
                    "N_exec": n_exec,
                    "exact": exact,
                    "mean": float(np.mean(estimates)),
                    "median": float(np.median(estimates)),
                    "q25": float(np.quantile(estimates, 0.25)),
                    "q75": float(np.quantile(estimates, 0.75)),
                    "q10": float(np.quantile(estimates, 0.10)),
                    "q90": float(np.quantile(estimates, 0.90)),
                }
            )

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_data()

    stem = f"ising_fig2_N{N_TOTAL}"
    df.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    df.to_pickle(OUTPUT_DIR / f"{stem}.pkl")
    print(f"Saved {OUTPUT_DIR / (stem + '.csv')}")
    print(f"Saved {OUTPUT_DIR / (stem + '.pkl')}")


if __name__ == "__main__":
    main()
