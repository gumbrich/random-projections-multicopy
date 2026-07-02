import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm

from tqdm import tqdm


# ============================================================
# States and true moments
# ============================================================

def noisy_ghz_state(n, p=0.3):
    d = 2**n
    ghz = np.zeros(d, dtype=complex)
    ghz[0] = 1 / np.sqrt(2)
    ghz[-1] = 1 / np.sqrt(2)
    pure = np.outer(ghz, ghz.conj())
    return (1 - p) * pure + p * np.eye(d) / d


def kron_n(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def thermal_tfim_state(n, beta=1.5, J=1.0, h=1.0, periodic=True):
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    d = 2**n
    H = np.zeros((d, d), dtype=complex)

    num_bonds = n if periodic else n - 1

    for i in range(num_bonds):
        j = (i + 1) % n
        ops = [Z if site in (i, j) else I for site in range(n)]
        H += -J * kron_n(ops)

    for i in range(n):
        ops = [X if site == i else I for site in range(n)]
        H += -h * kron_n(ops)

    expH = expm(-beta * H)
    return expH / np.trace(expH)


def moment(rho, k):
    return float(np.real_if_close(np.trace(np.linalg.matrix_power(rho, k))).real)


def true_moment_dict(state_name, n, K_list):
    if state_name == "noisy_GHZ_p0.3":
        rho = noisy_ghz_state(n, p=0.3)
    elif state_name == "thermal_beta1.5":
        rho = thermal_tfim_state(n, beta=1.5)
    else:
        raise ValueError(f"Unknown state_name: {state_name}")

    out = {1: 1.0}
    for K in K_list:
        out[int(K)] = moment(rho, int(K))
    return out


# ============================================================
# Moment reconstruction from projected moments
# ============================================================

def all_perms(K):
    return list(itertools.permutations(range(K)))


def compose(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


def inverse_perm(p):
    inv = [0] * len(p)
    for i, x in enumerate(p):
        inv[x] = i
    return tuple(inv)


def num_cycles(p):
    K = len(p)
    seen = [False] * K
    count = 0
    for i in range(K):
        if not seen[i]:
            count += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = p[j]
    return count


def cycle_lengths(p):
    K = len(p)
    seen = [False] * K
    lengths = []
    for i in range(K):
        if not seen[i]:
            length = 0
            j = i
            while not seen[j]:
                seen[j] = True
                length += 1
                j = p[j]
            lengths.append(length)
    return lengths


def K_cycle_perm(K):
    return tuple((i + 1) % K for i in range(K))


_COEFF_CACHE = {}
_GAMMA_CACHE = {}


def coeffs_general(K, d, m):
    K = int(K); d = int(d); m = int(m)
    perms = all_perms(K)
    Lp = len(perms)
    G = np.zeros((Lp, Lp), dtype=float)
    b = np.zeros(Lp, dtype=float)

    for i, pi in enumerate(perms):
        pi_inv = inverse_perm(pi)
        b[i] = m ** num_cycles(pi)
        for j, tau in enumerate(perms):
            rel = compose(pi_inv, tau)
            G[i, j] = d ** num_cycles(rel)

    c = np.linalg.pinv(G, rcond=1e-12) @ b
    return perms, c


def get_coeffs(K, d, m):
    key = (int(K), int(d), int(m))
    if key not in _COEFF_CACHE:
        _COEFF_CACHE[key] = coeffs_general(*key)
    return _COEFF_CACHE[key]


def moment_product_from_perm(perm, moments):
    prod = 1.0
    for ell in cycle_lengths(perm):
        prod *= moments[ell]
    return prod


def projected_mu_K(K, d, m, moments):
    perms, coeffs = get_coeffs(K, d, m)
    pi = K_cycle_perm(K)
    total = 0.0
    for tau, c_tau in zip(perms, coeffs):
        sigma = compose(tau, pi)
        total += c_tau * moment_product_from_perm(sigma, moments)
    return float(np.real_if_close(total).real)


def gamma_K(K, d, m):
    key = (int(K), int(d), int(m))
    if key in _GAMMA_CACHE:
        return _GAMMA_CACHE[key]
    perms, coeffs = get_coeffs(K, d, m)
    pi = K_cycle_perm(K)
    gam = 0.0
    for tau, c_tau in zip(perms, coeffs):
        sigma = compose(tau, pi)
        if num_cycles(sigma) == 1:
            gam += c_tau
    gam = float(np.real_if_close(gam).real)
    _GAMMA_CACHE[key] = gam
    return gam


def recover_pK_from_mu(mu_hat, K, d, m, known_lower_moments):
    moments_tmp = dict(known_lower_moments)
    moments_tmp[int(K)] = 0.0
    F = projected_mu_K(K, d, m, moments_tmp)
    gam = gamma_K(K, d, m)
    if abs(gam) < 1e-15:
        return np.nan
    return (mu_hat - F) / gam


# ============================================================
# File/data helpers, q convention, q=5 special case
# ============================================================

def find_file_for_q_case(data_dir, q, case, state_hint="noisy_ghz"):
    """
    Use only:
      - q=1,...,4: files containing NU100000_NM100
      - q=5: special fully coherent file *_q5_data.npz

    Ignore NU1_vary_NM files completely.

    The cases NM1_vary_NU and NM100_vary_NU both use the same
    NU100000_NM100 source file.
    """
    data_dir = Path(data_dir)
    q = int(q)

    if q == 0:
        raise ValueError("q=0 is intentionally ignored.")

    if q == 5:
        candidates = sorted(data_dir.glob(f"*q{q}_data.npz"))
    else:
        candidates = sorted(data_dir.glob(f"*q{q}_NU100000_NM100_data.npz"))

    if state_hint is not None:
        candidates = [p for p in candidates if state_hint in p.name]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No file found for q={q}, case={case}, state_hint={state_hint} in {data_dir}"
        )

    if len(candidates) > 1:
        print(f"Multiple candidates for q={q}, case={case}; using first:")
        for p in candidates:
            print("  ", p.name)

    return candidates[0]


def get_num_saved(data, K_list):
    if "num_saved" in data:
        return int(data["num_saved"])
    if "U_index" in data:
        return len(data["U_index"])
    K0 = int(K_list[0])
    return len(data[f"K{K0}_coincide_success"])


def infer_d_m_q(data, fallback_q=None, fallback_n=5):
    if "n" in data:
        n = int(data["n"])
    else:
        n = int(fallback_n)
    d = 2**n

    if "L" in data:
        L = int(data["L"])
        if L <= 0:
            raise ValueError(f"Invalid L={L}")
        m = d // L
        q = int(round(np.log2(m)))
    else:
        if fallback_q is None:
            raise ValueError("Need fallback_q if L is missing.")
        q = int(fallback_q)
        m = 2**q
    return d, m, q


def is_fully_coherent_data(data, fallback_q=None):
    d, m, q = infer_d_m_q(data, fallback_q=fallback_q)
    return m == d


def get_block_structure(data, K_list):
    num_saved = get_num_saved(data, K_list)
    if "U_index" in data:
        U_index = np.asarray(data["U_index"][:num_saved], dtype=int)
    else:
        U_index = np.zeros(num_saved, dtype=int)

    unique_U, first_indices, block_counts = np.unique(
        U_index,
        return_index=True,
        return_counts=True,
    )
    order = np.argsort(first_indices)
    return unique_U[order], first_indices[order], block_counts[order]


# ============================================================
# Coincide estimator only
# ============================================================

def estimate_mu_coincide(data, K, idx, K_list):
    num_saved = get_num_saved(data, K_list)
    success = np.asarray(data[f"K{K}_coincide_success"][:num_saved], dtype=float)
    swap0 = np.asarray(data[f"K{K}_coincide_swap0"][:num_saved], dtype=float)
    L = int(data["L"]) if "L" in data else 1
    s = float(np.sum(success[idx]))
    z = float(np.sum(swap0[idx]))
    N_eff = len(idx)
    return (2.0 * z - s) / N_eff / L


def estimate_all_K_from_idx(data, idx, K_list, true_moments, lower_mode, d, m):
    K_order = sorted([int(k) for k in K_list])
    mu_hat = {K: estimate_mu_coincide(data, K, idx, K_order) for K in K_order}
    estimates = {}

    if lower_mode == "exact":
        for K in K_order:
            known_lower = {j: true_moments[j] for j in range(1, K)}
            estimates[K] = recover_pK_from_mu(mu_hat[K], K, d, m, known_lower)
    elif lower_mode == "hierarchical":
        known = {1: 1.0}
        for K in K_order:
            estimates[K] = recover_pK_from_mu(mu_hat[K], K, d, m, known)
            known[K] = estimates[K]
    else:
        raise ValueError("lower_mode must be 'exact' or 'hierarchical'.")
    return estimates


# ============================================================
# Full-data error diagnostics
# ============================================================

def full_data_errors_one_file(data_path, q, K_list, true_moments, lower_modes=("exact", "hierarchical")):
    data = np.load(data_path, allow_pickle=True)
    d, m, q_inferred = infer_d_m_q(data, fallback_q=q)
    num_saved = get_num_saved(data, K_list)
    idx_all = np.arange(num_saved, dtype=int)

    rows = []
    for lower_mode in lower_modes:
        estimates = estimate_all_K_from_idx(
            data=data,
            idx=idx_all,
            K_list=K_list,
            true_moments=true_moments,
            lower_mode=lower_mode,
            d=d,
            m=m,
        )
        for K in K_list:
            K = int(K)
            abs_error = abs(estimates[K] - true_moments[K])
            rows.append({
                "q": int(q),
                "q_inferred": int(q_inferred),
                "K": K,
                "lower_mode": lower_mode,
                "full_data_estimate": float(estimates[K]),
                "true_value": float(true_moments[K]),
                "full_data_abs_error": float(abs_error),
                "num_saved": int(num_saved),
                "file": Path(data_path).name,
            })
    return pd.DataFrame(rows)


def compute_full_data_error_table(
    data_dir,
    q_values,
    K_list,
    true_moments,
    cases,
    lower_modes=("exact", "hierarchical"),
    state_hint="noisy_ghz",
):
    rows = []
    for case in cases:
        for q in q_values:
            q = int(q)
            if q == 0:
                continue
            path = find_file_for_q_case(data_dir, q, case, state_hint=state_hint)
            df = full_data_errors_one_file(path, q, K_list, true_moments, lower_modes=lower_modes)
            df["case"] = case
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def make_effective_epsilon_dict(full_error_table, base_epsilon=0.1, safety_factor=1.05):
    eps = {}
    for _, row in full_error_table.iterrows():
        key = (int(row["q"]), row["case"], row["lower_mode"], int(row["K"]))
        eps[key] = max(float(base_epsilon), float(safety_factor) * float(row["full_data_abs_error"]))
    return eps


# ============================================================
# Candidate grids and sampling
# ============================================================

def make_log_grid(min_val, max_val, num_grid):
    min_val = int(min_val)
    max_val = int(max_val)
    if max_val < min_val:
        return []
    if min_val < 1:
        min_val = 1
    if min_val == max_val:
        return [min_val]
    grid = np.unique(
        np.round(np.logspace(np.log10(min_val), np.log10(max_val), int(num_grid))).astype(int)
    )
    grid = grid[(grid >= min_val) & (grid <= max_val)]
    return list(grid)


def make_candidate_grid(data, case, K_list, num_grid):
    """
    Candidate grid of (N_U, N_M).

    For fully coherent q=5, rows are treated as i.i.d. protocol executions.
    We represent a candidate N_eff as (N_eff, 1), so N_eff=N_U*N_M.
    """
    num_saved = get_num_saved(data, K_list)

    if is_fully_coherent_data(data):
        N_grid = make_log_grid(10, num_saved, num_grid)
        return [(N, 1) for N in N_grid]

    _, _, block_counts = get_block_structure(data, K_list)
    max_NM = int(np.max(block_counts))

    if case == "NM1_vary_NU":
        N_M = 1
        max_NU = int(np.sum(block_counts >= N_M))
        return [(N_U, N_M) for N_U in make_log_grid(10, max_NU, num_grid)]

    if case == "NM100_vary_NU":
        N_M = 100
        max_NU = int(np.sum(block_counts >= N_M))
        return [(N_U, N_M) for N_U in make_log_grid(10, max_NU, num_grid)]

    if case == "NU1_vary_NM":
        N_U = 1
        return [(N_U, N_M) for N_M in make_log_grid(10, max_NM, num_grid)]

    raise ValueError(f"Unknown case: {case}")


def sample_indices_for_candidate(data, K_list, N_U_boot, N_M_boot, rng):
    num_saved = get_num_saved(data, K_list)

    if is_fully_coherent_data(data):
        N_eff = int(N_U_boot * N_M_boot)
        if N_eff > num_saved:
            raise ValueError(f"N_eff={N_eff} exceeds num_saved={num_saved}.")
        return rng.choice(num_saved, size=N_eff, replace=False)

    _, first_indices, block_counts = get_block_structure(data, K_list)
    N_U_boot = int(N_U_boot)
    N_M_boot = int(N_M_boot)
    usable = np.where(block_counts >= N_M_boot)[0]
    if len(usable) == 0:
        raise ValueError(f"No U block has at least N_M_boot={N_M_boot} rows.")
    if N_U_boot > len(usable):
        raise ValueError(f"N_U_boot={N_U_boot} exceeds available U blocks={len(usable)}.")

    chosen = rng.choice(usable, size=N_U_boot, replace=False)
    idx_parts = []
    for pos in chosen:
        start = first_indices[pos]
        count = block_counts[pos]
        offsets = rng.choice(count, size=N_M_boot, replace=False)
        idx_parts.append(start + offsets)
    return np.concatenate(idx_parts)


# ============================================================
# First crossing and total copy accounting
# ============================================================

def first_crossing_for_one_bootstrap(
    data,
    grid,
    K_list,
    true_moments,
    lower_mode,
    epsilon_dict,
    q,
    case,
    d,
    m,
    rng,
):
    """
    For one bootstrap repetition.

    exact:
        Find first crossing separately for each target K.

    hierarchical:
        For target K, find the first N such that all moments
        r=2,...,K are simultaneously within threshold.

        This gives one common N_req^{<=K}.
    """
    K_order = sorted(K_list)

    found = {
        K: {
            "found": False,
            "used_fallback": False,
            "N_req": np.nan,
            "N_U_req": np.nan,
            "N_M_req": np.nan,
            "abs_error_at_crossing": np.nan,
            "epsilon_eff": np.nan,
        }
        for K in K_order
    }

    for N_U_boot, N_M_boot in grid:
        idx = sample_indices_for_candidate(
            data=data,
            K_list=K_list,
            N_U_boot=N_U_boot,
            N_M_boot=N_M_boot,
            rng=rng,
        )

        estimates = estimate_all_K_from_idx(
            data=data,
            idx=idx,
            K_list=K_list,
            true_moments=true_moments,
            lower_mode=lower_mode,
            d=d,
            m=m,
        )

        abs_err = {
            K: abs(estimates[K] - true_moments[K])
            for K in K_order
        }

        if lower_mode == "exact":
            # Exact-lower-moment mode:
            # target K only needs p_K to be accurate.
            for K in K_order:
                if found[K]["found"]:
                    continue

                key = (int(q), case, lower_mode, int(K))
                epsilon_eff = float(epsilon_dict[key])

                if abs_err[K] <= epsilon_eff:
                    found[K]["found"] = True
                    found[K]["N_req"] = int(N_U_boot * N_M_boot)
                    found[K]["N_U_req"] = int(N_U_boot)
                    found[K]["N_M_req"] = int(N_M_boot)
                    found[K]["abs_error_at_crossing"] = float(abs_err[K])
                    found[K]["epsilon_eff"] = float(epsilon_eff)

        elif lower_mode == "hierarchical":
            # Hierarchical mode:
            # target K requires all r=2,...,K accurate at the same N.
            for K in K_order:
                if found[K]["found"]:
                    continue

                ok = True
                eps_used = []

                for r in K_order:
                    if r > K:
                        break

                    key = (int(q), case, lower_mode, int(r))
                    epsilon_eff_r = float(epsilon_dict[key])
                    eps_used.append(epsilon_eff_r)

                    if abs_err[r] > epsilon_eff_r:
                        ok = False
                        break

                if ok:
                    found[K]["found"] = True
                    found[K]["N_req"] = int(N_U_boot * N_M_boot)
                    found[K]["N_U_req"] = int(N_U_boot)
                    found[K]["N_M_req"] = int(N_M_boot)
                    found[K]["abs_error_at_crossing"] = float(
                        max(abs_err[r] for r in K_order if r <= K)
                    )
                    found[K]["epsilon_eff"] = float(max(eps_used))

        else:
            raise ValueError("lower_mode must be 'exact' or 'hierarchical'.")

        if all(found[K]["found"] for K in K_order):
            break

    # Fallback: if no crossing, use the whole available data.
    missing_K = [K for K in K_order if not found[K]["found"]]

    if len(missing_K) > 0:
        idx_all = np.arange(get_num_saved(data, K_list), dtype=int)

        estimates_all = estimate_all_K_from_idx(
            data=data,
            idx=idx_all,
            K_list=K_list,
            true_moments=true_moments,
            lower_mode=lower_mode,
            d=d,
            m=m,
        )

        abs_err_all = {
            K: abs(estimates_all[K] - true_moments[K])
            for K in K_order
        }

        num_saved = get_num_saved(data, K_list)

        if is_fully_coherent_data(data, fallback_q=q):
            fallback_N_U = num_saved
            fallback_N_M = 1
            fallback_N_eff = num_saved
        else:
            _, _, block_counts = get_block_structure(data, K_list)
            fallback_N_U = len(block_counts)
            fallback_N_M = int(np.min(block_counts))
            fallback_N_eff = int(num_saved)

        for K in missing_K:
            if lower_mode == "exact":
                key = (int(q), case, lower_mode, int(K))
                epsilon_eff = float(epsilon_dict[key])
                error_to_record = abs_err_all[K]

            else:
                eps_list = []
                err_list = []

                for r in K_order:
                    if r > K:
                        break

                    key = (int(q), case, lower_mode, int(r))
                    eps_list.append(float(epsilon_dict[key]))
                    err_list.append(float(abs_err_all[r]))

                epsilon_eff = max(eps_list)
                error_to_record = max(err_list)

            found[K]["found"] = True
            found[K]["used_fallback"] = True
            found[K]["N_req"] = int(fallback_N_eff)
            found[K]["N_U_req"] = int(fallback_N_U)
            found[K]["N_M_req"] = int(fallback_N_M)
            found[K]["abs_error_at_crossing"] = float(error_to_record)
            found[K]["epsilon_eff"] = float(epsilon_eff)

    return found


def total_copy_cost_from_crossings(found, K_list, lower_mode, q=None, n=None):
    """
    exact:
        cost_K = K * N_req(K)

    hierarchical:
        one common N_req^{<=K} is already stored in found[K]["N_req"],
        so:
            cost_K = (2+3+...+K) * N_req^{<=K}

    q=n:
        fully coherent direct SWAP test, always:
            cost_K = K * N_req(K)
    """
    out = {}
    K_order = sorted(K_list)

    fully_coherent = (
        q is not None
        and n is not None
        and int(q) == int(n)
    )

    for K in K_order:
        N_req = found[K]["N_req"]

        if not np.isfinite(N_req):
            out[K] = np.nan
            continue

        if fully_coherent or lower_mode == "exact":
            out[K] = K * N_req

        elif lower_mode == "hierarchical":
            factor = sum(r for r in K_order if r <= K)
            out[K] = factor * N_req

        else:
            raise ValueError("lower_mode must be 'exact' or 'hierarchical'.")

    return out


# ============================================================
# Main processing
# ============================================================

def process_one_file_first_crossing(
    data_path,
    q,
    case,
    lower_mode,
    K_list,
    true_moments,
    epsilon_dict,
    B=100,
    num_grid=18,
    seed=1234,
    show_progress=True,
):
    rng = np.random.default_rng(seed)
    data = np.load(data_path, allow_pickle=True)
    d, m, q_inferred = infer_d_m_q(data, fallback_q=q)
    if q_inferred != int(q):
        print(f"Warning: input q={q}, inferred q={q_inferred}")

    grid = make_candidate_grid(data, case, K_list, num_grid)
    bootstrap_rows = []

    iterator = range(int(B))
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc=f"q={q}, {case}, {lower_mode}", leave=False)

    for b in iterator:
        found = first_crossing_for_one_bootstrap(
            data=data,
            grid=grid,
            K_list=K_list,
            true_moments=true_moments,
            lower_mode=lower_mode,
            epsilon_dict=epsilon_dict,
            q=q,
            case=case,
            d=d,
            m=m,
            rng=rng,
        )
        n_file = int(np.log2(d))
        total_copy_costs = total_copy_cost_from_crossings(
            found,
            K_list,
            lower_mode,
            q=q,
            n=n_file,
)

        for K in K_list:
            K = int(K)
            bootstrap_rows.append({
                "bootstrap": int(b),
                "q": int(q),
                "K": int(K),
                "case": case,
                "lower_mode": lower_mode,
                "N_req": found[K]["N_req"],
                "N_U_req": found[K]["N_U_req"],
                "N_M_req": found[K]["N_M_req"],
                "total_copy_cost": total_copy_costs[K],
                "abs_error_at_crossing": found[K]["abs_error_at_crossing"],
                "epsilon_eff": found[K]["epsilon_eff"],
                "found": bool(found[K]["found"]),
                "used_fallback": bool(found[K].get("used_fallback", False)),
                "file": Path(data_path).name,
            })

    boot_df = pd.DataFrame(bootstrap_rows)

    summary_rows = []
    for K in K_list:
        K = int(K)
        sub = boot_df[boot_df["K"] == K]
        def safe_mean(col):
            return float(np.nanmean(sub[col])) if np.any(np.isfinite(sub[col])) else np.nan
        def safe_median(col):
            return float(np.nanmedian(sub[col])) if np.any(np.isfinite(sub[col])) else np.nan
        def safe_q(col, qv):
            return float(np.nanquantile(sub[col], qv)) if np.any(np.isfinite(sub[col])) else np.nan
        def safe_geom_mean(col):
            vals = sub[col].to_numpy(float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            return float(np.exp(np.mean(np.log(vals)))) if len(vals) > 0 else np.nan
        
        summary_rows.append({
            "q": int(q),
            "K": int(K),
            "case": case,
            "lower_mode": lower_mode,
            "mean_N_req": float(np.nanmean(sub["N_req"])),
            "median_N_req": float(np.nanmedian(sub["N_req"])),
            "q16_N_req": float(np.nanquantile(sub["N_req"], 0.16)),
            "q84_N_req": float(np.nanquantile(sub["N_req"], 0.84)),
            "mean_total_copy_cost": float(np.nanmean(sub["total_copy_cost"])),
            "median_total_copy_cost": float(np.nanmedian(sub["total_copy_cost"])),
            "geom_mean_total_copy_cost": safe_geom_mean("total_copy_cost"),
            "q16_total_copy_cost": float(np.nanquantile(sub["total_copy_cost"], 0.16)),
            "q84_total_copy_cost": float(np.nanquantile(sub["total_copy_cost"], 0.84)),
            "found_rate": float(np.mean(sub["found"])),
            "fallback_rate": float(np.mean(sub["used_fallback"])),
            "epsilon_eff_mean": float(np.nanmean(sub["epsilon_eff"])),
            "B": int(B),
            "num_grid": int(num_grid),
            "file": Path(data_path).name,
        })
    return pd.DataFrame(summary_rows), boot_df


def process_existing_npz_set_first_crossing(
    data_dir,
    q_values,
    K_list,
    true_moments,
    cases,
    lower_modes,
    state_name="noisy_GHZ_p0.3",
    state_hint="noisy_ghz",
    base_epsilon=0.1,
    epsilon_safety_factor=1.05,
    B=100,
    num_grid=18,
    seed=1234,
    show_progress=True,
):
    q_values = [int(q) for q in q_values if int(q) != 0]

    full_error_table = compute_full_data_error_table(
        data_dir=data_dir,
        q_values=q_values,
        K_list=K_list,
        true_moments=true_moments,
        cases=cases,
        lower_modes=lower_modes,
        state_hint=state_hint,
    )
    epsilon_dict = make_effective_epsilon_dict(
        full_error_table,
        base_epsilon=base_epsilon,
        safety_factor=epsilon_safety_factor,
    )

    tasks = [(q, case, lower_mode) for case in cases for lower_mode in lower_modes for q in q_values]
    outer = tasks
    if show_progress and tqdm is not None:
        outer = tqdm(tasks, desc="Total postprocessing", leave=True)

    all_summary = []
    all_boot = []
    for i, (q, case, lower_mode) in enumerate(outer):
        path = find_file_for_q_case(data_dir, q, case, state_hint=state_hint)
        summary_df, boot_df = process_one_file_first_crossing(
            data_path=path,
            q=q,
            case=case,
            lower_mode=lower_mode,
            K_list=K_list,
            true_moments=true_moments,
            epsilon_dict=epsilon_dict,
            B=B,
            num_grid=num_grid,
            seed=seed + 10000 * i,
            show_progress=show_progress,
        )
        summary_df["state"] = state_name
        boot_df["state"] = state_name
        all_summary.append(summary_df)
        all_boot.append(boot_df)

    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    boot = pd.concat(all_boot, ignore_index=True) if all_boot else pd.DataFrame()
    return summary, boot, full_error_table


# ============================================================
# Plotting
# ============================================================

def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


COLORS = {2: "#4C72B0", 3: "#55A868", 4: "#C44E52", 5: "#8172B2"}
MARKERS = {2: "o", 3: "s", 4: "D", 5: "^"}


def plot_total_copy_complexity(summary, case, lower_mode, output_path=None):
    set_plot_style()
    sub = summary[(summary["case"] == case) & (summary["lower_mode"] == lower_mode)].copy()

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for K in sorted(sub["K"].dropna().unique()):
        K = int(K)
        ss = sub[sub["K"] == K].sort_values("q")
        y = ss["mean_total_copy_cost"].to_numpy(float)
        ylo = ss["q16_total_copy_cost"].to_numpy(float)
        yhi = ss["q84_total_copy_cost"].to_numpy(float)
        yerr = np.vstack([y - ylo, yhi - y])
        ax.errorbar(
            ss["q"],
            y,
            yerr=yerr,
            marker=MARKERS.get(K, "o"),
            lw=2.0,
            ms=6.5,
            mfc="white",
            mew=1.5,
            capsize=3,
            color=COLORS.get(K, None),
            label=rf"$K={K}$",
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"kept qubits $q=\log_2 m$")
    ax.set_ylabel(r"total required copies")
    ax.set_title(f"{case}, {lower_mode}")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return fig, ax


def plot_full_data_errors(full_error_table, case=None, lower_mode=None, output_path=None):
    set_plot_style()
    sub = full_error_table.copy()
    if case is not None:
        sub = sub[sub["case"] == case]
    if lower_mode is not None:
        sub = sub[sub["lower_mode"] == lower_mode]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for K in sorted(sub["K"].dropna().unique()):
        K = int(K)
        ss = sub[sub["K"] == K].sort_values("q")
        ax.plot(
            ss["q"],
            ss["full_data_abs_error"],
            marker=MARKERS.get(K, "o"),
            lw=2,
            ms=6,
            color=COLORS.get(K, None),
            label=rf"$K={K}$",
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"kept qubits $q=\log_2 m$")
    ax.set_ylabel("full-data absolute error")
    title = "Full-data error"
    if case is not None:
        title += f", {case}"
    if lower_mode is not None:
        title += f", {lower_mode}"
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return fig, ax
