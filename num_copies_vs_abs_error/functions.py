import numpy as np
import matplotlib.pyplot as plt
from math import comb, factorial
import qutip as qt
from collections import Counter
from itertools import combinations, product, permutations
from tqdm import tqdm
import pennylane as qml
import pickle
import os
import math

def state_moment(rho, k):
    """Compute the k-th moment of the state rho."""
    eigvals = np.linalg.eigvals(rho)
    moment = np.sum(eigvals ** k)
    return np.real(moment)

# given a permutation, find the conjugacy class, for example in S_3, output whether it is [1,1,1], [2,1], or [3]
def permutation_cycle_type(perm):
    """
    Return the cycle type of a permutation given in one-line notation
    on {0, ..., n-1}, as produced by itertools.permutations(range(n)).

    Example:
        (0, 1, 2) -> [1, 1, 1]
        (1, 0, 2) -> [2, 1]
        (1, 2, 0) -> [3]
    """
    n = len(perm)
    seen = [False] * n
    cycle_lengths = []

    for i in range(n):
        if not seen[i]:
            length = 0
            j = i
            while not seen[j]:
                seen[j] = True
                j = perm[j]   # no -1 here
                length += 1
            cycle_lengths.append(length)

    cycle_lengths.sort(reverse=True)
    return cycle_lengths

def cycle_trace_product(perm, rho):
    """
    Computes
        prod_{c in cyc(perm)} tr(rho^{|c|})
    where perm is a permutation of (0,...,n-1).
    """
    cycle_lengths = permutation_cycle_type(perm)
    eigs = np.linalg.eigvals(rho)
    # print(np.prod([np.sum(eigs**l) for l in cycle_lengths]))
    return np.prod(tuple(np.sum(eigs**l) for l in cycle_lengths))


def alpha_lambda(partition, x):
    """
    partition is a list of row lengths, e.g.
        [3]
        [2, 1]
        [1, 1, 1]

    computes α_λ(x) = \Pi_{(i,j) in λ} (x + j - i), row i, column j in Young diagram
    """
    value = 1
    for i, row_len in enumerate(partition, start=1):
        for j in range(1, row_len + 1):
            value *= (x + j - i)
    return value


def f_lambda(partition):
    """
    partition is a list like [3], [2,1], [1,1,1], [3,2], ...
    returns f^lambda using the hook-length formula: n!/(\Pi h_{\lambda}(i,j))
    """
    K = sum(partition)
    hook_product = 1

    for i, row_len in enumerate(partition):          # i = 0,1,2,...
        for j in range(row_len):                     # j = 0,1,2,...
            # boxes to the right in same row
            right = row_len - j - 1

            # boxes below in same column
            below = sum(1 for r in partition[i+1:] if r > j)

            hook_length = right + below + 1
            hook_product *= hook_length

    return factorial(K) // hook_product

# Use Murnaghan–Nakayama rule to solve \chi^{\lambda}(\tau) only depends on partition \lambda and (partition of) \tau [not the specific permutation \tau!]
# Not really understand yet

def partition_boxes(partition):
    """
    partition like (4,2,1) -> set of boxes (r,c), 1-indexed
    """
    boxes = set()
    for r, row_len in enumerate(partition, start=1):
        for c in range(1, row_len + 1):
            boxes.add((r, c))
    return boxes


def boxes_to_partition(boxes):
    """
    Convert a Ferrers-diagram box set back to a partition tuple.
    Returns None if boxes do not form a valid partition.
    """
    if not boxes:
        return ()

    max_row = max(r for r, c in boxes)
    row_lengths = []

    for r in range(1, max_row + 1):
        cols = sorted(c for rr, c in boxes if rr == r)
        if not cols:
            row_lengths.append(0)
            continue

        # row must be exactly 1,2,...,k
        if cols != list(range(1, len(cols) + 1)):
            return None
        row_lengths.append(len(cols))

    # remove trailing zero rows
    while row_lengths and row_lengths[-1] == 0:
        row_lengths.pop()

    # row lengths must be weakly decreasing
    for i in range(len(row_lengths) - 1):
        if row_lengths[i] < row_lengths[i + 1]:
            return None

    return tuple(row_lengths)


def is_connected(box_subset):
    """
    Edge-connectedness of boxes in the grid.
    """
    if not box_subset:
        return False

    start = next(iter(box_subset))
    stack = [start]
    seen = {start}

    while stack:
        r, c = stack.pop()
        for nbr in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]:
            if nbr in box_subset and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)

    return seen == box_subset


def has_2x2(box_subset):
    """
    Check whether the box set contains a 2x2 block.
    """
    for r, c in box_subset:
        if ((r+1, c) in box_subset and
            (r, c+1) in box_subset and
            (r+1, c+1) in box_subset):
            return True
    return False


def border_strip_removals(partition, strip_size):
    """
    Generate all ways to remove a border strip of size strip_size from partition.

    Returns pairs:
        (new_partition, height)
    where height = number_of_rows_spanned - 1
    """
    lam_boxes = partition_boxes(partition)
    all_boxes = list(lam_boxes)

    results = []

    for subset in combinations(all_boxes, strip_size):
        subset = set(subset)
        remaining = lam_boxes - subset

        # remaining must still be a partition shape
        new_partition = boxes_to_partition(remaining)
        if new_partition is None:
            continue

        # removed skew shape must be a border strip:
        # connected and no 2x2
        if not is_connected(subset):
            continue
        if has_2x2(subset):
            continue

        rows = [r for r, c in subset]
        height = max(rows) - min(rows)

        results.append((new_partition, height))

    # remove duplicates
    results = list(dict.fromkeys(results))
    return results



def character_from_cycle_type(lambda_partition_tuple, cycle_type_tuple):
    """
    Compute chi^lambda(mu), where
      lambda_partition_tuple is a partition tuple, e.g. (3,2,1)
      cycle_type_tuple is a cycle type tuple, e.g. (3,2,1)

    Uses the Murnaghan-Nakayama rule.
    """
    if sum(lambda_partition_tuple) != sum(cycle_type_tuple):
        return 0

    if not cycle_type_tuple:
        return 1 if not lambda_partition_tuple else 0

    r = cycle_type_tuple[0]
    rest = cycle_type_tuple[1:]

    total = 0
    for new_partition, height in border_strip_removals(lambda_partition_tuple, r):
        total += ((-1) ** height) * character_from_cycle_type(new_partition, rest)

    return total


def chi_lambda_tau(lambda_partition, tau):
    """
    lambda_partition: list, e.g. [2,1]
    tau: permutation tuple, e.g. (1,0,2)

    Returns chi^lambda(tau).
    """
    tau_partition = permutation_cycle_type(tau)
    return character_from_cycle_type(tuple(lambda_partition), tuple(tau_partition))

def find_partitions(k, m, min_part=1):
    """
    Find all partitions of `k` into `m` non-decreasing parts where each part is at least `min_part`.
    i.e., k=k_1+k_2+...+k_m where 1<=k_1<=...<=k_m<=k

    k: number of copies
    m: number of cycles
    min_part=1: minimum number of elements required for each cycle, defuault by 1
    """
    if m == 1:
        if k >= min_part:
            return [[k]]
        else:
            return []
    partitions = []
    for first_part in range(min_part, k - m + 2):
        for subpartition in find_partitions(k - first_part, m - 1, first_part):
            partitions.append([first_part] + subpartition)
    return [sorted(p, reverse=True) for p in partitions]


def find_all_partitions(K):

    """
    provide all the partitions of K
    """

    all_partitions = []
    for m in range(1, K + 1):
        all_partitions.extend(find_partitions(K, m))
    return all_partitions

def compose(tau, pi):
    """
    Return tau ∘ pi, i.e. (tau pi)(i) = tau(pi(i)).

    tau, pi are tuples in one-line notation on {0, ..., K-1}.
    """
    return tuple(tau[pi[i]] for i in range(len(tau)))

def cyclic_pi(K):
    """
    Returns the K-cycle (0 1 2 ... K-1) written in one-line notation:
    i -> i+1 mod K
    """
    return tuple(list(range(1, K)) + [0])

def true_exp_value(K, m, d, rho):

    """
    rhs of the formula
    """

    result = 0
    for tau in permutations(range(K)):
        for lambda_partition in find_all_partitions(K):
            chi = chi_lambda_tau(lambda_partition, tau)
            f = f_lambda(lambda_partition)
            a_lambda_d = alpha_lambda(lambda_partition, d)
            a_lambda_m = alpha_lambda(lambda_partition, m)
            perm_tau_pi = compose(tau, cyclic_pi(K))
            state_power_tau_pi = cycle_trace_product(perm_tau_pi, rho)
            result += f * a_lambda_m / a_lambda_d * chi * state_power_tau_pi
    result /= factorial(K)
    return np.real(result)




def sample_tr_sigma_U_power(rho, K, d, m, U_samples):

    """
    sample tr(sigma_U^K) where sigma_U = P U rho U^dagger P, P is a projector of rank m, U is Haar random unitary
    """
    P = np.zeros((d, d))
    P[:m, :m] = np.eye(m) # suppose computational basis projector sum_{i=0}^{m-1} |i><i|
    result = 0
    for _ in tqdm(range(U_samples), desc="Processing random-U case", mininterval=1000, leave=True):
        U = qt.rand_unitary(d).full()
        PU = U @ P @ U.conj().T
        sigma_U = PU @ rho @ PU
        result += state_moment(sigma_U, K)
    result /= U_samples
    return np.real(result)



def random_single_qubit_unitary():
    return qt.rand_unitary(2).full()

def local_brickwork_template(one_qubit_unitaries, depth, n):
    for layer in range(depth):
        for q in range(n):
            qml.QubitUnitary(one_qubit_unitaries[layer][q], wires=q)
        start = layer % 2
        for q in range(start, n - 1, 2):
            qml.CZ(wires=[q, q + 1])

def local_brickwork_unitary_pennylane(n, depth, one_qubit_unitaries=None):

    """
    Generate the unitary matrix with local brickwork structure, with optional one-qubit unitaries. 
    If one_qubit_unitaries is None, random single-qubit unitaries will be generated for each layer and qubit.
    """

    if str(depth) == "inf":
        return qt.rand_unitary(2**n).full()

    depth = int(depth)

    if one_qubit_unitaries is None:
        one_qubit_unitaries = [[random_single_qubit_unitary() for _ in range(n)] for _ in range(depth)]

    def circuit():
        local_brickwork_template(one_qubit_unitaries, depth, n)

    U = qml.matrix(circuit, wire_order=list(range(n)))()
    return U

def outcome_mask(outcome, measured_registers, n):
    measured_registers = list(measured_registers)
    outcome_map = {q: int(outcome[j]) for j, q in enumerate(measured_registers)}

    mask = np.zeros(2**n, dtype=bool)
    for x in range(2**n):
        bits = format(x, f"0{n}b")
        ok = True
        for q, b in outcome_map.items():
            if int(bits[q]) != b:
                ok = False
                break
        mask[x] = ok
    return mask

def partial_trace_qutip(rho, keep_wires, n):
    rho_qobj = qt.Qobj(rho, dims=[[2] * n, [2] * n])
    rho_red = rho_qobj.ptrace(keep_wires)
    return np.array(rho_red.full(), dtype=complex)

def local_postselected_states_numpy(rho, measured_registers, U):

    """
    Output the corresponding data for each postselection outcome, including the probability of the outcome, the remaining wires, and the remaining state on those wires after postselection.
    """

    rho = np.asarray(rho, dtype=complex)
    n = int(np.log2(rho.shape[0]))
    if rho.shape != (2**n, 2**n):
        raise ValueError("rho must be a 2^n x 2^n density matrix.")

    measured_registers = sorted(measured_registers)
    unmeasured_registers = [q for q in range(n) if q not in measured_registers]
    l = len(measured_registers)

    rho_u = U @ rho @ U.conj().T
    results = {}

    for y in range(2**l):
        outcome = format(y, f"0{l}b")
        mask = outcome_mask(outcome, measured_registers, n)

        sigma_y = np.zeros_like(rho_u, dtype=complex)
        sigma_y[np.ix_(mask, mask)] = rho_u[np.ix_(mask, mask)]

        p_y = np.real_if_close(np.trace(sigma_y)).item()

        if p_y > 1e-15:
            rho_remain = partial_trace_qutip(sigma_y / p_y, unmeasured_registers, n)
        else:
            dim = 2 ** len(unmeasured_registers)
            rho_remain = np.zeros((dim, dim), dtype=complex)

        results[outcome] = {
            "probability": p_y,
            "remaining_wires": unmeasured_registers,
            "remaining_state": rho_remain,
        }

    return results


def estimate_tr_sigma_U_power(rho, depth, measured_registers, K, N_tot, branch, shots_per_U):

    n = np.log2(rho.shape[0]).astype(int)

    succeed_coincide = 0
    swap_test_coincide_0 = 0

    succeed_branch = 0
    swap_test_branch_0 = 0

    l = len(measured_registers)
    L = 2**l
    m = 2**(n-l)


    for _ in tqdm(range(N_tot), desc="Processing random-U case", mininterval=1000, leave=True):

        U = local_brickwork_unitary_pennylane(n, depth)
        results = local_postselected_states_numpy(rho, measured_registers, U)

        branches = list(results.keys())
        probs = [results[b]["probability"] for b in branches]

        for _ in range(shots_per_U):

            outcome_info = np.random.choice(branches, size=K, p=probs)

            if all(outcome_info == outcome_info[0]):
                succeed_coincide += 1

                if outcome_info[0] == branch:
                    succeed_branch += 1

                remaining_state = results[outcome_info[0]]["remaining_state"]
                tr_power = state_moment(remaining_state, K)
                p0 = np.real((1 + tr_power) / 2)
                p1 = np.real((1 - tr_power) / 2)

                result = np.random.choice([0, 1], p=[p0, p1])

                if result == 0:
                    swap_test_coincide_0 += 1

                    if outcome_info[0] == branch:
                        swap_test_branch_0 += 1

    N_eff = N_tot * shots_per_U

    branch_result = (2 * swap_test_branch_0 - succeed_branch) / N_eff
    coincide_result = (2 * swap_test_coincide_0 - succeed_coincide) / N_eff / L

    N_rho_copies = N_eff * K

    return branch_result, coincide_result, N_eff, N_rho_copies


def recover_tr_rho_K(K, m, d, expected_value, known_moments):
    global cycle_trace_product

    old_cycle_trace_product = cycle_trace_product
    try:
        cycle_trace_product = lambda perm, rho: np.prod([rho[l] for l in permutation_cycle_type(perm)])

        rho0 = dict(known_moments)
        rho0[K] = 0.0
        E0 = true_exp_value(K, m, d, rho0)

        rho1 = dict(known_moments)
        rho1[K] = 1.0
        E1 = true_exp_value(K, m, d, rho1)
    finally:
        cycle_trace_product = old_cycle_trace_product

    return np.real((expected_value - E0) / (E1 - E0))


def recover_moment_table_from_sigmaU_results(sigmaU_result_table, K_list, N_tot_list, m, d):

    sigmaU_result_table = np.asarray(sigmaU_result_table, dtype=float)

    moment_table = np.zeros_like(sigmaU_result_table, dtype=float)
    moment_dicts = []

    K_order = sorted(K_list)
    K_to_col = {K: j for j, K in enumerate(K_list)}

    for i, N_tot in enumerate(N_tot_list):

        known_moments = {1: 1.0}

        for K in K_order:
            col = K_to_col[K]
            expected_value_K = sigmaU_result_table[i, col]

            known_moments[K] = recover_tr_rho_K(K, m, d, expected_value_K, known_moments)

            moment_table[i, col] = known_moments[K]

        moment_dicts.append(known_moments)

    return moment_table, moment_dicts

def compare_recovered_moments_with_true(rho, K_list, N_tot_list, rho_power_table):

    rho_power_table = np.asarray(rho_power_table, dtype=float)

    true_power_values = {K: state_moment(rho, K) for K in K_list}

    abs_error_table = np.zeros_like(rho_power_table, dtype=float)

    for j, K in enumerate(K_list):
        abs_error_table[:, j] = np.abs(rho_power_table[:, j] - true_power_values[K])

    rows = []

    for i, N_tot in enumerate(N_tot_list):
        for j, K in enumerate(K_list):
            rows.append({"N_tot": N_tot, "K": K, "true_tr_rho_K": true_power_values[K], "estimate": rho_power_table[i, j], "abs_error": abs_error_table[i, j]})


    return abs_error_table



def store_bootstrap_counts_one_file(rho, depth, measured_registers, K_list, branch, N_U, N_M, save_path, save_every):
    """
    Store bootstrap-ready data in one .npz file.

    Parameters
    ----------
    N_U:
        Number of random unitaries.

    N_M:
        Number of measurement trials per fixed random unitary.

    N_tot:
        Total number of stored rows/samples, equal to N_U * N_M.

    One row = one measurement trial.

    If measured_registers == [], no random U is generated.
    Every row is treated as successful, and the swap test is done directly on rho.

    If measured_registers != [], then:
        - one random U is generated per outer loop,
        - the same U is reused for N_M trials,
        - tqdm and save_every are both based on total row/sample count.

    For each K, save:
        K{K}_coincide_success
        K{K}_coincide_swap0
        K{K}_branch_success
        K{K}_branch_swap0

    Also save:
        U_index[row] = which random U this row belongs to
        M_index[row] = which measurement trial under that U
    """

    rho = np.asarray(rho, dtype=complex)
    n = int(np.log2(rho.shape[0]))

    measured_registers = list(measured_registers)
    l = len(measured_registers)

    # do a ValueError if l >=n
    if l >= n:
        raise ValueError("Number of measured registers must be less than n.")

    L = 2**l

    N_U = int(N_U)
    N_M = int(N_M)
    N_tot = N_U * N_M

    data = {}

    for K in K_list:
        data[f"K{K}_coincide_success"] = np.zeros(N_tot, dtype=np.uint8)
        data[f"K{K}_coincide_swap0"] = np.zeros(N_tot, dtype=np.uint8)
        data[f"K{K}_branch_success"] = np.zeros(N_tot, dtype=np.uint8)
        data[f"K{K}_branch_swap0"] = np.zeros(N_tot, dtype=np.uint8)

    data["U_index"] = np.zeros(N_tot, dtype=np.int64)
    data["M_index"] = np.zeros(N_tot, dtype=np.int64)

    def save_checkpoint(row):
        save_dict = {
            "num_saved": np.asarray(row, dtype=np.int64),
            "N_tot": np.asarray(N_tot, dtype=np.int64),
            "N_U": np.asarray(N_U, dtype=np.int64),
            "N_M": np.asarray(N_M, dtype=np.int64),
            "n": np.asarray(n, dtype=np.int64),
            "depth": np.asarray(str(depth)),
            "measured_registers": np.asarray(measured_registers, dtype=np.int64),
            "K_list": np.asarray(K_list, dtype=np.int64),
            "branch": np.asarray(branch),
            "L": np.asarray(L, dtype=np.int64),
            "U_index": data["U_index"][:row],
            "M_index": data["M_index"][:row],
        }

        for K in K_list:
            for name in [
                "coincide_success",
                "coincide_swap0",
                "branch_success",
                "branch_swap0",
            ]:
                key = f"K{K}_{name}"
                save_dict[key] = data[key][:row]

        tmp_path = save_path + f".tmp_{os.getpid()}.npz"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(save_path):
            os.remove(save_path)
        np.savez_compressed(tmp_path, **save_dict)
        os.replace(tmp_path, save_path)

        print(
            f"Saved checkpoint at sample {row} / {N_tot} to {save_path}",
            flush=True,
        )

    row = 0

    pbar = tqdm(
        total=N_tot,
        desc="Collecting samples",
        unit="sample",
        mininterval=1.0,
    )

    try:
        if len(measured_registers) == 0:

            for u in range(N_U):

                for m_idx in range(N_M):

                    data["U_index"][row] = u
                    data["M_index"][row] = m_idx

                    for K in K_list:
                        data[f"K{K}_coincide_success"][row] = 1
                        data[f"K{K}_branch_success"][row] = 1

                        tr_power = state_moment(rho, K)

                        p0 = np.real((1 + tr_power) / 2)
                        p1 = np.real((1 - tr_power) / 2)

                        p0 = float(np.clip(p0, 0.0, 1.0))
                        p1 = float(np.clip(p1, 0.0, 1.0))

                        norm = p0 + p1
                        p0 = p0 / norm
                        p1 = p1 / norm

                        swap_result = np.random.choice([0, 1], p=[p0, p1])

                        if swap_result == 0:
                            data[f"K{K}_coincide_swap0"][row] = 1
                            data[f"K{K}_branch_swap0"][row] = 1

                    row += 1
                    pbar.update(1)

                    if row % save_every == 0 or row == N_tot:
                        save_checkpoint(row)

        else:

            branch_key = branch

            for u in range(N_U):

                U = local_brickwork_unitary_pennylane(n, depth)

                results = local_postselected_states_numpy(
                    rho,
                    measured_registers,
                    U,
                )

                branches = list(results.keys())
                probs = np.asarray(
                    [results[b]["probability"] for b in branches],
                    dtype=float,
                )
                probs = probs / probs.sum()

                for m_idx in range(N_M):

                    data["U_index"][row] = u
                    data["M_index"][row] = m_idx

                    for K in K_list:

                        outcome_info = np.random.choice(
                            branches,
                            size=K,
                            replace=True,
                            p=probs,
                        )

                        if np.all(outcome_info == outcome_info[0]):

                            data[f"K{K}_coincide_success"][row] = 1

                            outcome = outcome_info[0]
                            remaining_state = results[outcome]["remaining_state"]

                            tr_power = state_moment(remaining_state, K)

                            p0 = np.real((1 + tr_power) / 2)
                            p1 = np.real((1 - tr_power) / 2)

                            p0 = float(np.clip(p0, 0.0, 1.0))
                            p1 = float(np.clip(p1, 0.0, 1.0))

                            norm = p0 + p1
                            p0 = p0 / norm
                            p1 = p1 / norm

                            swap_result = np.random.choice([0, 1], p=[p0, p1])

                            if swap_result == 0:
                                data[f"K{K}_coincide_swap0"][row] = 1

                            if outcome == branch_key:
                                data[f"K{K}_branch_success"][row] = 1

                                if swap_result == 0:
                                    data[f"K{K}_branch_swap0"][row] = 1

                    row += 1
                    pbar.update(1)

                    if row % save_every == 0 or row == N_tot:
                        save_checkpoint(row)

    finally:
        pbar.close()

    return data


def store_full_register_measurement_data(rho, depth, N_U, N_M, save_path, save_every):
    """
    Store full-register measurement data after random U.

    This is for the l = n case, where all registers are measured.

    One row = one full computational-basis measurement outcome after applying one random unitary U.

    For each random U:
        1. Generate U.
        2. Compute rho_U = U rho U^dagger.
        3. Compute full-register measurement probabilities from diag(rho_U).
        4. Sample full-register outcomes in chunks.
        5. Save checkpoints every save_every total samples, or at the final sample.

    This version is safe for:
        N_U = 1, N_M very large,
        N_U very large, N_M fixed.

    If the program is aborted, the saved .npz file contains all rows up to the latest checkpoint.

    Saved arrays:
        outcomes[row]:
            Integer outcome in {0, ..., 2^n - 1}.

        U_index[row]:
            Which random unitary this row belongs to.

        M_index[row]:
            Which measurement trial under that U.
    """

    rho = np.asarray(rho, dtype=complex)
    n = int(np.log2(rho.shape[0]))
    d = 2**n


    N_U = int(N_U)
    N_M = int(N_M)
    save_every = int(save_every)


    N_tot = N_U * N_M

    data = {}
    data["outcomes"] = np.zeros(N_tot, dtype=np.uint64)
    data["U_index"] = np.zeros(N_tot, dtype=np.int64)
    data["M_index"] = np.zeros(N_tot, dtype=np.int64)

    def save_checkpoint(row):
        save_dict = {
            "num_saved": np.asarray(row, dtype=np.int64),
            "N_tot": np.asarray(N_tot, dtype=np.int64),
            "N_U": np.asarray(N_U, dtype=np.int64),
            "N_M": np.asarray(N_M, dtype=np.int64),
            "n": np.asarray(n, dtype=np.int64),
            "d": np.asarray(d, dtype=np.int64),
            "depth": np.asarray(str(depth)),
            "measurement_mode": np.asarray("full_register"),
            "outcomes": data["outcomes"][:row],
            "U_index": data["U_index"][:row],
            "M_index": data["M_index"][:row],
        }

        tmp_path = save_path + f".tmp_{os.getpid()}.npz"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(save_path):
            os.remove(save_path)
        np.savez_compressed(tmp_path, **save_dict)
        os.replace(tmp_path, save_path)

        print(f"Saved checkpoint at sample {row} / {N_tot} to {save_path}", flush=True)

    row = 0

    pbar = tqdm(total=N_tot, desc="Collecting full-register samples", unit="sample", mininterval=1.0)

    try:
        for u in range(N_U):
            U = local_brickwork_unitary_pennylane(n, depth)
            rho_U = U @ rho @ U.conj().T
            probs = np.real(np.diag(rho_U))
            probs = np.clip(probs, 0.0, None)

            prob_sum = probs.sum()

            if prob_sum <= 0:
                raise ValueError("Invalid measurement probability vector: sum is non-positive.")

            probs = probs / prob_sum

            m_idx = 0

            while m_idx < N_M:
                remaining_this_U = N_M - m_idx
                remaining_total = N_tot - row

                if row % save_every == 0:
                    samples_until_next_save = save_every
                else:
                    samples_until_next_save = save_every - (row % save_every)

                chunk_size = min(remaining_this_U, remaining_total, samples_until_next_save)

                sampled_outcomes = np.random.choice(d, size=chunk_size, replace=True, p=probs).astype(np.uint64)

                row_end = row + chunk_size
                m_end = m_idx + chunk_size

                data["outcomes"][row:row_end] = sampled_outcomes
                data["U_index"][row:row_end] = u
                data["M_index"][row:row_end] = np.arange(m_idx, m_end, dtype=np.int64)

                row = row_end
                m_idx = m_end

                pbar.update(chunk_size)

                if row % save_every == 0 or row == N_tot:
                    save_checkpoint(row)

    finally:
        pbar.close()

    return data


def estimate_sigmaU_from_indices(data, K, method, idx):
    L = int(data["L"])

    success = np.sum(data[f"K{K}_{method}_success"][idx])
    swap0 = np.sum(data[f"K{K}_{method}_swap0"][idx])

    N_eff = len(idx)

    if method == "coincide":
        estimate = (2 * swap0 - success) / N_eff / L
    elif method == "branch":
        estimate = (2 * swap0 - success) / N_eff
    else:
        raise ValueError("method must be 'coincide' or 'branch'")

    return estimate



def _comb_float_from_counts(counts, k):
    """
    Compute binom{count}{k} for each "count" in an array of "counts", returning the sum of the binomial coefficients as a float.
    """
    counts = np.asarray(counts, dtype=float)
    valid = counts >= k
    counts = counts[valid]
    if len(counts) == 0:
        return 0.0
    out = np.ones_like(counts, dtype=float)
    for r in range(k):
        out *= counts - r
    out /= math.factorial(k)
    return float(np.sum(out))


def _comb_float_scalar(N, k):
    """
    Compute binom{N}{k} for scalar N, returning a float.
    """
    N = float(N)
    if N < k:
        return 0.0
    out = 1.0
    for r in range(k):
        out *= N - r
    out /= math.factorial(k)
    return float(out)

def estimate_h_from_full_register_outcomes(outcomes, d, K_list):
    """
    Estimate h_k from full-register measurement outcomes for one fixed U.
    outcomes:
        Integer full-register outcomes under one fixed U.
    d:
        Hilbert-space dimension, d = 2^n.
    K_list:
        List of k values, for example [2,3,4,5].
    Returns:
        Dictionary {k: h_k_estimate}.
    """

    outcomes = np.asarray(outcomes, dtype=np.uint64)
    N_M = len(outcomes)

    if N_M == 0:
        raise ValueError("No outcomes were provided.")

    vals, counts = np.unique(outcomes, return_counts=True)

    h_dict = {}

    for k in K_list:

        if N_M < k:
            raise ValueError(f"Need N_M >= k. Got N_M={N_M}, k={k}.")

        collision_count = _comb_float_from_counts(counts, k)
        total_tuples = _comb_float_scalar(N_M, k)
        collision_prob = collision_count / total_tuples

        h_k = math.comb(d + k - 1, k) / d * collision_prob

        h_dict[k] = h_k

    return h_dict


def recover_moments_from_h_dict(h_dict, K_list):
    """
    Recover p_k = tr(rho^k) from h_k using Newton identities.

    Uses:
        k h_k = sum_{j=1}^k p_j h_{k-j}

    with:
        h_0 = 1
        p_1 = 1
        h_1 = 1
    """

    K_max = max(K_list)

    h = {0: 1.0, 1: 1.0}
    p = {1: 1.0}

    for k in range(2, K_max + 1):
        h[k] = float(h_dict[k])

    for k in range(2, K_max + 1):
        correction = 0.0
        for j in range(1, k):
            correction += p[j] * h[k - j]
        p[k] = k * h[k] - correction

    return {k: p[k] for k in K_list}


def bootstrap_recovered_moment_errors(data, method, K_list, true_values, N_U_boot, N_M_boot, B, fully_coherent=False):
    """
    Bootstrap/subsampling for q=1,...,n count-format data.

    For each bootstrap repetition:
        1. Pick N_U_boot U-blocks without replacement.
        2. For each picked U, pick N_M_boot rows without replacement.
        3. Estimate sigma_U moments from the selected rows.
        4. Recover tr(rho^K), unless fully_coherent=True.
        5. Compute absolute errors.

    This covers:
        NU1_vary_NM: N_U_boot=1, N_M_boot varies.
        NM100_vary_NU: N_M_boot=100, N_U_boot varies.
    """

    N_U_boot = int(N_U_boot)
    N_M_boot = int(N_M_boot)
    K_list = list(K_list)

    if N_U_boot <= 0:
        raise ValueError("N_U_boot must be positive.")
    if N_M_boot <= 0:
        raise ValueError("N_M_boot must be positive.")
    if method not in ["coincide", "branch"]:
        raise ValueError("method must be 'coincide' or 'branch'.")

    num_saved = int(data["num_saved"])
    n = int(data["n"])
    L = int(data["L"])
    d = 2**n
    m = d // L

    if "U_index" in data:
        U_index = np.asarray(data["U_index"][:num_saved], dtype=int)
    else:
        U_index = np.arange(num_saved, dtype=int)

    unique_U_all, first_indices, block_counts = np.unique(U_index, return_index=True, return_counts=True)
    usable_mask = block_counts >= N_M_boot

    if not np.any(usable_mask):
        raise ValueError("No usable U-blocks found. Need at least one U-block with at least N_M_boot rows.")

    usable_starts = first_indices[usable_mask]
    usable_counts = block_counts[usable_mask]
    n_U_available = len(usable_starts)

    if N_U_boot > n_U_available:
        raise ValueError(f"N_U_boot={N_U_boot} cannot exceed n_U_available={n_U_available} when sampling U without replacement.")

    success_arrays = {}
    swap0_arrays = {}

    for K in K_list:
        success_arrays[K] = np.asarray(data[f"K{K}_{method}_success"][:num_saved], dtype=np.float64)
        swap0_arrays[K] = np.asarray(data[f"K{K}_{method}_swap0"][:num_saved], dtype=np.float64)

    recovered = {}
    errors = {}
    sigmaU_bootstrap = {}

    for K in K_list:
        recovered[K] = np.zeros(B, dtype=float)
        errors[K] = np.zeros(B, dtype=float)
        sigmaU_bootstrap[K] = np.zeros(B, dtype=float)

    bootstrap_sizes = np.zeros(B, dtype=int)
    denom_factor = L if method == "coincide" else 1

    def compute_sigma_values_from_idx(idx):
        N_eff = len(idx)
        vals = []
        for K in K_list:
            success = float(np.sum(success_arrays[K][idx]))
            swap0 = float(np.sum(swap0_arrays[K][idx]))
            vals.append((2.0 * swap0 - success) / N_eff / denom_factor)
        return vals

    for b in tqdm(range(B), desc=f"Subsample NU={N_U_boot}, NM={N_M_boot}", mininterval=1.0):
        sampled_U_pos = np.random.choice(n_U_available, size=N_U_boot, replace=False)
        idx_list = []

        for u_pos in sampled_U_pos:
            start = usable_starts[u_pos]
            count = usable_counts[u_pos]
            sampled_offsets = np.random.choice(count, size=N_M_boot, replace=False)
            idx_list.append(start + sampled_offsets)

        idx = np.concatenate(idx_list)
        bootstrap_sizes[b] = len(idx)

        sigmaU_values = compute_sigma_values_from_idx(idx)

        for j, K in enumerate(K_list):
            sigmaU_bootstrap[K][b] = sigmaU_values[j]

        if fully_coherent:
            for j, K in enumerate(K_list):
                recovered[K][b] = sigmaU_values[j]
                errors[K][b] = abs(recovered[K][b] - true_values[K])
        else:
            sigmaU_result_table = np.array([sigmaU_values], dtype=float)
            moment_table, moment_dicts = recover_moment_table_from_sigmaU_results(sigmaU_result_table, K_list, [len(idx)], m, d)

            for j, K in enumerate(K_list):
                recovered[K][b] = moment_table[0, j]
                errors[K][b] = abs(recovered[K][b] - true_values[K])

    result = {}

    for K in K_list:
        result[f"mean_error_K{K}"] = float(np.mean(errors[K]))
        result[f"std_error_K{K}"] = float(np.std(errors[K], ddof=1))
        result[f"errors_K{K}"] = errors[K]
        result[f"recovered_K{K}"] = recovered[K]
        result[f"sigmaU_bootstrap_K{K}"] = sigmaU_bootstrap[K]

    result["num_saved"] = num_saved
    result["N_U_boot"] = N_U_boot
    result["N_M_boot"] = N_M_boot
    result["N_tot_boot"] = N_U_boot * N_M_boot
    result["n_U_available"] = n_U_available
    result["bootstrap_sizes"] = bootstrap_sizes

    if "N_U" in data:
        result["planned_N_U"] = int(data["N_U"])
    if "N_M" in data:
        result["planned_N_M"] = int(data["N_M"])

    return result


def bootstrap_full_register_moment_errors(data, K_list, true_values, N_U_boot, N_M_boot, B):
    """
    Bootstrap/subsampling for q=0 full-register measurement data.

    For each bootstrap repetition:
        1. Pick N_U_boot U-blocks without replacement.
        2. For each picked U, pick N_M_boot outcomes without replacement.
        3. Estimate h_k by collision counting for each picked U.
        4. Average h_k over picked U's.
        5. Recover tr(rho^K).
        6. Compute absolute errors.

    For q=0, N_M_boot must be at least max(K_list).
    """

    N_U_boot = int(N_U_boot)
    N_M_boot = int(N_M_boot)
    K_list = list(K_list)
    K_max = max(K_list)

    if N_U_boot <= 0:
        raise ValueError("N_U_boot must be positive.")
    if N_M_boot <= 0:
        raise ValueError("N_M_boot must be positive.")
    if N_M_boot < K_max:
        raise ValueError(f"N_M_boot must be at least max(K_list). Got N_M_boot={N_M_boot}, max(K_list)={K_max}.")

    num_saved = int(data["num_saved"])
    n = int(data["n"])
    d = int(data["d"]) if "d" in data else 2**n

    outcomes = np.asarray(data["outcomes"][:num_saved], dtype=np.int64)
    U_index = np.asarray(data["U_index"][:num_saved], dtype=int)

    unique_U_all, first_indices, block_counts = np.unique(U_index, return_index=True, return_counts=True)
    usable_mask = block_counts >= N_M_boot

    if not np.any(usable_mask):
        raise ValueError("No usable U-blocks found. Need at least one U-block with at least N_M_boot outcomes.")

    usable_starts = first_indices[usable_mask]
    usable_counts = block_counts[usable_mask]
    n_U_available = len(usable_starts)

    if N_U_boot > n_U_available:
        raise ValueError(f"N_U_boot={N_U_boot} cannot exceed n_U_available={n_U_available} when sampling U without replacement.")

    recovered = {}
    errors = {}
    h_bootstrap = {}

    for K in K_list:
        recovered[K] = np.zeros(B, dtype=float)
        errors[K] = np.zeros(B, dtype=float)
        h_bootstrap[K] = np.zeros(B, dtype=float)

    bootstrap_sizes = np.zeros(B, dtype=int)
    comb_d_factor = {}
    tuple_denom = {}

    for K in K_list:
        comb_d_factor[K] = math.comb(d + K - 1, K) / d
        tuple_denom[K] = _comb_float_scalar(N_M_boot, K)

    def comb_counts_matrix(counts_matrix, K):
        counts_float = counts_matrix.astype(float)
        out = np.ones_like(counts_float, dtype=float)

        for r in range(K):
            out *= counts_float - r

        out /= math.factorial(K)
        out[counts_matrix < K] = 0.0
        return np.sum(out, axis=1)

    for b in tqdm(range(B), desc=f"Subsample full-register NU={N_U_boot}, NM={N_M_boot}", mininterval=1.0):
        sampled_U_pos = np.random.choice(n_U_available, size=N_U_boot, replace=False)
        sampled_outcomes_list = []

        for u_pos in sampled_U_pos:
            start = usable_starts[u_pos]
            count = usable_counts[u_pos]
            sampled_offsets = np.random.choice(count, size=N_M_boot, replace=False)
            sampled_outcomes_list.append(outcomes[start + sampled_offsets])

        sampled_outcomes = np.vstack(sampled_outcomes_list)
        counts_matrix = np.zeros((N_U_boot, d), dtype=np.int64)
        row_ids = np.repeat(np.arange(N_U_boot), N_M_boot)
        np.add.at(counts_matrix, (row_ids, sampled_outcomes.reshape(-1)), 1)

        h_avg = {}

        for K in K_list:
            collision_counts_by_U = comb_counts_matrix(counts_matrix, K)
            collision_probs_by_U = collision_counts_by_U / tuple_denom[K]
            h_by_U = comb_d_factor[K] * collision_probs_by_U
            h_avg[K] = float(np.mean(h_by_U))
            h_bootstrap[K][b] = h_avg[K]

        recovered_dict = recover_moments_from_h_dict(h_avg, K_list)

        for K in K_list:
            recovered[K][b] = recovered_dict[K]
            errors[K][b] = abs(recovered[K][b] - true_values[K])

        bootstrap_sizes[b] = N_U_boot * N_M_boot

    result = {}

    for K in K_list:
        result[f"mean_error_K{K}"] = float(np.mean(errors[K]))
        result[f"std_error_K{K}"] = float(np.std(errors[K], ddof=1))
        result[f"errors_K{K}"] = errors[K]
        result[f"recovered_K{K}"] = recovered[K]
        result[f"h_bootstrap_K{K}"] = h_bootstrap[K]

    result["num_saved"] = num_saved
    result["N_U_boot"] = N_U_boot
    result["N_M_boot"] = N_M_boot
    result["N_tot_boot"] = N_U_boot * N_M_boot
    result["n_U_available"] = n_U_available
    result["bootstrap_sizes"] = bootstrap_sizes

    if "N_U" in data:
        result["planned_N_U"] = int(data["N_U"])
    if "N_M" in data:
        result["planned_N_M"] = int(data["N_M"])

    return result