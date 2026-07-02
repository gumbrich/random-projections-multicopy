import numpy as np
from functions import *
from states import *
import os

n = 5 # number of qubits
q = 2 # number of qubits compressed n -> q (q=0 means fully coherent algo, q=n means fully local algo)
plot_mode = "NU1_vary_NM" # plot_mode can be either "NU1_vary_NM" (varying N_M with fixed N_U=1) or "NM100_vary_NU" (varying N_U with fixed N_M=100)

raw_dir = "raw_data" # directory where the raw data is stored
result_dir = "bootstrap_data" # directory where the bootstrap results will be saved
p = 0.3 # noise parameter for the noisy GHZ state
depth = 5 # brickwork circuit depth. Positive integers or np.inf (denoting perfect haar randomness)
K_list = [2, 3, 4, 5] # the state powers to be estimated. For example, K_list = [2, 3, 4, 5] means the goal is to estimate tr(rho^2), tr(rho^3), tr(rho^4), tr(rho^5)..
method = "coincide" # method can be either "coincide" or "branch". "coincide" matches the method throughout the manuscript. "branch" is the method with fixed projector (measurement outcomes)
B = 10 # number of bootstrap trials

os.makedirs(result_dir, exist_ok=True)

if q < 0 or q > n:
    raise ValueError("q must satisfy 0 <= q <= n.")

if plot_mode not in ["NM100_vary_NU", "NU1_vary_NM"]:
    raise ValueError("plot_mode must be one of: NM100_vary_NU, NU1_vary_NM.")

param_tag = f"p{str(p).replace('.', 'd')}"
rho = noisy_ghz_state(n, p)
true_values = {K: state_moment(rho, K) for K in K_list}
fully_coherent = q == n
fully_local = q == 0

if plot_mode == "NU1_vary_NM":
    N_boot_list = np.unique(np.logspace(2, 5, 30).astype(int)) # set the suitable range of N_boot_list to vary NU or NM. For example, np.logspace(2, 5, 30) means N_boot varies from 10^2 to 10^5 with 30 points in log scale.
    raw_path = os.path.join(raw_dir, f"noisy_ghz_n{n}_{param_tag}_depth{depth}_q{q}_NU1_NM10000000_data.npz") # 
else:
    N_U_boot_list = np.unique(np.logspace(0, 5, 30).astype(int)) # similar here
    N_boot_list = 100 * N_U_boot_list
    raw_path = os.path.join(raw_dir, f"noisy_ghz_n{n}_{param_tag}_depth{depth}_q{q}_NU100000_NM100_data.npz")

if not os.path.exists(raw_path):
    print("Missing file:", raw_path, flush=True)
    raise SystemExit(0)

print("Starting bootstrap", flush=True)
print("n =", n, flush=True)
print("q =", q, flush=True)
print("plot_mode =", plot_mode, flush=True)
print("fully_local =", fully_local, flush=True)
print("fully_coherent =", fully_coherent, flush=True)
print("raw_dir =", raw_dir, flush=True)
print("result_dir =", result_dir, flush=True)
print("raw_path =", raw_path, flush=True)
print("B =", B, flush=True)
print("N_boot_list =", N_boot_list, flush=True)

data = np.load(raw_path)
x_values = []
mean_errors_by_K = {K: [] for K in K_list}
std_errors_by_K = {K: [] for K in K_list}

for N_boot in N_boot_list:
    if plot_mode == "NU1_vary_NM":
        N_U_boot, N_M_boot, x_value = 1, int(N_boot), int(N_boot)
    else:
        if N_boot < 100:
            continue
        N_M_boot = 100
        N_U_boot = max(1, int(N_boot) // N_M_boot)
        x_value = N_U_boot * N_M_boot

    try:
        if fully_local:
            result = bootstrap_full_register_moment_errors(data=data, K_list=K_list, true_values=true_values, N_U_boot=N_U_boot, N_M_boot=N_M_boot, B=B)
        else:
            result = bootstrap_recovered_moment_errors(data=data, method=method, K_list=K_list, true_values=true_values, N_U_boot=N_U_boot, N_M_boot=N_M_boot, B=B, fully_coherent=fully_coherent)
    except ValueError as e:
        print(f"Skip n={n}, q={q}, plot_mode={plot_mode}, N_boot={N_boot}, N_U_boot={N_U_boot}, N_M_boot={N_M_boot}: {e}", flush=True)
        continue

    x_values.append(x_value)

    for K in K_list:
        mean_errors_by_K[K].append(result[f"mean_error_K{K}"])
        std_errors_by_K[K].append(result[f"std_error_K{K}"])

    print(f"n={n}, q={q}, plot_mode={plot_mode}, x={x_value}, N_U_boot={N_U_boot}, N_M_boot={N_M_boot}", flush=True)

    for K in K_list:
        print(f"    K={K}, mean={result[f'mean_error_K{K}']:.6e}, std={result[f'std_error_K{K}']:.6e}", flush=True)

data.close()

x_values = np.asarray(x_values, dtype=int)
save_dict = {"N_values": x_values, "K_list": np.asarray(K_list, dtype=int)}

for K in K_list:
    save_dict[f"mean_error_K{K}"] = np.asarray(mean_errors_by_K[K], dtype=float)
    save_dict[f"std_error_K{K}"] = np.asarray(std_errors_by_K[K], dtype=float)

save_curve_path = os.path.join(result_dir, f"bootstrap_curve_noisy_ghz_n{n}_{param_tag}_depth{depth}_q{q}_{plot_mode}.npz")
np.savez_compressed(save_curve_path, **save_dict)

print("Saved curve:", save_curve_path, flush=True)
print("Finished bootstrap", flush=True)