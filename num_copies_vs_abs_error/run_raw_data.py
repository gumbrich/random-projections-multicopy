from functions import *
from states import *
import os

n = 5  # number of qubits
depth = 5 # brickwork circuit depth. Positive integers or np.inf (denoting perfect haar randomness)
q = 2 # number of qubits compressed n -> q (q=0 means fully coherent algo, q=n means fully local algo)
N_U = 10 # number of random unitaries
N_M = 100 # number of measurements per random unitary, total number of circuit executions N_U * N_M

save_every = 10 # save the raw data every 'save_every' runs
outdir = "raw_data" # output directory for the raw data

if q < 0 or q > n:
    raise ValueError("q must satisfy 0 <= q <= n.")

### This part can be modified to different states and parameters
p = 0.3 # noise parameter
rho = noisy_ghz_state(n, p) # noisy ghz state
### This part can be modified to different states and parameters

measured_registers = list(range(n - q))
branch = "0" * len(measured_registers) # branch result such that the projector is fixed, i.e., we only continue when the measured qubits are all 0 (by default).
K_list = [2, 3, 4, 5] # the state powers to be estimated. For example, K_list = [2, 3, 4, 5] means the goal is to estimate tr(rho^2), tr(rho^3), tr(rho^4), tr(rho^5)..

os.makedirs(outdir, exist_ok=True)

param_tag = f"p{str(p).replace('.', 'd')}"
depth_tag = str(depth).replace(".", "d")

print("n =", n, flush=True)
print("p =", p, flush=True)
print("depth =", depth, flush=True)
print("q =", q, flush=True)
print("measured register length =", len(measured_registers), flush=True)
print("measured_registers =", measured_registers, flush=True)
print("N_U =", N_U, flush=True)
print("N_M =", N_M, flush=True)
print("save_every =", save_every, flush=True)
print("K_list =", K_list, flush=True)

if len(measured_registers) == 0:
    mode = "fully coherent"
    save_path = os.path.join(outdir, f"noisy_ghz_n{n}_{param_tag}_depth{depth_tag}_q{q}_NU{N_U}_NM{N_M}_data_fully_coherent.npz")
    print("mode =", mode, flush=True)
    print("save_path =", save_path, flush=True)
    data = store_bootstrap_counts_one_file(rho, depth, measured_registers, K_list, branch, N_U, N_M, save_path, save_every)

elif len(measured_registers) < n:
    mode = "partial randomized projection"
    save_path = os.path.join(outdir, f"noisy_ghz_n{n}_{param_tag}_depth{depth_tag}_q{q}_NU{N_U}_NM{N_M}_data_measured_{len(measured_registers)}_qubits.npz")
    print("mode =", mode, flush=True)
    print("save_path =", save_path, flush=True)
    data = store_bootstrap_counts_one_file(rho, depth, measured_registers, K_list, branch, N_U, N_M, save_path, save_every)

elif len(measured_registers) == n:
    mode = "fully local full-register measurement"
    save_path = os.path.join(outdir, f"noisy_ghz_n{n}_{param_tag}_depth{depth_tag}_q{q}_NU{N_U}_NM{N_M}_data_fully_local.npz")
    print("mode =", mode, flush=True)
    print("save_path =", save_path, flush=True)
    data = store_full_register_measurement_data(rho, depth, N_U, N_M, save_path, save_every)

