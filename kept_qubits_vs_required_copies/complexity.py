from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from existing_npz_coincide_postprocess_defs import *


DATA_DIR = Path("raw_npz_data") # Directory containing the raw NPZ data files. The files can be generated with codes in the directory "num_copies_vs_abs_error"
OUT_DIR = Path("existing_npz_postprocess_outputs") # Directory where the post-processed outputs will be saved.
OUT_DIR.mkdir(parents=True, exist_ok=True)



n = 5 # Number of qubits in the state.
state_name = "noisy_GHZ_p0.3" # Name of the quantum state.

q_values = [1, 2, 3, 4, 5] # List of q values for which the complexity is computed. The code will look for files with these q values in the DATA_DIR.
K_list = [2, 3, 4] # List of K values for which the complexity is computed. The code will look for files with these K values in the DATA_DIR.

cases = ["NM1_vary_NU", "NM100_vary_NU"] # List of cases for which the complexity is computed. The code will look for files with these cases in the DATA_DIR.
lower_modes = ["exact", "hierarchical"] # List of lower bounding modes for which the complexity is computed. "exact" means the lower powers are known exactly, while "hierarchical" means the lower powers are estimated hierarchically.

base_epsilon = 0.1 # required epsilon_p. Note that this should be tund depending on the number of samples in the raw data. If the number of samples is small, the base_epsilon should be larger to acquire the accurate results as much as possible.
epsilon_safety_factor = 1.05 # Safety factor for numerical stability.

B = 300 # Number of bootstrap trials for estimating the required N.
num_grid = 70 # Number of grid points for the interpolation of the error function. The code will interpolate the error function to find the first crossing point at epsilon_p.



true_moments = true_moment_dict(state_name, n, K_list) # compute exact moments



summary, boot, full_error_table = process_existing_npz_set_first_crossing(
    data_dir=DATA_DIR,
    q_values=q_values,
    K_list=K_list,
    true_moments=true_moments,
    cases=cases,
    lower_modes=lower_modes,
    state_name=state_name,
    state_hint="noisy_ghz",
    base_epsilon=base_epsilon,
    epsilon_safety_factor=epsilon_safety_factor,
    B=B,
    num_grid=num_grid,
    seed=np.random.randint(0, 1000000),
    show_progress=True,
)


summary.to_csv(OUT_DIR / "total_copy_complexity_summary.csv", index=False)
boot.to_csv(OUT_DIR / "bootstrap_first_crossings.csv", index=False)
full_error_table.to_csv(OUT_DIR / "full_data_errors.csv", index=False)