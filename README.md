# Random Projections for Multi-Copy Quantum Algorithms

This repository contains the numerical examples and plotting scripts used for the manuscript

**Random Projections for Multi-Copy Quantum Algorithms**
https://arxiv.org/abs/2606.20238

## Folder overview

| Folder                            | Manuscript figure | Purpose                                                                                                 |
| --------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------- |
| `Ising_example/`                  | Fig. 2            | Thermal transverse-field Ising example: estimate `tr(rho_beta^3)` as a function of inverse temperature. |
| `num_copies_vs_abs_error/`        | Fig. 3            | Plot absolute estimation error versus the number of consumed copies.                                    |
| `kept_qubits_vs_required_copies/` | Fig. 4            | Postprocess noisy GHZ data to estimate the required number of copies as a function of kept qubits.      |
| `Bargmann_invariants/`            | Fig. 5            | Reconstruct the three-state Bargmann invariant in the complex plane.                                    |

## `Ising_example/`

This folder is for the thermal transverse-field Ising model example in **Fig. 2**.

The goal is to estimate

```text
p_3 = tr(rho_beta^3)
```

for thermal states of a 5-qubit transverse-field Ising Hamiltonian, as a function of inverse temperature.

### Files

```text
Ising_example/
├── generate_data.py
├── ising_protocol.py
├── plot_one_panel.py
└── data/
    ├── ising_fig2_N5000.csv
    └── ising_fig2_N5000.pkl
```

### How this folder works

`ising_protocol.py` contains the core functions for this example, including the Ising Hamiltonian, thermal-state generation, Haar-random projection, projected-moment estimation, and moment reconstruction.

`generate_data.py` generates the data for Fig. 2 with specified number of copies. The important parameters are set at the top of the file:

```python
N_QUBITS = 5
J = 1.0
H_FIELD = 1.0
K_TARGET = 3
N_TOTAL = 5000
Q_KEEP_LIST = (1, 3)
B = 100
```

`plot_one_panel.py` loads the generated `.pkl` file from `data/` and plots one panel of Fig. 2.


## `num_copies_vs_abs_error/`

This folder is for the noisy GHZ absolute-error curves in **Fig. 3**.

The goal is to estimate

```text
E[|p_hat_K - p_K|]
```

as a function of the number of consumed state copies, for different numbers of kept coherent qubits `q`.

In the manuscript, Fig. 3 focuses on `K = 3`, but the scripts can also process `K = 2, 3, 4, 5`.

### Files

```text
num_copies_vs_abs_error/
├── run_raw_data.py
├── process_raw_data.py
├── functions.py
├── states.py
├── plots.ipynb
├── raw_data/
└── bootstrap_data/
```

### How this folder works

This folder has a three-step workflow.

### Step 1: Generate raw data

`run_raw_data.py` generates raw `.npz` files for the noisy GHZ state. The main parameters are at the top of the file:

```python
n = 5
depth = 5
q = 2
N_U = 10
N_M = 100
p = 0.3
K_list = [2, 3, 4, 5]
```

Here:

* `n` is the number of qubits.
* `q` is the number of kept coherent qubits after projection.
* `depth` is the brickwork circuit depth.
* `N_U` is the number of random unitaries.
* `N_M` is the number of measurements per random unitary.
* `p = 0.3` is the depolarizing noise parameter for the noisy GHZ state.


### Step 2: Bootstrap the raw data

`process_raw_data.py` reads the raw `.npz` files and generates bootstrapped error curves.

The key setting is

```python
plot_mode = "NU1_vary_NM"
```

or

```python
plot_mode = "NM100_vary_NU"
```

These two modes correspond to the two panels of Fig. 3:

```text
NU1_vary_NM      -> fix N_U = 1 and vary N_M
NM100_vary_NU    -> fix N_M = 100 and vary N_U
```

### Step 3: Plot the curves

`plots.ipynb` loads the files in `bootstrap_data/` and produces the Fig. 3-style plot.

The most important plotting parameters are:

```python
q_list = [0, 1, 2, 3, 4, 5]
plot_mode = "NU1_vary_NM"
K_plot = 3
x_axis_mode = "copy"
```

## `kept_qubits_vs_required_copies/`

This folder is for the required-copy scaling plot in **Fig. 4**.

The goal is to estimate how many total copies are required to reach a target error

```text
epsilon_p ≲ 0.1
```

as a function of the number of kept coherent qubits.

This folder does not generate the original measurement data from scratch. Instead, it postprocesses existing `.npz` files, which are compatible with the raw data generated in `num_copies_vs_abs_error/`.

### Files

```text
kept_qubits_vs_required_copies/
├── complexity.py
├── existing_npz_coincide_postprocess_defs.py
├── plot_complexity.ipynb
├── raw_npz_data/
└── existing_npz_postprocess_outputs/
```

### How this folder works

`raw_npz_data/` contains the raw `.npz` inputs. These files can be copied from

```text
num_copies_vs_abs_error/raw_data/
```

or generated using `run_raw_data.py` in the previous folder.

`complexity.py` reads the raw `.npz` files and estimates the first crossing point where the error reaches the target threshold. The main parameters are:

```python
n = 5
q_values = [1, 2, 3, 4, 5]
K_list = [2, 3, 4]
cases = ["NM1_vary_NU", "NM100_vary_NU"]
lower_modes = ["exact", "hierarchical"]
base_epsilon = 0.1
B = 300
```

The script writes three CSV files:

```text
existing_npz_postprocess_outputs/total_copy_complexity_summary.csv
existing_npz_postprocess_outputs/bootstrap_first_crossings.csv
existing_npz_postprocess_outputs/full_data_errors.csv
```


## `Bargmann_invariants/`

This folder is for the Bargmann-invariant reconstruction in **Fig. 5**.

The goal is to reconstruct the complex three-state Bargmann invariant

```text
Delta_123 = <psi_1|psi_2><psi_2|psi_3><psi_3|psi_1>
```

from random projections.

The plot shows the reconstructed values in the complex plane and compares them with the exact trajectory.

### Files

```text
Bargmann_invariants/
├── plot_bargmann_k3.py
├── plotting.py
└── bargmann_n4_high_samples_data.npz
```

### How this folder works

`plot_bargmann_k3.py` performs the full simulation and plotting. It constructs three four-qubit product states, varies the azimuthal angle of the third state, reconstructs `Delta_123`, and plots the result in the complex plane.

The script produces three output figures:

```text
bargmann_n4_high_samples_complex.pdf
bargmann_n4_high_samples_error.png
bargmann_n4_high_samples_phase.png
```

and one data file:

```text
bargmann_n4_high_samples_data.npz
```

The main figure used in the manuscript is

```text
bargmann_n4_high_samples_complex.pdf
```

Optional arguments include:

```bash
python plot_bargmann_k3.py --n 4 --N 50000 --num-batches 50 --num-phi 121
```

Here:

* `--n` is the number of qubits.
* `--N` is the number of protocol executions per estimator.
* `--num-batches` controls the number of independent batches used for the standard error.
* `--num-phi` controls the number of points along the exact trajectory.

---

## Notation

Throughout the repository, `q` denotes the number of kept coherent qubits after projection.

Thus:

```text
q = n     -> fully coherent generalized swap-test limit
q = 0     -> fully local randomized-measurement limit
0 < q < n -> intermediate random-projection protocol
```

The projected subspace dimension is

```text
m = 2^q
```

and the original Hilbert-space dimension is

```text
d = 2^n.
```

Smaller `q` means stronger compression, which reduces the coherent Hilbert-space size but increases the statistical sampling overhead.
