"""Plot one generated thermal Ising panel without saving the figure."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Match this to N_TOTAL in generate_data.py.
N_TOTAL = 5000 # number of copies. It is considered as the total copies used for p2 and p3 each. The total number of state copies is 2*n_exec + 3*n_exec = 5*n_exec.
DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    df = pd.read_pickle(DATA_DIR / f"ising_fig2_N{N_TOTAL}.pkl")

    fig, ax = plt.subplots(figsize=(3.4, 2.9))

    exact = df.groupby("beta")["exact"].first().sort_index()
    ax.plot(exact.index, exact.values, "--", color="black", lw=1.8, label="exact")

    markers = {1: "s", 3: "^"}

    for q_keep in sorted(df["q_keep"].unique()):
        sub = df[df["q_keep"] == q_keep].sort_values("beta")
        ax.plot(
            sub["beta"],
            sub["mean"],
            marker=markers.get(int(q_keep), "o"),
            lw=1.6,
            ms=4.8,
            label=rf"$q={int(q_keep)}$",
        )
        ax.fill_between(sub["beta"], sub["q25"], sub["q75"], alpha=0.15, linewidth=0)

    ax.set_xlabel(r"Inverse temperature, $\beta$")
    ax.set_ylabel(r"$\widehat p_3=\widehat{\mathrm{tr}(\rho_\beta^3)}$")
    ax.set_title(rf"$N={N_TOTAL:g}$")
    ax.set_ylim(-0.05, 0.72)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
