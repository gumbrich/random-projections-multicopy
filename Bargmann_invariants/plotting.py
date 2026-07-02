import matplotlib.pyplot as plt
import numpy as np


def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 18,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


COLORS = {
    2: "#4C72B0",
    3: "#55A868",
    4: "#C44E52",
    5: "#8172B2",
}


MARKERS = {
    2: "o",
    3: "s",
    4: "D",
    5: "^",
}


def plot_required_samples_vs_q(
    df,
    state,
    nreq_col="N_req",
    savepath=None,
):

    set_plot_style()

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    sub_state = df[df["state"] == state].copy()

    for K in sorted(sub_state["K"].unique()):

        sub = sub_state[sub_state["K"] == K]
        sub = sub.sort_values("q")

        q = sub["q"].to_numpy(float)
        y = sub[nreq_col].to_numpy(float)

        ax.plot(
            q,
            y,
            lw=2.6,
            alpha=0.75,
            color=COLORS[K],
            marker=MARKERS[K],
            ms=10,
            mfc="white",
            mec=COLORS[K],
            mew=2.0,
            label=rf"$K={K}$",
        )

    ax.set_yscale("log")

    ax.set_xlabel(r"Remaining qubits, $q$")
    ax.set_ylabel(r"Samples, $N_{\rm req}$")

    ax.legend(frameon=False)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=600, bbox_inches="tight")

    plt.show()

def format_total_copies(N_protocol, copies_per_execution=5):
    import numpy as np

    N_total = copies_per_execution * int(N_protocol)
    exponent = int(np.floor(np.log10(N_total)))
    mantissa = N_total / 10**exponent

    if abs(mantissa - 1.0) < 1e-12:
        return rf"$N=10^{{{exponent}}}$" # INFO: previously used N_\mathrm{{copies}}, but back to N because this is how we introduce it in the manuscript now

    if abs(mantissa - round(mantissa)) < 1e-12:
        mantissa = int(round(mantissa))

    return rf"$N={mantissa}\times10^{{{exponent}}}$"

def plot_thermal_reconstruction_two_N(df, use="mean", savepath=None):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    df = df[df["q"].isin([1, 3])].copy()
    N_values = sorted(df["N"].unique())

    colors = {1: "#4C72B0", 3: "#C44E52"}
    markers = {1: "s", 3: "^"}

    fig, axes = plt.subplots(
        1,
        len(N_values),
        figsize=(6.2, 2.9),
        sharey=True,
    )

    if len(N_values) == 1:
        axes = [axes]

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, (ax, N) in enumerate(zip(axes, N_values)):
        subN = df[df["N"] == N]

        exact = (
            subN.groupby("beta")["exact"]
            .first()
            .sort_index()
        )

        for q in sorted(subN["q"].unique()):
            sub = subN[subN["q"] == 4-q].sort_values("beta") # INFO: 4-q, because original notation was wrong and data obtained differently (can be fixed, but runs couple hours)

            x = sub["beta"].values
            y = sub[use].values
            ylow = sub["q25"].values
            yhigh = sub["q75"].values

            ax.plot(
                x,
                y,
                marker=markers[q],
                ms=4.8,
                lw=1.6,
                color=colors[q],
                label=fr"$q={q}$",
                zorder=5,
            )

            ax.fill_between(
                x,
                ylow,
                yhigh,
                color=colors[q],
                alpha=0.08,
                linewidth=0,
                zorder=1,
            )

        ax.plot(
            exact.index,
            exact.values,
            "--",
            color="black",
            lw=1.9,
            label="exact" if i == 0 else None,
            zorder=20,
        )

        ax.set_title(format_total_copies(N), pad=5)
        ax.set_xlim(exact.index.min(), exact.index.max())
        ax.set_ylim(-0.05, 0.72)

        ax.text(
            0.04,
            0.92,
            panel_labels[i],
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="left",
        )

        ax.tick_params(axis="both", which="major", pad=3)

    axes[0].set_ylabel(
        r"$\widehat p_3=\widehat{\mathrm{tr}(\rho_\beta^3)}$"
    )

    fig.supxlabel(
        r"Inverse temperature, $\beta$",
        fontsize=17,
        y=0.06,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        handlelength=2.1,
        columnspacing=1.5,
    )

    plt.subplots_adjust(
        left=0.12,
        right=0.985,
        bottom=0.25,
        top=0.90,
        wspace=0.10,
    )

    if savepath is not None:
        fig.savefig(savepath, dpi=600, bbox_inches="tight")

        if str(savepath).endswith(".pdf"):
            fig.savefig(
                savepath, #str(savepath).replace(".pdf", ".png"),
                dpi=600,
                bbox_inches="tight",
            )

    plt.show()