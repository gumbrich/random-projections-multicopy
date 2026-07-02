#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from plotting import set_plot_style

set_plot_style()

COLORS_Q = {
    1: "#4C72B0",
    2: "#55A868",
    3: "#C44E52",
    4: "#8172B2",
}

MARKERS_Q = {
    1: "o",
    2: "s",
    3: "D",
    4: "^",
}

def haar_unitary(d, rng):
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    phases = np.diag(r) / np.abs(np.diag(r))
    return q * phases.conj()


def kron_power(v, n):
    out = np.array([1.0 + 0.0j])
    for _ in range(n):
        out = np.kron(out, v)
    return out


def product_state(theta, phi, n):
    single = np.array(
        [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)],
        dtype=complex,
    )
    return kron_power(single, n)


def Delta_123(psi1, psi2, psi3):
    return np.vdot(psi1, psi2) * np.vdot(psi2, psi3) * np.vdot(psi3, psi1)


def Delta_132(psi1, psi2, psi3):
    return np.vdot(psi1, psi3) * np.vdot(psi3, psi2) * np.vdot(psi2, psi1)


def Delta2_Sigma_exact(psi1, psi2, psi3):
    return (
        abs(np.vdot(psi1, psi2)) ** 2
        + abs(np.vdot(psi1, psi3)) ** 2
        + abs(np.vdot(psi2, psi3)) ** 2
    )


def k2_coefficients(d, m):
    gamma_e = (d * m**2 - m) / (d * (d**2 - 1))
    gamma_t = (d * m - m**2) / (d * (d**2 - 1))
    return gamma_e, gamma_t


def k3_coefficients(d, m):
    a_sym = m * (m + 1) * (m + 2) / (d * (d + 1) * (d + 2))
    a_std = m * (m**2 - 1) / (d * (d**2 - 1))
    a_asym = m * (m - 1) * (m - 2) / (d * (d - 1) * (d - 2))

    gamma_e = (a_sym + 4 * a_std + a_asym) / 6
    gamma_t = (a_sym - a_asym) / 6
    gamma_c = (a_sym - 2 * a_std + a_asym) / 6
    return gamma_e, gamma_t, gamma_c


def projected_trace_pure_2(psi_a, psi_b, PU):
    return np.vdot(psi_b, PU @ psi_a) * np.vdot(psi_a, PU @ psi_b)


def projected_trace_pure_123(psi1, psi2, psi3, PU):
    return (
        np.vdot(psi3, PU @ psi1)
        * np.vdot(psi1, PU @ psi2)
        * np.vdot(psi2, PU @ psi3)
    )


def reconstruct_pair_overlap(raw_mean_2, d, m):
    gamma_e, gamma_t = k2_coefficients(d, m)
    return (raw_mean_2 - gamma_t) / gamma_e


def reconstruct_Delta123_pure(raw_mean_123, Delta2Sigma_hat, d, m):
    gamma_e, gamma_t, gamma_c = k3_coefficients(d, m)

    # Corrected K=3 Bargmann reconstruction:
    # raw_mean_123 = gamma_e Delta_123 + gamma_c Delta_132
    #              + gamma_t Delta2Sigma + gamma_c
    # for normalized states, with Delta_132 = conj(Delta_123).
    F = raw_mean_123 - gamma_t * Delta2Sigma_hat - gamma_c
    A = gamma_e
    B = gamma_c

    return (A * F - B * np.conj(F)) / (A**2 - B**2)


def stderr_from_batches(x):
    if x.shape[0] <= 1:
        return np.zeros(x.shape[1], dtype=float)
    return np.std(x, axis=0, ddof=1) / np.sqrt(x.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--N", type=int, default=50000, help="Protocol executions per estimator.")
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--num-phi", type=int, default=121)
    parser.add_argument("--theta", type=float, default=np.pi / 2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-prefix", type=str, default="bargmann_n4_high_samples")
    args = parser.parse_args()

    if args.N % args.num_batches != 0:
        raise ValueError("N must be divisible by num_batches.")

    rng = np.random.default_rng(args.seed)

    n = args.n
    d = 2**n
    N = args.N
    Nprime = 5 * N
    batch_size = N // args.num_batches
    phis = np.linspace(0, 2 * np.pi, args.num_phi)

    psi1 = product_state(theta=0.0, phi=0.0, n=n)
    psi2 = product_state(theta=args.theta, phi=0.0, n=n)

    psi3s = []
    Delta123_exact = np.empty(args.num_phi, dtype=complex)
    Delta132_exact = np.empty(args.num_phi, dtype=complex)
    Delta2Sigma_exact_vals = np.empty(args.num_phi, dtype=float)

    for i, phi in enumerate(phis):
        psi3 = product_state(theta=args.theta, phi=phi, n=n)
        psi3s.append(psi3)
        Delta123_exact[i] = Delta_123(psi1, psi2, psi3)
        Delta132_exact[i] = Delta_132(psi1, psi2, psi3)
        Delta2Sigma_exact_vals[i] = Delta2_Sigma_exact(psi1, psi2, psi3)

    q_values = list(range(1, n + 1))

    Delta123_hat_mean = {}
    Delta123_hat_stderr = {}
    Delta123_hat_batches = {}
    Delta2Sigma_hat_mean = {}
    Delta2Sigma_hat_stderr = {}

    for q_keep in q_values:
        m = 2**q_keep
        print(f"\nq={q_keep}, m={m}")

        batch_raw_12 = np.zeros((args.num_batches, args.num_phi), dtype=complex)
        batch_raw_13 = np.zeros((args.num_batches, args.num_phi), dtype=complex)
        batch_raw_23 = np.zeros((args.num_batches, args.num_phi), dtype=complex)
        batch_raw_123 = np.zeros((args.num_batches, args.num_phi), dtype=complex)

        for b in range(args.num_batches):
            for _ in range(batch_size):
                # One Haar projection for each of the four estimators.
                # This uses the same number N of executions per estimator.
                U2_12 = haar_unitary(d, rng)
                U2_13 = haar_unitary(d, rng)
                U2_23 = haar_unitary(d, rng)
                U3 = haar_unitary(d, rng)

                P12 = U2_12[:, :m] @ U2_12[:, :m].conj().T
                P13 = U2_13[:, :m] @ U2_13[:, :m].conj().T
                P23 = U2_23[:, :m] @ U2_23[:, :m].conj().T
                P123 = U3[:, :m] @ U3[:, :m].conj().T

                for i, psi3 in enumerate(psi3s):
                    batch_raw_12[b, i] += projected_trace_pure_2(psi1, psi2, P12)
                    batch_raw_13[b, i] += projected_trace_pure_2(psi1, psi3, P13)
                    batch_raw_23[b, i] += projected_trace_pure_2(psi2, psi3, P23)
                    batch_raw_123[b, i] += projected_trace_pure_123(
                        psi1, psi2, psi3, P123
                    )

            print(
                f"  batch {b + 1}/{args.num_batches} "
                f"({(b + 1) * batch_size}/{N} executions per estimator)"
            )

        raw12 = batch_raw_12 / batch_size
        raw13 = batch_raw_13 / batch_size
        raw23 = batch_raw_23 / batch_size
        raw123 = batch_raw_123 / batch_size

        pair12 = reconstruct_pair_overlap(raw12, d, m)
        pair13 = reconstruct_pair_overlap(raw13, d, m)
        pair23 = reconstruct_pair_overlap(raw23, d, m)

        Delta2Sigma_batches = (pair12 + pair13 + pair23).real

        Delta123_batches = np.zeros_like(raw123)
        for b in range(args.num_batches):
            for i in range(args.num_phi):
                Delta123_batches[b, i] = reconstruct_Delta123_pure(
                    raw_mean_123=raw123[b, i],
                    Delta2Sigma_hat=Delta2Sigma_batches[b, i],
                    d=d,
                    m=m,
                )

        Delta2Sigma_hat_mean[q_keep] = np.mean(Delta2Sigma_batches, axis=0)
        Delta2Sigma_hat_stderr[q_keep] = stderr_from_batches(Delta2Sigma_batches)

        Delta123_hat_batches[q_keep] = Delta123_batches
        Delta123_hat_mean[q_keep] = np.mean(Delta123_batches, axis=0)
        Delta123_hat_stderr[q_keep] = stderr_from_batches(Delta123_batches)

    plt.figure(figsize=(6.8, 6.4))
    plt.plot(
        Delta123_exact.real,
        Delta123_exact.imag,
        "--",
        linewidth=3,
        label=r"exact $\Delta_{123}$",
    )

    for q_keep in q_values:
        z = Delta123_hat_mean[q_keep]
        err = Delta123_hat_stderr[q_keep]
        step = max(1, args.num_phi // 30)

        # plt.errorbar(
        #     z.real[::step],
        #     z.imag[::step],
        #     xerr=err[::step],
        #     yerr=err[::step],
        #     fmt="o",
        #     markersize=3.0,
        #     capsize=2,
        #     alpha=0.8,
        #     label=fr"$q={q_keep}$",
        # )
        plt.errorbar(
            z.real[::step],
            z.imag[::step],
            xerr=err[::step],
            yerr=err[::step],
            fmt=MARKERS_Q[q_keep],
            ms=5.0,
            mfc="white",
            mec=COLORS_Q[q_keep],
            mew=1.6,
            color=COLORS_Q[q_keep],
            alpha=0.8,
            capsize=2,
            label=fr"$q={q_keep}$",
        )

    plt.xlabel(r"$\mathrm{Re}\,\Delta_{123}$")
    plt.ylabel(r"$\mathrm{Im}\,\Delta_{123}$")
    plt.axis("equal")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_complex.pdf", dpi=300)

    plt.figure(figsize=(7.2, 4.6))
    for q_keep in q_values:
        z = Delta123_hat_mean[q_keep]
        abs_err = np.abs(z - Delta123_exact)

        batch_abs_err = np.abs(
            Delta123_hat_batches[q_keep] - Delta123_exact[None, :]
        )
        abs_err_stderr = stderr_from_batches(batch_abs_err)

        plt.errorbar(
            phis,
            abs_err,
            yerr=abs_err_stderr,
            fmt="o-",
            markersize=3.0,
            capsize=2,
            label=fr"$q={q_keep}$",
        )

    plt.yscale("log")
    plt.xlabel(r"Azimuthal angle $\phi$")
    plt.ylabel(r"$|\widehat{\Delta}_{123}-\Delta_{123}|$")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_error.png", dpi=300)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(
        phis,
        np.unwrap(np.angle(Delta123_exact)),
        "-",
        linewidth=3,
        label=r"Exact $\arg\Delta_{123}$",
    )

    for q_keep in q_values:
        phase_batches = np.unwrap(np.angle(Delta123_hat_batches[q_keep]), axis=1)
        phase_mean = np.mean(phase_batches, axis=0)
        phase_stderr = stderr_from_batches(phase_batches)

        plt.plot(phis, phase_mean, "o", markersize=3.0, alpha=0.75, label=fr"$q={q_keep}$")
        plt.fill_between(
            phis,
            phase_mean - phase_stderr,
            phase_mean + phase_stderr,
            alpha=0.18,
        )

    plt.xlabel(r"Azimuthal angle $\phi$")
    plt.ylabel(r"$\arg \Delta_{123}$")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_phase.png", dpi=300)

    np.savez(
        f"{args.output_prefix}_data.npz",
        n=n,
        d=d,
        theta=args.theta,
        phis=phis,
        q_values=np.array(q_values),
        N=N,
        Nprime=Nprime,
        num_batches=args.num_batches,
        batch_size=batch_size,
        Delta123_exact=Delta123_exact,
        Delta132_exact=Delta132_exact,
        Delta2Sigma_exact=Delta2Sigma_exact_vals,
        **{f"Delta123_hat_mean_q{q}": Delta123_hat_mean[q] for q in q_values},
        **{f"Delta123_hat_stderr_q{q}": Delta123_hat_stderr[q] for q in q_values},
        **{f"Delta2Sigma_hat_mean_q{q}": Delta2Sigma_hat_mean[q] for q in q_values},
        **{f"Delta2Sigma_hat_stderr_q{q}": Delta2Sigma_hat_stderr[q] for q in q_values},
    )

    print("\nSaved:")
    print(f"  {args.output_prefix}_complex.pdf")
    print(f"  {args.output_prefix}_error.png")
    print(f"  {args.output_prefix}_phase.png")
    print(f"  {args.output_prefix}_data.npz")
    print(f"\nProtocol executions per estimator: N = {N}")
    print(f"Total copy count: N' = 5N = {Nprime}")


if __name__ == "__main__":
    main()