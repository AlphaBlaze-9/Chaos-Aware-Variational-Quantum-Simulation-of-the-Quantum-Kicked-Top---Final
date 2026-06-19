

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from classical_kicked_top import (
    CKTopParams,
    FTLEConfig,
    ftle_scan_over_k,
    poincare_points_for_k,
    first_positive_threshold,
)


def set_plot_style():
    
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.grid": False,
        }
    )


def lyapunov_scan_and_plot(
    kmin: float,
    kmax: float,
    numk: int,
    p: float,
    outdir: str,
    steps: int = 1500,
    burn_in: int = 150,
    n_seeds: int = 8,
):
    set_plot_style()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    ks = np.linspace(kmin, kmax, numk)

    cfg = FTLEConfig(steps=steps, burn_in=burn_in, delta0=1e-6, rescale_each=1)
    results = ftle_scan_over_k(ks, p=p, n_seeds=n_seeds, ftle_cfg=cfg)

    
    csv_path = Path(outdir) / "ftle_vs_k.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "mean_ftle", "std_ftle", "p", "steps", "burn_in", "n_seeds"])
        for k, m, s in zip(results["k"], results["mean_ftle"], results["std_ftle"]):
            w.writerow([k, m, s, p, steps, burn_in, n_seeds])

    
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(results["k"], results["mean_ftle"], label="Mean FTLE",
            linewidth=2.0, color="C0")
    ax.fill_between(
        results["k"],
        results["mean_ftle"] - results["std_ftle"],
        results["mean_ftle"] + results["std_ftle"],
        alpha=0.25,
        label=r"$\pm 1\sigma$",
        color="C0",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1, color="k", alpha=0.6)
    ax.set_xlabel("$k$ (kick strength)", fontsize=16)
    ax.set_ylabel("Finite-time Lyapunov exponent", fontsize=16)
    
    ax.legend(fontsize=13)
    ax.tick_params(axis="both", labelsize=14)
    fig.tight_layout()
    fig_path = Path(outdir) / "ftle_vs_k.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    
    k_star = first_positive_threshold(results["k"], results["mean_ftle"], smooth=3)
    txt_path = Path(outdir) / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            f"Estimated first k where mean FTLE > 0 (smoothed): {k_star}\n"
        )
        f.write(
            f"Parameters: p={p}, steps={steps}, burn-in={burn_in}, n_seeds={n_seeds}\n"
        )

    return {
        "csv": str(csv_path),
        "png": str(fig_path),
        "k_star": k_star,
        "k": results["k"],
        "mean_ftle": results["mean_ftle"],
        "std_ftle": results["std_ftle"],
    }


def phasespace_portraits(
    k_list,
    p: float,
    outdir: str,
    n_ic: int = 64,
    steps: int = 1200,
    discard: int = 200,
):
    
    set_plot_style()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    saved = []
    for k in k_list:
        phi, z = poincare_points_for_k(k, p, n_ic=n_ic, steps=steps, discard=discard)
        fig = plt.figure(figsize=(7.2, 4.8))
        ax = fig.add_subplot(111)
        ax.scatter(phi, z, s=1.0, linewidths=0.0)
        ax.set_xlabel(r"$\phi$ (radians)", fontsize=16)
        ax.set_ylabel(r"$z = \cos\theta$", fontsize=16)
        ax.set_xlim(0.0, 2.0 * np.pi)
        ax.set_ylim(-1.0, 1.0)
        ax.tick_params(axis="both", labelsize=14)
        
        fig.tight_layout()
        path = Path(outdir) / f"poincare_k_{k:.3f}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    return saved


def main():
    ap = argparse.ArgumentParser(description="Classical kicked top analysis")
    ap.add_argument(
        "--p", type=float, default=np.pi / 2, help="Rotation angle about y (radians)"
    )
    ap.add_argument("--kmin", type=float, default=0.5)
    ap.add_argument("--kmax", type=float, default=4.0)
    ap.add_argument("--numk", type=int, default=40)
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument(
        "--scan-only", action="store_true", help="Only run Lyapunov scan"
    )
    ap.add_argument(
        "--portraits-only", action="store_true", help="Only run phase-space portraits"
    )
    ap.add_argument("--nseeds", type=int, default=8, help="FTLE seeds per k")
    ap.add_argument("--steps", type=int, default=1500, help="FTLE steps per seed")
    ap.add_argument("--burnin", type=int, default=150, help="FTLE burn-in steps")
    ap.add_argument(
        "--portraits-k",
        type=str,
        default="0.5,1.0,1.5,2.0,3.0,4.0",
        help="Comma-separated list of k values for portraits",
    )
    ap.add_argument("--portrait-nic", type=int, default=64)
    ap.add_argument("--portrait-steps", type=int, default=1200)
    ap.add_argument("--portrait-discard", type=int, default=200)

    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    
    if not args.portraits_only:
        scan = lyapunov_scan_and_plot(
            args.kmin,
            args.kmax,
            args.numk,
            p=args.p,
            outdir=args.outdir,
            steps=args.steps,
            burn_in=args.burnin,
            n_seeds=args.nseeds,
        )
        print("Lyapunov scan saved:", scan["png"], "and", scan["csv"])

    
    if not args.scan_only:
        k_list = [float(x) for x in args.portraits_k.replace(" ", "").split(",")]
        saved = phasespace_portraits(
            k_list,
            p=args.p,
            outdir=args.outdir,
            n_ic=args.portrait_nic,
            steps=args.portrait_steps,
            discard=args.portrait_discard,
        )
        print("Saved portraits:")
        for pth in saved:
            print("  ", pth)


if __name__ == "__main__":
    main()
