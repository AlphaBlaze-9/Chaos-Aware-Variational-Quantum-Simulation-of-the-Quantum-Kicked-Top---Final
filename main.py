import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys


_SYS_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
if _SYS_DIR not in sys.path:
    sys.path.insert(0, _SYS_DIR)

from spin_operators import collective_J, hadamard_state
from qkt_quantum import floquet_U, evolve_state, husimi_Q_grid


from otoc import compute_otoc_QKT, save_otoc_plot

from loschmidt import compute_echo_QKT, save_echo_plot

from vqs import vqs_compare_depths, vqs_floquet_two_k


try:
    from classical_kicked_top import (
        CKTopParams,  
        FTLEConfig,
        ftle_scan_over_k,
        poincare_points_for_k,
        first_positive_threshold,
    )
    _HAS_CLASSICAL = True
except Exception as e:
    _HAS_CLASSICAL = False
    _CLASSICAL_IMPORT_ERR = e


def set_seed(seed: int = 1234):
    np.random.seed(seed)


def set_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.grid": True,
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )






def verify_spin(N: int = 4):
    
    Jx, Jy, Jz = collective_J(N)

    def norm(A):
        return np.linalg.norm(A)

    c1 = norm(Jx @ Jy - Jy @ Jx - 1j * Jz)
    c2 = norm(Jy @ Jz - Jz @ Jy - 1j * Jx)
    c3 = norm(Jz @ Jx - Jx @ Jz - 1j * Jy)
    print(
        f"[spin] N={N}  "
        f"||[Jx,Jy]-iJz||={c1:.2e}  "
        f"||[Jy,Jz]-iJx||={c2:.2e}  "
        f"||[Jz,Jx]-iJy||={c3:.2e}"
    )


def qkt_norm_plots(k: float = 2.5, p: float = np.pi / 2, T: int = 60, outdir: str = "figures"):
    
    os.makedirs(outdir, exist_ok=True)
    for N in (6, 8):
        U = floquet_U(N, k=k, p=p)
        psi0 = hadamard_state(N)
        traj = evolve_state(U, psi0, T=T)
        norms = [np.vdot(v, v).real for v in traj]
        plt.figure(figsize=(7.2, 4.0))
        plt.plot(norms, lw=2.0)
        plt.xlabel("$t$", fontsize=16)
        plt.ylabel("norm", fontsize=16)
        plt.tick_params(axis="both", labelsize=14)
        
        plt.tight_layout()
        path = os.path.join(outdir, f"qkt_norm_N{N}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"[saved] {path}")


def husimi_single(
    N: int = 8,
    k: float = 2.5,
    p: float = np.pi / 2,
    t: int = 30,
    n_theta: int = 48,
    n_phi: int = 96,
    outdir: str = "figures",
):
    
    os.makedirs(outdir, exist_ok=True)
    U = floquet_U(N, k=k, p=p)
    psi_t = evolve_state(U, hadamard_state(N), T=t)[-1]
    ths = np.linspace(0.0, np.pi, n_theta)
    phs = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    Q = husimi_Q_grid(psi_t, N, ths, phs)

    plt.figure(figsize=(7.2, 3.6))
    plt.imshow(Q, origin="lower", aspect="auto",
               extent=(phs[0], phs[-1], ths[0], ths[-1]))
    plt.xlabel(r"$\phi$", fontsize=16)
    plt.ylabel(r"$\theta$", fontsize=16)
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    path = os.path.join(outdir, f"husimi_Q_N{N}_k{k:.2f}_t{t}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


def husimi_compare(
    N: int = 8,
    p: float = np.pi / 2,
    t: int = 30,
    n_theta: int = 48,
    n_phi: int = 96,
    outdir: str = "figures",
):
    
    os.makedirs(outdir, exist_ok=True)
    ths = np.linspace(0.0, np.pi, n_theta)
    phs = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)

    def Q_of(k):
        U = floquet_U(N, k=k, p=p)
        psi_t = evolve_state(U, hadamard_state(N), T=t)[-1]
        return husimi_Q_grid(psi_t, N, ths, phs)

    Q_reg = Q_of(0.5)
    Q_cha = Q_of(2.5)

    
    fig, axs = plt.subplots(2, 1, figsize=(5.6, 7.6), constrained_layout=True)

    axs[0].imshow(Q_reg, origin="lower", aspect="auto",
                  extent=(phs[0], phs[-1], ths[0], ths[-1]))
    
    axs[0].set_ylabel(r"$\theta$", fontsize=16)
    axs[0].tick_params(axis="both", labelsize=14)
    
    axs[0].set_xticklabels([])

    im1 = axs[1].imshow(Q_cha, origin="lower", aspect="auto",
                        extent=(phs[0], phs[-1], ths[0], ths[-1]))
    axs[1].set_xlabel(r"$\phi$", fontsize=16)
    axs[1].set_ylabel(r"$\theta$", fontsize=16)
    axs[1].tick_params(axis="both", labelsize=14)

    
    cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label(r"Husimi $Q(\theta,\phi)$", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    path = os.path.join(outdir, f"husimi_compare_N{N}_t{t}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")






def classical_lyapunov_scan_and_plot(
    kmin: float,
    kmax: float,
    numk: int,
    p: float,
    outdir: str,
    steps: int = 1500,
    burn_in: int = 150,
    n_seeds: int = 8,
):
    if not _HAS_CLASSICAL:
        raise RuntimeError(f"classical_kicked_top import failed: {_CLASSICAL_IMPORT_ERR}")
    os.makedirs(outdir, exist_ok=True)
    ks = np.linspace(kmin, kmax, numk)
    cfg = FTLEConfig(steps=steps, burn_in=burn_in, delta0=1e-6, rescale_each=1)
    results = ftle_scan_over_k(ks, p=p, n_seeds=n_seeds, ftle_cfg=cfg)

    
    csv_path = os.path.join(outdir, "ftle_vs_k.csv")
    import csv
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
    fig_path = os.path.join(outdir, "ftle_vs_k.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    
    k_star = first_positive_threshold(results["k"], results["mean_ftle"], smooth=3)
    txt_path = os.path.join(outdir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Estimated first k where mean FTLE > 0 (smoothed): {k_star}\n")
        f.write(f"Parameters: p={p}, steps={steps}, burn-in={burn_in}, n_seeds={n_seeds}\n")

    print(f"[saved] {fig_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {txt_path}")


def classical_portraits(
    k_list,
    p: float,
    outdir: str,
    n_ic: int = 64,
    steps: int = 1200,
    discard: int = 200,
):
    
    if not _HAS_CLASSICAL:
        raise RuntimeError(f"classical_kicked_top import failed: {_CLASSICAL_IMPORT_ERR}")
    os.makedirs(outdir, exist_ok=True)
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
        path = os.path.join(outdir, f"poincare_k_{k:.3f}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print(f"[saved] {path}")
    return saved


if __name__ == "__main__":
    set_seed(1234)
    set_plot_style()

    ap = argparse.ArgumentParser(description="QKT / CKT runner")
    
    ap.add_argument("--verify-spin", action="store_true", help="check [Jx,Jy]=iJz (cyclic) at N=4")
    ap.add_argument("--qkt", action="store_true", help="evolve |+>^N and plot norm stability for N=6,8")
    ap.add_argument("--husimi", action="store_true", help="single Husimi-Q snapshot (defaults N=8,k=2.5,t=30)")
    ap.add_argument("--husimi-compare", action="store_true", help="VERTICAL Husimi-Q for k=0.5 (top) vs 2.5 (bottom)")

    
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--k", type=float, default=2.5)
    ap.add_argument("--p", type=float, default=np.pi / 2)
    ap.add_argument("--t", type=int, default=30)
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--outdir", type=str, default="figures")

    
    ap.add_argument("--classical-scan", action="store_true", help="Run classical FTLE scan vs k and save plot/CSV")
    ap.add_argument("--classical-portraits", action="store_true", help="Generate classical Poincaré portraits")

    
    ap.add_argument("--kmin", type=float, default=0.5)
    ap.add_argument("--kmax", type=float, default=4.0)
    ap.add_argument("--numk", type=int, default=36)
    ap.add_argument("--nseeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--burnin", type=int, default=150)

    
    ap.add_argument("--portraits-k", type=str, default="0.5,1.0,1.5,2.0,3.0,4.0")
    ap.add_argument("--portrait-nic", type=int, default=64)
    ap.add_argument("--portrait-steps", type=int, default=1200)
    ap.add_argument("--portrait-discard", type=int, default=200)

    
    ap.add_argument("--otoc", action="store_true",
                    help="Compute OTOC vs time for the current k,p (single curve)")
    ap.add_argument("--otoc-two", action="store_true",
                    help="Compute OTOC vs time for k=0.5 and k=2.5 (comparison)")
    ap.add_argument("--otoc-mode", type=str, default="trace",
                    choices=["trace", "state", "echo"],
                    help="OTOC evaluation mode: trace (∞-temp), state, or echo")
    ap.add_argument("--otoc-T", type=int, default=60, help="OTOC horizon (time steps)")
    ap.add_argument("--otoc-W", type=str, default="X1", help="Local operator W spec, e.g., X1")
    ap.add_argument("--otoc-V", type=str, default="Z2", help="Local operator V spec, e.g., Z2")

    
    ap.add_argument("--echo", action="store_true",
                    help="Compute Loschmidt echo L(t) for current k,p")
    ap.add_argument("--echo-two", action="store_true",
                    help="Compare echo for k=0.5 and k=2.5 on one plot")
    ap.add_argument("--echo-T", type=int, default=60, help="Echo horizon (time steps)")
    ap.add_argument("--echo-rel-dk", type=float, default=0.01,
                    help="Relative perturbation: k'=(1+rel_dk)*k (default 1%)")

    
    ap.add_argument("--vqs", action="store_true",
                    help="Run VQS McLachlan with hardware-efficient ansatz; compare depths")
    ap.add_argument("--vqs-depths", type=str, default="1,2,3",
                    help="Comma-separated depths to compare, e.g. 1,2,3")
    ap.add_argument("--vqs-steps", type=int, default=200, help="Number of VQS time steps")
    ap.add_argument("--vqs-dt", type=float, default=0.05, help="Time step dt for VQS ODE")
    ap.add_argument("--vqs-reg", type=float, default=1e-6, help="Regularization for A matrix")

    
    ap.add_argument("--vqs-floquet", action="store_true",
                    help="Run VQS on Floquet H_eff for k=0.5 and k=2.5 at fixed depth")
    ap.add_argument("--vqs-floquet-k1", type=float, default=0.5,
                    help="First k (regular regime) for Floquet VQS")
    ap.add_argument("--vqs-floquet-k2", type=float, default=2.5,
                    help="Second k (chaotic regime) for Floquet VQS")
    ap.add_argument("--vqs-floquet-D", type=int, default=2,
                    help="Ansatz depth D for Floquet VQS benchmark")

    args = ap.parse_args()

    
    if args.verify_spin:
        verify_spin(N=4)
    if args.qkt:
        qkt_norm_plots(k=args.k, p=args.p, T=args.T, outdir=args.outdir)
    if args.husimi:
        husimi_single(N=args.N, k=args.k, p=args.p, t=args.t, outdir=args.outdir)
    if args.husimi_compare:
        husimi_compare(N=args.N, p=args.p, t=args.t, outdir=args.outdir)

    
    if args.classical_scan:
        classical_lyapunov_scan_and_plot(
            args.kmin,
            args.kmax,
            args.numk,
            p=args.p,
            outdir=args.outdir,
            steps=args.steps,
            burn_in=args.burnin,
            n_seeds=args.nseeds,
        )
    if args.classical_portraits:
        k_list = [float(x) for x in args.portraits_k.replace(" ", "").split(",") if x]
        classical_portraits(
            k_list,
            p=args.p,
            outdir=args.outdir,
            n_ic=args.portrait_nic,
            steps=args.portrait_steps,
            discard=args.portrait_discard,
        )

    
    if args.otoc:
        res = compute_otoc_QKT(
            N=args.N, k=args.k, p=args.p, T=args.otoc_T,
            W_spec=args.otoc_W, V_spec=args.otoc_V, mode=args.otoc_mode
        )
        out_png = os.path.join(args.outdir, f"otoc_k_{args.k:.2f}_{args.otoc_mode}.png")
        os.makedirs(args.outdir, exist_ok=True)
        save_otoc_plot(
            [res],
            title=f"OTOC vs time (N={args.N}, p={args.p:.3f}, W={args.otoc_W}, V={args.otoc_V})",
            out_png=out_png,
        )
        print(f"[saved] {out_png}")

    if args.otoc_two:
        res1 = compute_otoc_QKT(
            N=args.N, k=0.5, p=args.p, T=args.otoc_T,
            W_spec=args.otoc_W, V_spec=args.otoc_V, mode=args.otoc_mode
        )
        res2 = compute_otoc_QKT(
            N=args.N, k=2.5, p=args.p, T=args.otoc_T,
            W_spec=args.otoc_W, V_spec=args.otoc_V, mode=args.otoc_mode
        )
        out_png = os.path.join(args.outdir, f"otoc_compare_{args.otoc_mode}.png")
        os.makedirs(args.outdir, exist_ok=True)
        save_otoc_plot(
            [res1, res2],
            title=f"OTOC (N={args.N}, p={args.p:.3f}, W={args.otoc_W}, V={args.otoc_V})",
            out_png=out_png,
        )
        print(f"[saved] {out_png}")

    
    if args.echo:
        res = compute_echo_QKT(
            N=args.N, k=args.k, p=args.p, T=args.echo_T, rel_delta_k=args.echo_rel_dk
        )
        os.makedirs(args.outdir, exist_ok=True)
        out_png = os.path.join(args.outdir, f"echo_k_{args.k:.2f}_reldk_{args.echo_rel_dk:.3f}.png")
        save_echo_plot([res],
                       out_png=out_png,
                       title=f"Loschmidt Echo (N={args.N}, p={args.p:.3f})")
        print(f"[saved] {out_png}")

    if args.echo_two:
        res_reg = compute_echo_QKT(
            N=args.N, k=0.5, p=args.p, T=args.echo_T, rel_delta_k=args.echo_rel_dk
        )
        res_cha = compute_echo_QKT(
            N=args.N, k=2.5, p=args.p, T=args.echo_T, rel_delta_k=args.echo_rel_dk
        )
        os.makedirs(args.outdir, exist_ok=True)
        out_png = os.path.join(args.outdir, f"echo_compare_reldk_{args.echo_rel_dk:.3f}.png")
        save_echo_plot([res_reg, res_cha],
                       out_png=out_png,
                       title=f"Loschmidt Echo (N={args.N}, p={args.p:.3f})")
        print(f"[saved] {out_png}")

    
    if args.vqs:
        depths = [int(x) for x in args.vqs_depths.replace(" ", "").split(",") if x]
        out_png = os.path.join(args.outdir, "vqs_fidelity_compare.png")
        vqs_compare_depths(
            N=args.N, k=args.k, p=args.p,
            depths=depths,
            steps=args.vqs_steps,
            dt=args.vqs_dt,
            reg=args.vqs_reg,
            out_png=out_png,
        )
        print(f"[saved] {out_png}")

    
    if args.vqs_floquet:
        out_png = os.path.join(args.outdir, "vqs_floquet_k_compare.png")
        vqs_floquet_two_k(
            N=args.N,
            p=args.p,
            k1=args.vqs_floquet_k1,
            k2=args.vqs_floquet_k2,
            D=args.vqs_floquet_D,
            steps=args.vqs_steps,
            dt=args.vqs_dt,
            reg=args.vqs_reg,
            out_png=out_png,
        )
        print(f"[saved] {out_png}")

    if not any(
        [
            args.verify_spin,
            args.qkt,
            args.husimi,
            args.husimi_compare,
            args.classical_scan,
            args.classical_portraits,
            args.otoc,
            args.otoc_two,
            args.echo,
            args.echo_two,
            args.vqs,
            args.vqs_floquet,
        ]
    ):
        print(
            "Nothing to do. Try flags like: --verify-spin, --qkt, --husimi, --husimi-compare, "
            "--classical-scan, --classical-portraits, --otoc, --otoc-two, --echo, --echo-two, "
            "--vqs, --vqs-floquet"
        )
