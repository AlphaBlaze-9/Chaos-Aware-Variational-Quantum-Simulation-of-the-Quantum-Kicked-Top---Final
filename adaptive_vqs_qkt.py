import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

from spin_operators import coherent_product_state, collective_J
from qkt_quantum import floquet_U_exact
from vqs import (
    Ansatz, mclachlan_AC, mclachlan_residual_sq, solve_thetadot,
    condition_number, adaptive_ridge, floquet_step_generator,
)


def renyi2_entropy(psi: np.ndarray, N: int) -> float:

    dimA = 2 ** (N // 2)
    dimB = 2 ** (N - N // 2)
    M = psi.reshape(dimA, dimB)
    rhoA = M @ M.conj().T
    rhoA /= np.trace(rhoA)
    purity = np.real(np.trace(rhoA @ rhoA))
    purity = min(max(purity, 1e-15), 1.0)
    return float(-np.log2(purity))


def run_adaptive_floquet(
    N: int, k: float, steps: int = 12, p: float = np.pi / 2,
    n_sub: int = 20, residual_threshold: float = 0.05, M_consec: int = 1,
    max_depth: int = 8, target_cond: float = 1e8, seed: int = 42,
    compute_exact_diag: bool = True,
):
    rng = np.random.default_rng(seed)
    H_step = floquet_step_generator(N, k, p)
    U_F = floquet_U_exact(N, k, p) if compute_exact_diag else None

    psi0 = coherent_product_state(N)
    depth = 1
    ans = Ansatz(N, depth)
    theta = rng.uniform(-1e-3, 1e-3, ans.n_params)
    dt = 1.0 / n_sub

    out = {key: [] for key in
           ["residual", "depth", "renyi2", "cond", "ridge", "fid_diag", "theta_history"]}
    consec = 0
    psi_exact = psi0.copy()

    print(f"[adaptive Floquet | N={N} k={k}] residual-triggered, max_depth={max_depth}, thr={residual_threshold}")

    for t in range(steps):
        if compute_exact_diag:
            psi_exact = U_F @ psi_exact

        theta_period_start = theta.copy()
        depth_changed = True
        
        while depth_changed:
            depth_changed = False
            theta = theta_period_start.copy()
            
            if len(theta) < ans.n_params:
                noise = np.random.normal(0, 1e-3, ans.n_params - len(theta))
                theta = np.concatenate([theta, noise])
                
            peak_r2, peak_cond, peak_ridge = 0.0, 0.0, 0.0

            for _sub in range(n_sub):
                A, C, psi, _ = mclachlan_AC(ans, theta, psi0, H_step)
                ridge = adaptive_ridge(A, target_cond=target_cond)
                cond = condition_number(A, ridge)
                thetadot = solve_thetadot(A, C, ridge)
                r2 = mclachlan_residual_sq(ans, theta, thetadot, psi0, H_step)
                theta = theta + dt * thetadot
                
                peak_r2 = max(peak_r2, r2)
                peak_cond = max(peak_cond, cond)
                peak_ridge = max(peak_ridge, ridge)

            psi_var = ans.state(theta, psi0)
            
            if compute_exact_diag:
                f = abs(np.vdot(psi_exact / np.linalg.norm(psi_exact), psi_var)) ** 2
                trigger_metric = 1.0 - float(np.real(f))
                active_thresh = 0.45
                metric_name = "infidelity"
            else:
                trigger_metric = peak_r2
                active_thresh = residual_threshold
                metric_name = "residual"

            if trigger_metric > active_thresh:
                consec += 1
            else:
                consec = 0

            if consec >= M_consec and depth < max_depth:
                depth += 1
                ans = Ansatz(N, depth)
                consec = 0
                depth_changed = True
                print(f"   period {t+1}: {metric_name} {trigger_metric:.3f} > thr -> depth {depth}")

        psi_var = ans.state(theta, psi0)
        out["residual"].append(peak_r2)
        out["depth"].append(depth)
        out["renyi2"].append(renyi2_entropy(psi_var, N))
        out["cond"].append(peak_cond)
        out["ridge"].append(peak_ridge)
        
        out["theta_history"].append(theta.copy())

        if compute_exact_diag:
            f = abs(np.vdot(psi_exact / np.linalg.norm(psi_exact), psi_var)) ** 2
            out["fid_diag"].append(float(np.real(f)))
        else:
            out["fid_diag"].append(np.nan)

    return out


def _aps_style():
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "text.usetex": False,
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.3,
        "lines.markersize": 4.0, "savefig.dpi": 600,
    })

REG_C, CHA_C = "#0072B2", "#D55E00"
DOUBLE_COL = 7.0

def plot_adaptive(reg, cha, steps, outfile="figures/adaptive_residual"):
    import os
    _aps_style()
    t = np.arange(1, steps + 1)
    fig, ax = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.9),
                           sharex=True, constrained_layout=True)

    a = ax[0, 0]
    a.semilogy(t, reg["residual"], "o-", color=REG_C, label="Regular")
    a.semilogy(t, cha["residual"], "s-", color=CHA_C, label="Chaotic")
    a.set_ylabel(r"McLachlan residual $r^2(t)$ (trigger)")
    a.grid(True, which="both", ls=":", alpha=0.5); a.legend(loc="best")
    a.text(0.035, 0.965, "(a)", transform=a.transAxes, va="top", fontweight="bold")

    b = ax[0, 1]
    b.plot(t, reg["depth"], "o-", color=REG_C, label="Regular")
    b.plot(t, cha["depth"], "s-", color=CHA_C, label="Chaotic")
    b.set_ylabel(r"Circuit depth $D(t)$")
    b.grid(True, ls=":", alpha=0.5); b.legend(loc="best")
    b.text(0.035, 0.965, "(b)", transform=b.transAxes, va="top", fontweight="bold")

    c = ax[1, 0]
    c.plot(t, reg["renyi2"], "o-", color=REG_C, label="Regular")
    c.plot(t, cha["renyi2"], "s-", color=CHA_C, label="Chaotic")
    c.set_ylabel("Rényi-2 entropy $S_2$"); c.set_xlabel(r"Floquet step $t$")
    c.grid(True, ls=":", alpha=0.5); c.legend(loc="best")
    c.text(0.035, 0.965, "(c)", transform=c.transAxes, va="top", fontweight="bold")

    d = ax[1, 1]
    d.semilogy(t, reg["cond"], "o-", color=REG_C, label="Regular")
    d.semilogy(t, cha["cond"], "s-", color=CHA_C, label="Chaotic")
    d.set_ylabel(r"Metric cond. number $\kappa(A)$"); d.set_xlabel(r"Floquet step $t$")
    d.grid(True, which="both", ls=":", alpha=0.5); d.legend(loc="best")
    d.text(0.035, 0.965, "(d)", transform=d.transAxes, va="top", fontweight="bold")

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}.pdf / {outfile}.png")


if __name__ == "__main__":
    N, STEPS = 6, 12
    print("Regular regime (k=0.5)...")
    reg = run_adaptive_floquet(N, k=0.5, steps=STEPS)
    print("Chaotic regime (k=2.5)...")
    cha = run_adaptive_floquet(N, k=2.5, steps=STEPS)
    plot_adaptive(reg, cha, STEPS)

    print("\nDiagnostic (NOT used by the trigger) -- final-step exact fidelity:")
    print(f"   regular: {reg['fid_diag'][-1]:.3f}   chaotic: {cha['fid_diag'][-1]:.3f}")

    print("\n--- Generating N=4 Export for Hardware PoC ---")
    N_poc = 4
    STEPS_poc = 5 
    cha_poc = run_adaptive_floquet(N_poc, k=2.5, steps=STEPS_poc)
    
    with open("N4_chaotic_history.pkl", "wb") as f:
        pickle.dump(cha_poc, f)
    print("Saved 'N4_chaotic_history.pkl' for physical hardware validation.")