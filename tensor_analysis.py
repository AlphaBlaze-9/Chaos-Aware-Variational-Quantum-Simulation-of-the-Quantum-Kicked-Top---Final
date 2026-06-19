

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from spin_operators import coherent_product_state
from vqs import Ansatz, mclachlan_AC, condition_number, adaptive_ridge, \
    floquet_step_generator


def condition_vs_depth(N: int, k: float, depths, p: float = np.pi / 2,
                       seed: int = 0, target_cond: float = 1e8):
    
    rng = np.random.default_rng(seed)
    H = floquet_step_generator(N, k, p)
    psi0 = coherent_product_state(N)
    raw, reg, ridges = [], [], []
    for D in depths:
        ans = Ansatz(N, D)
        theta = rng.uniform(0.0, 2 * np.pi, ans.n_params)
        A, _, _, _ = mclachlan_AC(ans, theta, psi0, H)
        r = adaptive_ridge(A, target_cond=target_cond)
        raw.append(condition_number(A, 0.0))
        reg.append(condition_number(A, r))
        ridges.append(r)
    return np.array(raw), np.array(reg), np.array(ridges)


def tikhonov_solve(A, C, ridge):
    
    return np.linalg.solve(A + ridge * np.eye(A.shape[0]), C)


def _aps_style():
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 600,
    })


def plot_condition(N=4, k=2.5, depths=range(1, 7),
                   outfile="figures/metric_condition"):
    import os
    depths = list(depths)
    raw, reg, ridges = condition_vs_depth(N, k, depths)
    _aps_style()
    fig, ax = plt.subplots(figsize=(3.375, 2.7), constrained_layout=True)
    ax.semilogy(depths, raw, "o-", color="#D55E00", label=r"unregularized $\kappa(A)$")
    ax.semilogy(depths, reg, "s-", color="#0072B2", label=r"after Tikhonov")
    ax.set_xlabel(r"Circuit depth $D$")
    ax.set_ylabel(r"Condition number $\kappa(A)$")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best")
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}.pdf / {outfile}.png")
    for D, rr, rg, rd in zip(depths, raw, reg, ridges):
        print(f"  D={D}: kappa_raw={rr:.2e}  ridge={rd:.1e}  kappa_reg={rg:.2e}")


if __name__ == "__main__":
    plot_condition()
