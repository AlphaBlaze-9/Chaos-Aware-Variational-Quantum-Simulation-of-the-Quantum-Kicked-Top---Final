"""
fig7_fidelity_compare.py  --  REPLACEMENT generator for Fig. 7 (vqs_fidelity_compare).

WHY THIS FILE EXISTS / WHAT CHANGED (review item A1)
----------------------------------------------------
1) The original `vqs_compare_depths` that `main.py --vqs` imported from `vqs.py`
   is no longer in the repository, so Fig. 7 could not be regenerated.

2) IMPORTANT: switching the McLachlan *integrator* to RK4 does NOT fix Fig. 7.
   Direct test (see RESULTS_AND_TEX_EDITS.md) shows the integrator collapses in
   BOTH regimes -- e.g. regular k=0.5, D=1 gives F = 0.47, 0.04, 0.37, ... and
   gets worse with depth. This is the same behavior `observable_jz2.py` documents
   and is why that figure already uses direct optimization.

3) This script therefore generates Fig. 7 with DIRECT L-BFGS-B OPTIMIZATION at
   each fixed depth -- the same engine used for Result 1 (Figs. 10, 11) and for
   Fig. 12. It reproduces the manuscript narrative correctly:
      regular  -> F ~ 0.96-1.0 at all depths (near-unit, matches the text)
      chaotic  -> F collapses at D=1, recovers at D=2,3.

ACTION AFTER RUNNING: update the Fig. 7 caption to say the curves are
"best fixed-depth direct-optimization fidelity" and DROP the
"unconverged forward-Euler integrator / illustrative only" disclaimer.
(See RESULTS_AND_TEX_EDITS.md for the exact caption replacement.)

Pure NumPy/SciPy. Run:  python fig7_fidelity_compare.py
"""
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from spin_operators import coherent_product_state, normalize
from qkt_quantum import floquet_U_exact
from vqs import Ansatz


def best_fixed_depth_fidelity(N, k, depth, steps=12, p=np.pi / 2,
                              n_restarts=20, seed=0):
    """For fixed depth, best achievable F(t)=|<U_F^t psi0 | psi_var>|^2 via
    direct L-BFGS-B optimization (multi-restart)."""
    psi0 = coherent_product_state(N)
    U_F = floquet_U_exact(N, k, p)
    ans = Ansatz(N, depth)
    fids, psi_exact = [], psi0.copy()
    for t in range(steps):
        psi_exact = normalize(U_F @ psi_exact)

        def cost(th):
            return 1.0 - abs(np.vdot(psi_exact, ans.state(th, psi0))) ** 2

        best = -np.inf
        for r in range(n_restarts):
            rng = np.random.default_rng((seed, t, r))
            x0 = rng.uniform(0, 2 * np.pi, ans.n_params)
            res = minimize(cost, x0, method="L-BFGS-B", options={"maxiter": 300})
            best = max(best, 1.0 - res.fun)
        fids.append(float(best))
    return np.array(fids)


def make_figure(N=6, steps=12, depths=(1, 2, 3),
                outfile="figures/vqs_fidelity_compare"):
    os.makedirs("figures", exist_ok=True)
    t = np.arange(1, steps + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    colors = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
    for col, (k, label) in enumerate([(0.5, "Regular ($k=0.5$)"),
                                      (2.5, "Chaotic ($k=2.5$)")]):
        ax = axes[col]
        for D in depths:
            f = best_fixed_depth_fidelity(N, k, D, steps=steps)
            ax.plot(t, f, "o-", color=colors[D], label=f"$D={D}$", ms=4)
            print(f"  N={N} k={k} D={D}: F(1)={f[0]:.3f}  F(end)={f[-1]:.3f}  "
                  f"mean={f.mean():.3f}")
        ax.set_xlabel("Floquet step $t$")
        ax.set_title(f"({'ab'[col]}) {label}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("Fidelity $F(t)$")
        ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(outfile + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outfile}.png / .pdf")


if __name__ == "__main__":
    print("Generating Fig. 7 (vqs_fidelity_compare) via direct optimization "
          "(converged, integrator-free)...")
    make_figure(N=6, steps=12)
