import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from vqs import Ansatz
from spin_operators import coherent_product_state, normalize
from qkt_quantum import floquet_U_exact


# =====================================================================
# D_max vs system size, via DIRECT optimization.
#
# IMPORTANT FIX: the previous version of this script computed D_max with the
# McLachlan *integrator* (expand depth while the residual stays high). That
# integrator does NOT track the Floquet state in this repository -- its
# residual never settles below the 0.05 target even at depth 16, so every
# point just ran to the depth ceiling and the "D_max" it reported was the
# ceiling, not a converged depth. (You can see this in the old console
# output: "after settling: r^2=0.46" at depth 16, still nowhere near 0.05.)
#
# This is the SAME conceptual error that was fixed for Fig. 10: D_max must be
# the minimum depth at which DIRECT optimization can represent U_F^t|psi0> to
# infidelity <= eps_opt, independent of any integrator. This version uses that
# definition, so it is consistent with Fig. 10 and actually converges.
# =====================================================================

EPS_OPT      = 0.05
N_STEPS      = 10          # Floquet steps in the time window t in {1,...,N_STEPS}
K_CHAOTIC    = 2.5
K_REGULAR    = 0.5
N_RESTARTS   = 12
MAX_DEPTH    = 12          # finite ceiling; points that hit it are lower bounds


def dmax_direct(N, k, steps=N_STEPS, eps_opt=EPS_OPT,
                max_depth=MAX_DEPTH, n_restarts=N_RESTARTS):
    psi0 = coherent_product_state(N)
    U_F  = floquet_U_exact(N, k, np.pi / 2)
    fid_target = 1.0 - eps_opt

    dmax = 0
    ceiling_hit = False
    psi_t = psi0.copy()
    for t in range(1, steps + 1):
        psi_t = normalize(U_F @ psi_t)

        # minimum depth that reaches the target infidelity for THIS t
        depth_t, hit_t = max_depth, True
        for D in range(1, max_depth + 1):
            ans = Ansatz(N, D)

            def cost(th):
                psi = ans.state(th, psi0)
                return 1.0 - abs(np.vdot(psi_t, psi)) ** 2

            best_fid = -np.inf
            for r in range(n_restarts):
                rng = np.random.default_rng((N, t, D, r))
                x0 = rng.uniform(0, 2 * np.pi, ans.n_params)
                res = minimize(cost, x0, method="L-BFGS-B",
                               options={"maxiter": 300})
                best_fid = max(best_fid, 1.0 - res.fun)
            if best_fid >= fid_target:
                depth_t, hit_t = D, False
                break

        dmax = max(dmax, depth_t)
        if hit_t:
            ceiling_hit = True
        print(f"   N={N} k={k} t={t:2d}: min sufficient depth={depth_t} "
              f"{'(ceiling, lower bound)' if hit_t else ''}", flush=True)

    return dmax, ceiling_hit


def main():
    system_sizes = [4, 6, 8, 10]
    dmax_chaotic, dmax_regular = [], []
    ceiling_cha, ceiling_reg = [], []

    os.makedirs("figures", exist_ok=True)

    for N in system_sizes:
        print(f"\n{'='*50}")
        print(f"N={N}  |  chaotic (k={K_CHAOTIC})...")
        dmax, hit = dmax_direct(N, K_CHAOTIC)
        print(f"-> D_max={dmax}  ceiling_hit={hit}")
        dmax_chaotic.append(dmax)
        ceiling_cha.append(hit)

        print(f"N={N}  |  regular (k={K_REGULAR})...")
        dmax, hit = dmax_direct(N, K_REGULAR)
        print(f"-> D_max={dmax}  ceiling_hit={hit}")
        dmax_regular.append(dmax)
        ceiling_reg.append(hit)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(system_sizes, dmax_regular, "o-", color="#0072B2",
            label="Regular (k=0.5)")
    for i, hit in enumerate(ceiling_reg):
        if hit:
            ax.plot(system_sizes[i], dmax_regular[i], "o",
                    color="#0072B2", mfc="none", ms=10)

    ax.plot(system_sizes, dmax_chaotic, "s-", color="#D55E00",
            label="Chaotic (k=2.5)")
    for i, hit in enumerate(ceiling_cha):
        if hit:
            ax.plot(system_sizes[i], dmax_chaotic[i], "s",
                    color="#D55E00", mfc="none", ms=10)

    ax.set_xlabel("System Size $N$ (qubits)")
    ax.set_ylabel(r"Maximum Adaptive Depth $D_{\max}$")
    ax.set_title(r"$D_{\max}$ vs. System Size (direct optimization)")
    ax.set_xticks(system_sizes)
    ax.text(0.50, 0.10,
            "open markers: depth ceiling reached\n(lower bound, not converged $D_{\\max}$)",
            transform=ax.transAxes, fontsize=7.5, color="gray")
    ax.grid(True, ls=":")
    ax.legend()
    fig.tight_layout()
    # The manuscript's Fig. 12 includes figures/depth_scaling.png.
    for base in ("figures/depth_scaling", "figures/exact_dmax_scaling"):
        fig.savefig(base + ".pdf")
        fig.savefig(base + ".png", dpi=300)
    plt.close(fig)

    print("\nSaved: figures/depth_scaling.{pdf,png} (and exact_dmax_scaling.*)")
    print("Regular D_max :", dict(zip(system_sizes, dmax_regular)))
    print("Chaotic D_max :", dict(zip(system_sizes, dmax_chaotic)))
    print("[VERIFY] Update the manuscript's depth-scaling discussion "
          "(Sec. III J) with these D_max values. Chaotic should exceed "
          "regular at each N; open markers are lower bounds.")


if __name__ == "__main__":
    main()