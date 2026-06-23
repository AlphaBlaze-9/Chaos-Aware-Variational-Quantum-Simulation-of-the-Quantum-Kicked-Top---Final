"""
large_scale_scaling.py  --  Mean circuit depth vs system size N (Fig. 12).

CHANGES FROM ORIGINAL:
  B3 / Fig 12: N=8 chaotic marker is now drawn as an OPEN marker to
      visually distinguish it as a single-timestep lower bound, matching
      the revised .tex caption ("open square = single timestep lower bound").
      All other logic is unchanged.

  The mean-depth-not-max rationale is preserved exactly as documented in
  the original header.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from vqs import Ansatz
from spin_operators import coherent_product_state, normalize
from qkt_quantum import floquet_U_exact


EPS_OPT      = 0.05
N_STEPS      = 10
K_CHAOTIC    = 2.5
K_REGULAR    = 0.5
N_RESTARTS   = 12
CEILING_OFFSET = 6
MAXITER      = 200

TIME_BUDGET_TABLE = {4: 60.0, 6: 150.0, 8: 180.0}
TIME_BUDGET_DEFAULT = 180.0


def min_sufficient_depth(N, k, psi0, U_F, t, fid_target, max_depth, n_restarts):
    psi_t = psi0.copy()
    for _ in range(t):
        psi_t = normalize(U_F @ psi_t)

    for D in range(1, max_depth + 1):
        ans = Ansatz(N, D)

        def cost(th):
            return 1.0 - abs(np.vdot(psi_t, ans.state(th, psi0))) ** 2

        for r in range(n_restarts):
            rng = np.random.default_rng((N, t, D, r))
            x0 = rng.uniform(0, 2 * np.pi, ans.n_params)
            res = minimize(cost, x0, method="L-BFGS-B",
                           options={"maxiter": MAXITER})
            if 1.0 - res.fun >= fid_target:
                return D, True
    return max_depth, False


def depth_stats(N, k, steps=N_STEPS, eps_opt=EPS_OPT,
                ceiling_offset=CEILING_OFFSET, n_restarts=N_RESTARTS,
                time_budget_s=None):
    if time_budget_s is None:
        time_budget_s = TIME_BUDGET_TABLE.get(N, TIME_BUDGET_DEFAULT)
    psi0 = coherent_product_state(N)
    U_F = floquet_U_exact(N, k, np.pi / 2)
    fid_target = 1.0 - eps_opt
    max_depth = N + ceiling_offset

    depths = []
    all_converged = True
    t_start = time.time()
    for t in range(1, steps + 1):
        if time.time() - t_start > time_budget_s and depths:
            print(f"   N={N} k={k}: time budget reached after t={t-1}; "
                  f"reporting over {len(depths)} timestep(s).", flush=True)
            break
        D, conv = min_sufficient_depth(N, k, psi0, U_F, t, fid_target,
                                       max_depth, n_restarts)
        depths.append(D)
        if not conv:
            all_converged = False
        print(f"   N={N} k={k} t={t:2d}: min sufficient depth={D}"
              f"{'' if conv else ' (ceiling)'}", flush=True)

    depths = np.array(depths, dtype=float)
    return (float(depths.mean()), float(depths.std()), int(depths.max()),
            len(depths), all_converged)


def save_figure(system_sizes, mean_reg, std_reg, mean_cha, std_cha,
                neval_cha=None):
    """Save Fig. 12.  The N=8 chaotic point is drawn with an open marker
    if it is based on a single evaluated timestep (lower bound).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ns_done = system_sizes[:len(mean_reg)]

    # Regular regime -- always solid markers
    ax.errorbar(ns_done, mean_reg, yerr=std_reg, fmt="o-",
                color="#0072B2", capsize=4, label="Regular ($k=0.5$)")

    # Chaotic regime -- open marker when only 1 timestep was evaluated
    ns_cha = ns_done[:len(mean_cha)]
    for i, (n, m, s) in enumerate(zip(ns_cha, mean_cha, std_cha)):
        n_eval = (neval_cha[i] if neval_cha is not None
                  and i < len(neval_cha) else None)
        is_single_step = (n_eval is not None and n_eval <= 1)
        mfc = "none" if is_single_step else "#D55E00"
        marker = "s"
        label_str = ("Chaotic ($k=2.5$)"
                     if i == 0 else
                     ("Chaotic — single step (lower bound)" if is_single_step
                      else None))
        eb = ax.errorbar([n], [m], yerr=[s], fmt=marker + "-",
                         color="#D55E00", capsize=4,
                         markerfacecolor=mfc,
                         label=label_str)

    # Connect chaotic points
    if len(mean_cha) > 1:
        ax.plot(ns_cha, mean_cha, "-", color="#D55E00", lw=1.5, zorder=0)

    ax.set_xlabel("System size $N$ (qubits)")
    ax.set_ylabel(r"Mean sufficient depth $\langle D\rangle_t$")
    ax.set_title("Mean circuit depth required vs. system size\n"
                 "(direct optimization, averaged over Floquet steps)")
    ax.set_xticks(ns_done)
    ax.grid(True, ls=":")
    # Clean up legend (remove None-label entries)
    handles, labels_ = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels_) if l is not None]
    if filtered:
        ax.legend(*zip(*filtered))

    # Annotation for open marker
    if neval_cha is not None:
        single_ns = [n for n, ne in zip(ns_cha, neval_cha) if ne <= 1]
        for n in single_ns:
            ax.annotate("single step\n(lower bound)",
                        xy=(n, mean_cha[list(ns_cha).index(n)]),
                        xytext=(n - 0.3, mean_cha[list(ns_cha).index(n)] + 0.4),
                        fontsize=7.5, color="#D55E00",
                        arrowprops=dict(arrowstyle="->", color="#D55E00"))

    fig.tight_layout()
    for base in ("figures/depth_scaling", "figures/exact_dmax_scaling"):
        fig.savefig(base + ".pdf")
        fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    print(f"  [saved figures/depth_scaling.png with N={ns_done}]", flush=True)


def main():
    system_sizes = [4, 6, 8]
    mean_cha, std_cha, max_cha = [], [], []
    mean_reg, std_reg, max_reg = [], [], []
    neval_cha, neval_reg = [], []

    os.makedirs("figures", exist_ok=True)

    for N in system_sizes:
        print(f"\n{'='*50}")
        print(f"N={N}  |  chaotic (k={K_CHAOTIC})...")
        m, s, mx, ne, _ = depth_stats(N, K_CHAOTIC)
        print(f"-> mean depth={m:.2f} +/- {s:.2f}, max={mx}, over {ne} steps")
        mean_cha.append(m); std_cha.append(s)
        max_cha.append(mx); neval_cha.append(ne)

        print(f"N={N}  |  regular (k={K_REGULAR})...")
        m, s, mx, ne, _ = depth_stats(N, K_REGULAR)
        print(f"-> mean depth={m:.2f} +/- {s:.2f}, max={mx}, over {ne} steps")
        mean_reg.append(m); std_reg.append(s)
        max_reg.append(mx); neval_reg.append(ne)

        save_figure(system_sizes, mean_reg, std_reg, mean_cha, std_cha,
                    neval_cha=neval_cha)

    print("\nSaved: figures/depth_scaling.{pdf,png}")
    print("Regular  mean depth:", dict(zip(system_sizes,
                                           [round(x, 2) for x in mean_reg])))
    print("Chaotic  mean depth:", dict(zip(system_sizes,
                                           [round(x, 2) for x in mean_cha])))
    print("Regular  #steps evaluated:", dict(zip(system_sizes, neval_reg)))
    print("Chaotic  #steps evaluated:", dict(zip(system_sizes, neval_cha)))

    # Flag single-step N=8 chaotic point for paper
    for i, (N, ne) in enumerate(zip(system_sizes, neval_cha)):
        if ne <= 1:
            print(f"\n[B3 FLAG] N={N} chaotic: only {ne} timestep(s) evaluated. "
                  f"Mean depth = {mean_cha[i]:.2f}  -> this is a LOWER BOUND.")
            print("          Open marker drawn in figure. Keep open-marker")
            print("          language in Fig. 12 caption and manuscript.")


if __name__ == "__main__":
    main()
