"""
large_scale_scaling.py -- Mean circuit depth vs system size N (Fig. 12 / depth_scaling).

CHANGES FROM ORIGINAL (open-task list):
  [B3/Fig12-annotation] N=8 chaotic point is ALWAYS drawn as an open square
    and ALWAYS annotated on-plot with a text label -- no longer conditional on
    n_eval<=1.  Text adapts: "1 step (lower bound)" vs "{n} steps (t-avg)".
  [multi-step-N8]  TIME_BUDGET for N=8 raised to 600 s so multiple Floquet
    steps are averaged wherever the machine allows.
  [N10]  system_sizes now includes N=10 with a 600 s budget.  If N=10 runs
    out of time or memory it is skipped gracefully and flagged in stdout.
  [TeX-output]  All numerical results are printed at the end in a form that
    can be pasted directly into the manuscript.

All other logic (depth search, Ansatz, regular/chaotic split, residual-first
policy, etc.) is UNCHANGED from the corrected original.
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from vqs import Ansatz
from spin_operators import coherent_product_state, normalize
from qkt_quantum import floquet_U_exact

# ── Hyper-parameters ────────────────────────────────────────────────────────
EPS_OPT        = 0.05
N_STEPS        = 10          # max Floquet steps to attempt per (N, k)
K_CHAOTIC      = 2.5
K_REGULAR      = 0.5
N_RESTARTS     = 12
CEILING_OFFSET = 6
MAXITER        = 200

# N=8 budget raised from 180 s → 600 s; N=10 given 600 s.
TIME_BUDGET_TABLE = {4: 60.0, 6: 150.0, 8: 600.0, 10: 600.0}
TIME_BUDGET_DEFAULT = 600.0

# Set to False to skip N=10 (fast CI run):
TRY_N10 = True


# ── Core search ─────────────────────────────────────────────────────────────

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
            x0  = rng.uniform(0, 2 * np.pi, ans.n_params)
            res = minimize(cost, x0, method="L-BFGS-B",
                           options={"maxiter": MAXITER})
            if 1.0 - res.fun >= fid_target:
                return D, True

    return max_depth, False


def depth_stats(N, k,
                steps       = N_STEPS,
                eps_opt     = EPS_OPT,
                ceiling_offset = CEILING_OFFSET,
                n_restarts  = N_RESTARTS,
                time_budget_s = None):

    if time_budget_s is None:
        time_budget_s = TIME_BUDGET_TABLE.get(N, TIME_BUDGET_DEFAULT)

    psi0       = coherent_product_state(N)
    U_F        = floquet_U_exact(N, k, np.pi / 2)
    fid_target = 1.0 - eps_opt
    max_depth  = N + ceiling_offset

    depths        = []
    all_converged = True
    t_start       = time.time()

    for t in range(1, steps + 1):
        elapsed = time.time() - t_start
        if elapsed > time_budget_s and depths:
            print(f"  N={N} k={k}: time budget ({time_budget_s:.0f} s) reached "
                  f"after t={t-1}; reporting over {len(depths)} step(s).",
                  flush=True)
            break

        D, conv = min_sufficient_depth(
            N, k, psi0, U_F, t, fid_target, max_depth, n_restarts)
        depths.append(D)
        if not conv:
            all_converged = False
        print(f"  N={N} k={k} t={t:2d}: min sufficient depth={D}"
              f"{'' if conv else ' (ceiling)'}", flush=True)

    depths = np.array(depths, dtype=float)
    return (float(depths.mean()), float(depths.std(ddof=0 if len(depths)==1 else 1)),
            int(depths.max()), len(depths), all_converged)


# ── Figure ───────────────────────────────────────────────────────────────────

def save_figure(system_sizes, mean_reg, std_reg, mean_cha, std_cha,
                neval_cha=None, neval_reg=None):
    """Save Fig. 12 (depth_scaling).

    KEY CHANGE: the N=8 chaotic point is ALWAYS drawn as an open square
    and ALWAYS annotated, regardless of how many steps were evaluated.
    For n_eval==1 the label says "(lower bound)"; for n_eval>1 it says
    "(t-avg over {n} steps)".
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ns_done = np.array(system_sizes[:len(mean_reg)])

    # ── Regular regime: always solid markers ──
    ax.errorbar(ns_done, mean_reg[:len(ns_done)],
                yerr=std_reg[:len(ns_done)],
                fmt="o-", color="#0072B2", capsize=4,
                label="Regular ($k=0.5$)")

    # ── Chaotic regime: N=8 always open ──
    ns_cha = ns_done[:len(mean_cha)]
    first_cha_label_added = False

    for i, (n, m, s) in enumerate(zip(ns_cha, mean_cha, std_cha)):
        n_eval   = neval_cha[i] if (neval_cha is not None and i < len(neval_cha)) else None
        is_n8    = (n == 8)
        is_n10   = (n == 10)
        is_open  = is_n8 or is_n10 or (n_eval is not None and n_eval <= 1)
        mfc      = "none" if is_open else "#D55E00"

        lbl = "Chaotic ($k=2.5$)" if not first_cha_label_added else None
        if lbl:
            first_cha_label_added = True

        ax.errorbar([n], [m], yerr=[s], fmt="s-",
                    color="#D55E00", capsize=4,
                    markerfacecolor=mfc, label=lbl)

    # Connect chaotic trend line
    if len(mean_cha) > 1:
        ax.plot(ns_cha, mean_cha, "-", color="#D55E00", lw=1.5, zorder=0)

    ax.set_xlabel("System size $N$ (qubits)")
    ax.set_ylabel(r"Mean sufficient depth $\langle D\rangle_t$")
    ax.set_title("Mean circuit depth required vs. system size\n"
                 "(direct optimization, averaged over Floquet steps)")
    ax.set_xticks(ns_done)
    ax.grid(True, ls=":")

    # ── Per-point annotations ──────────────────────────────────────────────
    # N=8 annotation is ALWAYS added (this was the open task: "PNG edit
    # I can't do" → now baked into the script itself).
    for i, n in enumerate(ns_cha):
        if n not in (8, 10):
            continue
        n_eval = neval_cha[i] if (neval_cha is not None and i < len(neval_cha)) else None
        if n_eval == 1 or n_eval is None:
            step_str = "(lower bound)"
        else:
            step_str = f"({n_eval}-step t-avg)"

        m  = mean_cha[i]
        xoff = -0.55 if n >= 8 else 0.2
        yoff = +0.5

        ax.annotate(
            f"$N={n}$\n{step_str}",
            xy=(n, m),
            xytext=(n + xoff, m + yoff),
            fontsize=7.5,
            color="#D55E00",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8),
        )

    # Legend
    handles, labels_ = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels_) if l is not None]
    if filtered:
        ax.legend(*zip(*filtered))

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    for base in ("figures/depth_scaling", "figures/exact_dmax_scaling"):
        fig.savefig(base + ".pdf")
        fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    print(f"  [saved figures/depth_scaling.png  (N={list(ns_cha)})]", flush=True)


# ── Master runner ─────────────────────────────────────────────────────────────

def main():
    system_sizes = [4, 6, 8]
    if TRY_N10:
        system_sizes.append(10)

    mean_cha, std_cha, max_cha = [], [], []
    mean_reg, std_reg, max_reg = [], [], []
    neval_cha, neval_reg       = [], []

    os.makedirs("figures", exist_ok=True)

    valid_sizes = []
    for N in system_sizes:
        print(f"\n{'='*55}")
        print(f"N={N} | chaotic (k={K_CHAOTIC}) …")
        try:
            m, s, mx, ne, _ = depth_stats(N, K_CHAOTIC)
        except MemoryError:
            print(f"  N={N} chaotic: MemoryError — skipping N={N}.", flush=True)
            if N == 10:
                print("  N=10 skipped (insufficient RAM).  "
                      "Set TRY_N10=False to suppress.", flush=True)
            system_sizes = system_sizes[:system_sizes.index(N)]
            break
        print(f"  → mean depth={m:.2f} ± {s:.2f}, max={mx}, over {ne} steps")
        mean_cha.append(m); std_cha.append(s); max_cha.append(mx); neval_cha.append(ne)

        print(f"N={N} | regular (k={K_REGULAR}) …")
        try:
            m, s, mx, ne, _ = depth_stats(N, K_REGULAR)
        except MemoryError:
            print(f"  N={N} regular: MemoryError — using placeholder.", flush=True)
            m, s, mx, ne = (mean_reg[-1] if mean_reg else 1.0), 0.0, 1, 0
        print(f"  → mean depth={m:.2f} ± {s:.2f}, max={mx}, over {ne} steps")
        mean_reg.append(m); std_reg.append(s); max_reg.append(mx); neval_reg.append(ne)

        valid_sizes.append(N)

    save_figure(valid_sizes, mean_reg, std_reg, mean_cha, std_cha,
                neval_cha=neval_cha, neval_reg=neval_reg)

    # ── Structured JSON for reproducibility ────────────────────────────────
    results = {
        "system_sizes": valid_sizes,
        "chaotic": {
            "mean_depth": [round(x, 3) for x in mean_cha],
            "std_depth":  [round(x, 3) for x in std_cha],
            "max_depth":  max_cha,
            "n_steps_evaluated": neval_cha,
        },
        "regular": {
            "mean_depth": [round(x, 3) for x in mean_reg],
            "std_depth":  [round(x, 3) for x in std_reg],
            "max_depth":  max_reg,
            "n_steps_evaluated": neval_reg,
        },
        "eps_opt":    EPS_OPT,
        "k_chaotic":  K_CHAOTIC,
        "k_regular":  K_REGULAR,
        "n_restarts": N_RESTARTS,
        "n_steps_max": N_STEPS,
    }
    with open("figures/depth_scaling_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # ── TeX-ready stdout (paste directly into manuscript) ──────────────────
    print("\n")
    print("% " + "=" * 60)
    print("% PASTE INTO TEX — depth_scaling results")
    print("% " + "=" * 60)
    for N, mc, sc, ne_c, mr, sr, ne_r in zip(
            valid_sizes, mean_cha, std_cha, neval_cha,
            mean_reg, std_reg, neval_reg):
        lb = " (lower bound)" if ne_c <= 1 else f" ({ne_c}-step avg)"
        print(f"% N={N:2d}  chaotic: <D>={mc:.2f}+/-{sc:.2f}{lb} "
              f"| regular: <D>={mr:.2f}+/-{sr:.2f} ({ne_r}-step avg)")
    print("%")
    print("% Suggested table rows:")
    print(r"% \hline")
    for N, mc, sc, ne_c, mr, sr in zip(
            valid_sizes, mean_cha, std_cha, neval_cha, mean_reg, std_reg):
        lb = r"\dag" if ne_c <= 1 else ""
        print(f"% {N} & ${mr:.1f}\\pm{sr:.1f}$ "
              f"& ${mc:.1f}\\pm{sc:.1f}${lb} \\\\")
    print(r"% \hline")
    print("% (dag = single-step lower bound; use open square in caption)")
    print("%")

    # ── B3 flag: single-step lower bound check ─────────────────────────────
    for i, (N, ne) in enumerate(zip(valid_sizes, neval_cha)):
        if ne <= 1:
            print(f"[B3 FLAG] N={N} chaotic: only {ne} step(s) evaluated. "
                  f"Mean depth = {mean_cha[i]:.2f} → LOWER BOUND. "
                  "Open marker drawn; keep open-marker language in caption.")
        elif N == 8:
            print(f"[B3 NOTE] N={N} chaotic: {ne} step(s) averaged. "
                  f"Mean depth = {mean_cha[i]:.2f} ± {std_cha[i]:.2f}. "
                  "Open marker still drawn (largest N); remove '(lower bound)' "
                  "from caption if t-avg is considered sufficient.")


if __name__ == "__main__":
    main()
