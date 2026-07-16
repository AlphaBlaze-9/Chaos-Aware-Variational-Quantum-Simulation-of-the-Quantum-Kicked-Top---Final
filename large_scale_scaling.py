"""
large_scale_scaling.py -- Mean circuit depth vs system size N (Fig. 12 / depth_scaling).

CHANGES FROM ORIGINAL (open-task list):
  [B3/Fig12-annotation] N=8 chaotic point is ALWAYS drawn as an open square
    and ALWAYS annotated on-plot with a text label -- no longer conditional on
    n_eval<=1.  Text adapts: "1 step (lower bound)" vs "{n} steps (t-avg)".
  [multi-step-N8]  TIME_BUDGET for N=8 raised so multiple Floquet
    steps are averaged wherever the machine allows.
  [N10]  system_sizes now includes N=10 with its own budget.  If N=10 runs
    out of time or memory it is skipped gracefully and flagged in stdout.
  [TeX-output]  All numerical results are printed at the end in a form that
    can be pasted directly into the manuscript.

CHANGES FOR 50-RESTART RERUN (this version):
  [N_RESTARTS]  Bumped 12 -> 50 to match Table III's stated methodology
    ("Optimizer restarts -- 50") instead of leaving Fig. 11/12 on a silently
    weaker search than every other depth result in the paper.
  [TIME_BUDGET_TABLE]  Scaled by the same 50/12 ~= 4.17x factor, since the
    restart loop below runs every restart to completion before declaring a
    depth insufficient, so per-depth cost scales ~linearly with n_restarts.
    Leaving the old (12-restart-tuned) budgets in place would have caused
    this run to evaluate even fewer Floquet steps than the sparse 1-2 you
    already have at N=8/10 -- possibly zero. Treat these as a floor, not a
    target: raise further if you want denser step coverage.
  [PRECOMPUTED -> checkpoint]  The old PRECOMPUTED dict hardcoded results
    computed under N_RESTARTS=12. Setting N_RESTARTS=50 without touching
    that dict would have caused the script to print
    "[RESTORED FROM PRIOR RUN, NOT RECOMPUTED]" and silently reuse the old
    12-restart numbers for every N -- defeating the entire point of this
    change. It has been replaced with a disk-backed checkpoint
    (figures/depth_scaling_checkpoint.json) keyed on the FULL run config
    (eps_opt, n_steps, ceiling_offset, n_restarts). If the saved config
    doesn't match the current constants below, the checkpoint is ignored
    and everything recomputes fresh -- this class of bug (silently reusing
    results computed under a different setting) can't happen again even if
    you change N_RESTARTS a third time. It also checkpoints incrementally
    after each (N, k) finishes, not just at the very end of main(), so a
    crash on N=10 no longer costs you the already-finished N=4/6/8 results.

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
K_REGULAR_ALT  = 1.5         # [item xiii] second regular-plateau point (FTLE~0
                              # here too, per Fig. 4) -- tests whether the N-scaling
                              # gap tracks chaoticity (should track K_REGULAR) or
                              # just the larger rotation angle at k=2.5 (would
                              # track K_CHAOTIC instead).
RUN_REGULAR_ALT = True       # set False to skip this and only run the original pair
N_RESTARTS     = 50          # was 12 -- now matches Table III's stated "50"
CEILING_OFFSET = 6
MAXITER        = 200

# Scaled from the original {4: 60.0, 6: 150.0, 8: 600.0, 10: 600.0} by the
# same 50/12 ~= 4.1667x factor as N_RESTARTS, since the restart loop runs to
# completion on failure (needed to certify a depth "insufficient"), so cost
# scales ~linearly with n_restarts. This is a starting point, not a
# guarantee of full 10-step coverage at N=8/10 -- it wasn't enough for that
# even at 12 restarts. Raise further if you want denser data and are
# willing to let it run longer (overnight with caffeinate, as before).
TIME_BUDGET_TABLE = {4: 250.0, 6: 625.0, 8: 2500.0, 10: 2500.0}
TIME_BUDGET_DEFAULT = 2500.0

# Set to False to skip N=10 (fast CI run):
TRY_N10 = True

# ── [PATCH] Disk-backed checkpoint, replacing the old hardcoded PRECOMPUTED
# dict. Keyed on the full run config so results computed under a different
# N_RESTARTS (or eps_opt / n_steps / ceiling_offset) are never silently
# reused -- if the saved config doesn't match, the checkpoint is ignored
# and that (N, k) recomputes fresh. Saved incrementally after every (N, k)
# so a crash partway through (e.g. during N=10) doesn't lose the results
# already computed for smaller N.
CHECKPOINT_PATH = "figures/depth_scaling_checkpoint.json"


def _checkpoint_config():
    return {
        "eps_opt": EPS_OPT,
        "n_steps": N_STEPS,
        "ceiling_offset": CEILING_OFFSET,
        "n_restarts": N_RESTARTS,
    }


def _load_checkpoint():
    """Load per-(N,k) results already computed under the CURRENT config.
    If no checkpoint file exists, or it was written under a different
    config (e.g. an older N_RESTARTS=12 run), returns an empty cache so
    every (N,k) recomputes fresh -- rather than silently mixing results
    computed under different restart counts."""
    if not os.path.exists(CHECKPOINT_PATH):
        return {}
    try:
        with open(CHECKPOINT_PATH) as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError):
        print(f"[CHECKPOINT] {CHECKPOINT_PATH} unreadable -- starting fresh.",
              flush=True)
        return {}

    if raw.get("config") != _checkpoint_config():
        print(f"[CHECKPOINT] Found {CHECKPOINT_PATH} but its config "
              f"{raw.get('config')} does not match the current config "
              f"{_checkpoint_config()} -- ignoring stale checkpoint, "
              f"recomputing everything under the current config.",
              flush=True)
        return {}

    cache = {}
    for key_str, v in raw.get("results", {}).items():
        N_str, k_str = key_str.split(",")
        cache[(int(N_str), float(k_str))] = tuple(v)
    if cache:
        print(f"[CHECKPOINT] Loaded {len(cache)} completed (N,k) result(s) "
              f"from {CHECKPOINT_PATH} (config matches current run).",
              flush=True)
    return cache


def _save_checkpoint(cache):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH) or ".", exist_ok=True)
    serializable = {f"{N},{k}": list(v) for (N, k), v in cache.items()}
    with open(CHECKPOINT_PATH, "w") as fh:
        json.dump({"config": _checkpoint_config(), "results": serializable},
                  fh, indent=2)


# ── Core search ─────────────────────────────────────────────────────────────

def min_sufficient_depth(N, k, psi0, U_F, t, fid_target, max_depth, n_restarts,
                         time_budget_s=None, t_start=None):
    """[PATCH] time_budget_s/t_start: if given, checked before every new
    depth and after every restart, not just between t's. Without this, a
    single non-converging t (e.g. deep in a high-N search) can run
    unbounded, since the caller's between-t check never gets a chance to
    fire until this whole function returns. Returns (D, converged, timed_out).
    """
    psi_t = psi0.copy()
    for _ in range(t):
        psi_t = normalize(U_F @ psi_t)

    for D in range(1, max_depth + 1):
        if time_budget_s is not None and t_start is not None:
            if time.time() - t_start > time_budget_s:
                return D, False, True

        ans = Ansatz(N, D)

        def cost(th):
            return 1.0 - abs(np.vdot(psi_t, ans.state(th, psi0))) ** 2

        for r in range(n_restarts):
            rng = np.random.default_rng((N, t, D, r))
            x0  = rng.uniform(0, 2 * np.pi, ans.n_params)
            res = minimize(cost, x0, method="L-BFGS-B",
                           options={"maxiter": MAXITER})
            if 1.0 - res.fun >= fid_target:
                return D, True, False

            if time_budget_s is not None and t_start is not None:
                if time.time() - t_start > time_budget_s:
                    return D, False, True

    return max_depth, False, False


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
        if elapsed > time_budget_s:
            print(f"  N={N} k={k}: time budget ({time_budget_s:.0f} s) reached "
                  f"after t={t-1}; reporting over {len(depths)} step(s).",
                  flush=True)
            break

        D, conv, timed_out = min_sufficient_depth(
            N, k, psi0, U_F, t, fid_target, max_depth, n_restarts,
            time_budget_s=time_budget_s, t_start=t_start)
        if timed_out:
            print(f"  N={N} k={k} t={t:2d}: time budget ({time_budget_s:.0f} s) "
                  f"reached mid-search at D={D}; discarding this partial "
                  f"step, reporting over {len(depths)} step(s).", flush=True)
            break
        depths.append(D)
        if not conv:
            all_converged = False
        print(f"  N={N} k={k} t={t:2d}: min sufficient depth={D}"
              f"{'' if conv else ' (ceiling)'}", flush=True)

    if len(depths) == 0:
        print(f"  N={N} k={k}: time budget reached before any step could "
              f"be evaluated -- reporting nothing for this (N,k).",
              flush=True)
        return (float('nan'), float('nan'), 0, 0, False)

    depths = np.array(depths, dtype=float)
    return (float(depths.mean()), float(depths.std(ddof=0 if len(depths)==1 else 1)),
            int(depths.max()), len(depths), all_converged)


# ── Figure ───────────────────────────────────────────────────────────────────

def save_figure(system_sizes, mean_reg, std_reg, mean_cha, std_cha,
                neval_cha=None, neval_reg=None,
                mean_reg2=None, std_reg2=None, k_reg2=None):
    """Save Fig. 12 (depth_scaling).

    KEY CHANGE: the N=8 chaotic point is ALWAYS drawn as an open square
    and ALWAYS annotated, regardless of how many steps were evaluated.
    For n_eval==1 the label says "(lower bound)"; for n_eval>1 it says
    "(t-avg over {n} steps)".

    [PATCH] Regular-regime points with n_eval < 5 (currently just N=10,
    at 3/10 steps) now get the same open-marker + annotation treatment
    the chaotic side already had. Without this, N=10 regular's thin,
    likely-optimistic 3-step average (1.00) would be plotted identically
    to N=4/N=6's full 10-step averages, implying equal confidence it
    doesn't have.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ns_done = np.array(system_sizes[:len(mean_reg)])

    # ── Regular regime: solid, EXCEPT undersampled points (n_eval<5) ──
    UNDERSAMPLED_THRESHOLD = 5
    reg_open_mask = np.array([
        (neval_reg[i] if (neval_reg is not None and i < len(neval_reg)) else 999)
        < UNDERSAMPLED_THRESHOLD
        for i in range(len(ns_done))
    ])
    # Split into solid and open groups so errorbar can style them separately
    if reg_open_mask.any():
        solid_idx = ~reg_open_mask
        if solid_idx.any():
            ax.errorbar(ns_done[solid_idx], np.array(mean_reg)[solid_idx],
                        yerr=np.array(std_reg)[solid_idx],
                        fmt="o-", color="#0072B2", capsize=4,
                        label="Regular ($k=0.5$)")
        ax.errorbar(ns_done[reg_open_mask], np.array(mean_reg)[reg_open_mask],
                    yerr=np.array(std_reg)[reg_open_mask],
                    fmt="o", color="#0072B2", capsize=4,
                    markerfacecolor="none",
                    label=None if solid_idx.any() else "Regular ($k=0.5$)")
        # Connect all regular points with a line regardless of marker style
        ax.plot(ns_done, mean_reg[:len(ns_done)], "-", color="#0072B2",
                lw=1.5, zorder=0)
    else:
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

    # ── [item xiii] Regular-alt (k=1.5) confound-check series ──────────────
    if mean_reg2:
        ns_reg2 = ns_done[:len(mean_reg2)]
        ax.errorbar(ns_reg2, mean_reg2, yerr=std_reg2, fmt="^--",
                    color="#009E73", capsize=4,
                    label=f"Regular-alt ($k={k_reg2}$)" if k_reg2 else "Regular-alt")

    ax.set_xlabel("System size $N$ (qubits)")
    ax.set_ylabel(r"Mean sufficient depth $\langle D\rangle_t$")
    ax.set_title("Mean circuit depth required vs. system size\n"
                 "(direct optimization, averaged over Floquet steps)",
                 pad=14)
    ax.set_xticks(ns_done)
    ax.grid(True, ls=":")
    # A6 fix: add top headroom so the N=8 "(lower bound)" annotation
    # cannot collide with the two-line title.
    _all_vals = list(mean_reg[:len(ns_done)]) + list(mean_cha)
    _ymax = max(_all_vals) if _all_vals else 6.0
    ax.set_ylim(0, _ymax + 2.0)

    # ── Per-point annotations (chaotic side, unchanged) ────────────────────
    for i, n in enumerate(ns_cha):
        if n not in (8, 10):
            continue
        n_eval = neval_cha[i] if (neval_cha is not None and i < len(neval_cha)) else None
        if n_eval == 1 or n_eval is None:
            step_str = "(lower bound)"
        else:
            step_str = f"({n_eval}-step t-avg)"

        m  = mean_cha[i]
        xoff = -0.85 if n >= 8 else 0.2
        yoff = -1.4 if n >= 8 else +0.5

        ax.annotate(
            f"$N={n}$\n{step_str}",
            xy=(n, m),
            xytext=(n + xoff, m + yoff),
            fontsize=7.5,
            color="#D55E00",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8),
        )

    # ── [PATCH] Per-point annotations, regular side (undersampled only) ────
    for i, n in enumerate(ns_done):
        n_eval = neval_reg[i] if (neval_reg is not None and i < len(neval_reg)) else None
        if n_eval is None or n_eval >= UNDERSAMPLED_THRESHOLD:
            continue
        m = mean_reg[i]
        ax.annotate(
            f"$N={n}$\n({n_eval}-step avg,\nlikely optimistic)",
            xy=(n, m),
            xytext=(n - 0.3, m + 1.8),
            fontsize=7.5,
            color="#0072B2",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.8),
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


def replot_from_json(path="figures/depth_scaling_results.json"):
    """[PATCH] Regenerate the figure from the already-saved JSON without
    rerunning any computation -- takes seconds, not minutes. Use this after
    changing save_figure()'s styling (e.g. this open-marker patch) instead
    of rerunning main()."""
    with open(path) as fh:
        r = json.load(fh)
    save_figure(
        r["system_sizes"],
        r["regular"]["mean_depth"], r["regular"]["std_depth"],
        r["chaotic"]["mean_depth"], r["chaotic"]["std_depth"],
        neval_cha=r["chaotic"]["n_steps_evaluated"],
        neval_reg=r["regular"]["n_steps_evaluated"],
    )
    print("Replotted from saved JSON -- no computation was rerun.")


# ── Master runner ─────────────────────────────────────────────────────────────

def main():
    system_sizes = [4, 6, 8]
    if TRY_N10:
        system_sizes.append(10)

    checkpoint = _load_checkpoint()

    mean_cha, std_cha, max_cha = [], [], []
    mean_reg, std_reg, max_reg = [], [], []
    neval_cha, neval_reg       = [], []
    mean_reg2, std_reg2, max_reg2, neval_reg2 = [], [], [], []

    os.makedirs("figures", exist_ok=True)

    valid_sizes = []
    for N in system_sizes:
        print(f"\n{'='*55}")
        print(f"N={N} | chaotic (k={K_CHAOTIC}) …")
        key_cha = (N, K_CHAOTIC)
        if key_cha in checkpoint:
            m, s, mx, ne, _ = checkpoint[key_cha]
            print(f"  [RESTORED FROM CHECKPOINT, NOT RECOMPUTED "
                  f"-- n_restarts={N_RESTARTS} matches saved config]")
        else:
            try:
                m, s, mx, ne, conv = depth_stats(N, K_CHAOTIC)
            except MemoryError:
                print(f"  N={N} chaotic: MemoryError — skipping N={N}.", flush=True)
                if N == 10:
                    print("  N=10 skipped (insufficient RAM).  "
                          "Set TRY_N10=False to suppress.", flush=True)
                system_sizes = system_sizes[:system_sizes.index(N)]
                break
            checkpoint[key_cha] = (m, s, mx, ne, conv)
            _save_checkpoint(checkpoint)
            print(f"  [COMPUTED FRESH -- checkpointed to {CHECKPOINT_PATH}]")
        print(f"  → mean depth={m:.2f} ± {s:.2f}, max={mx}, over {ne} steps")
        mean_cha.append(m); std_cha.append(s); max_cha.append(mx); neval_cha.append(ne)

        print(f"N={N} | regular (k={K_REGULAR}) …")
        key_reg = (N, K_REGULAR)
        if key_reg in checkpoint:
            m, s, mx, ne, _ = checkpoint[key_reg]
            print(f"  [RESTORED FROM CHECKPOINT, NOT RECOMPUTED "
                  f"-- n_restarts={N_RESTARTS} matches saved config]")
        else:
            try:
                m, s, mx, ne, conv = depth_stats(N, K_REGULAR)
            except MemoryError:
                print(f"  N={N} regular: MemoryError — using placeholder.", flush=True)
                m, s, mx, ne, conv = (mean_reg[-1] if mean_reg else 1.0), 0.0, 1, 0, False
            checkpoint[key_reg] = (m, s, mx, ne, conv)
            _save_checkpoint(checkpoint)
            print(f"  [COMPUTED FRESH -- checkpointed to {CHECKPOINT_PATH}]")
        print(f"  → mean depth={m:.2f} ± {s:.2f}, max={mx}, over {ne} steps")
        mean_reg.append(m); std_reg.append(s); max_reg.append(mx); neval_reg.append(ne)

        # [item xiii] Second regular-plateau point at K_REGULAR_ALT=1.5.
        # Still FTLE~0 (regular plateau extends to k<=1.5 per Fig. 4), so if
        # <D>_t here tracks K_REGULAR (0.5) rather than K_CHAOTIC (2.5), that's
        # direct evidence the N-scaling gap is driven by chaoticity and not
        # merely by k=2.5 being a larger rotation angle.
        if RUN_REGULAR_ALT:
            print(f"N={N} | regular-alt (k={K_REGULAR_ALT}) …")
            key_reg2 = (N, K_REGULAR_ALT)
            if key_reg2 in checkpoint:
                m2, s2, mx2, ne2, _ = checkpoint[key_reg2]
                print(f"  [RESTORED FROM CHECKPOINT, NOT RECOMPUTED "
                      f"-- n_restarts={N_RESTARTS} matches saved config]")
            else:
                try:
                    m2, s2, mx2, ne2, conv2 = depth_stats(N, K_REGULAR_ALT)
                except MemoryError:
                    print(f"  N={N} regular-alt: MemoryError — using placeholder.", flush=True)
                    m2, s2, mx2, ne2, conv2 = (mean_reg2[-1] if mean_reg2 else 1.0), 0.0, 1, 0, False
                checkpoint[key_reg2] = (m2, s2, mx2, ne2, conv2)
                _save_checkpoint(checkpoint)
                print(f"  [COMPUTED FRESH -- checkpointed to {CHECKPOINT_PATH}]")
            print(f"  → mean depth={m2:.2f} ± {s2:.2f}, max={mx2}, over {ne2} steps")
            mean_reg2.append(m2); std_reg2.append(s2); max_reg2.append(mx2); neval_reg2.append(ne2)

        valid_sizes.append(N)

    save_figure(valid_sizes, mean_reg, std_reg, mean_cha, std_cha,
                neval_cha=neval_cha, neval_reg=neval_reg,
                mean_reg2=mean_reg2 if RUN_REGULAR_ALT else None,
                std_reg2=std_reg2 if RUN_REGULAR_ALT else None,
                k_reg2=K_REGULAR_ALT if RUN_REGULAR_ALT else None)

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
        "regular_alt": {
            "k": K_REGULAR_ALT,
            "mean_depth": [round(x, 3) for x in mean_reg2],
            "std_depth":  [round(x, 3) for x in std_reg2],
            "max_depth":  max_reg2,
            "n_steps_evaluated": neval_reg2,
        } if RUN_REGULAR_ALT else None,
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

    # ── [item xiii] Confound check: does k=1.5 track k=0.5 or k=2.5? ───────
    if RUN_REGULAR_ALT and mean_reg2:
        print("\n" + "%" + "=" * 60)
        print("% ITEM (xiii) CONFOUND CHECK -- paste verdict into Sec. IV.C")
        print("%" + "=" * 60)
        for i, N in enumerate(valid_sizes[:len(mean_reg2)]):
            d05, d15, d25 = mean_reg[i], mean_reg2[i], mean_cha[i]
            # [FIX] np.isnan comparisons are always False in Python, which
            # was silently falling into the "tracks k=2.5" branch whenever
            # a point hadn't actually converged/evaluated -- reporting a
            # confound verdict from no data. Guard against that explicitly.
            if any(np.isnan(v) for v in (d05, d15, d25)):
                print(f"% N={N:2d}: <D>_t(k=0.5)={d05}  <D>_t(k=1.5)={d15}  "
                      f"<D>_t(k=2.5)={d25}  -> INSUFFICIENT DATA (one or more "
                      f"points is NaN -- not a real verdict, do not cite)")
                continue
            dist_to_05 = abs(d15 - d05)
            dist_to_25 = abs(d15 - d25)
            verdict = ("tracks k=0.5 (supports chaoticity, not rotation angle)"
                       if dist_to_05 < dist_to_25 else
                       "tracks k=2.5 (confound NOT resolved -- flag in paper)")
            print(f"% N={N:2d}: <D>_t(k=0.5)={d05:.2f}  <D>_t(k=1.5)={d15:.2f}  "
                  f"<D>_t(k=2.5)={d25:.2f}  -> {verdict}")

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
    import sys
    if "--replot" in sys.argv:
        replot_from_json()
    else:
        main()