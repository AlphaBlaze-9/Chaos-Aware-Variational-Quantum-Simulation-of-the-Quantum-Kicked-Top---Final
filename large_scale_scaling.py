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
# [PERF] Must be set BEFORE numpy is imported, and before any worker process
# touches numpy, or each of the N_WORKERS processes below will itself try to
# spawn multiple BLAS threads -- on a 6-core machine this oversubscription
# (6 processes x however many BLAS threads each) typically makes parallel
# execution SLOWER than serial, not faster. This is the single most common
# way to sabotage a multiprocessing + numpy speedup.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import json
import platform
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from vqs import Ansatz, infidelity_and_grad
from spin_operators import coherent_product_state, normalize
from qkt_quantum import floquet_U_exact

# [PERF] Number of parallel worker processes for the restart loop. Defaults
# to physical/logical core count; override if you want to leave a core free
# for foreground work while this runs.
N_WORKERS = int(os.environ.get("SCALING_N_WORKERS", os.cpu_count() or 4))

# [PROVENANCE] Identify the machine this run executes on. This matters
# because the manuscript's Funding Information section names specific
# hardware (CPU/GPU/RAM) that the reported N=8/N=10 numbers were computed
# on. If this script is ever run on a DIFFERENT machine, its results must
# not silently end up in the same checkpoint file as results from the
# machine actually named in the paper -- that would make the hardware
# disclosure inaccurate without anyone noticing. Two independent layers of
# protection below: (1) the checkpoint file itself is machine-tagged by
# default, so different machines write to different files entirely; (2) the
# machine tag is also stored inside the checkpoint's config, so even a
# manually copied/renamed checkpoint file gets flagged as a mismatch and
# recomputed rather than silently trusted (same existing mechanism this
# file already uses to invalidate stale N_RESTARTS/eps_opt configs).
_MACHINE_TAG = (platform.node() or "unknown_host").replace(" ", "_")
_CPU_TAG = (platform.processor() or "unknown_cpu").replace(" ", "_")[:40]

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

# [PATCH v6 -- per-N ceiling, decoupled from the global checkpoint config]
# Previously CEILING_OFFSET was one global constant, and it was part of the
# checkpoint's file-level config -- meaning ANY change to it (e.g. to give
# N=10 more room) invalidated N=4/6/8 too, forcing a full, expensive
# recompute of data that was already correct and hard-won. That's no
# longer true: ceiling is now tracked PER (N,k) ENTRY inside the
# checkpoint (see _load_checkpoint/_save_checkpoint below), migrated
# automatically from any old flat-format checkpoint file. Raising
# CEILING_OFFSET_BY_N[10] below will recompute ONLY N=10 entries; N=4/6/8,
# already computed under offset=12, stay cached and untouched.
#
# N=8's own results are the reason N=10 gets a much bigger jump than N=8
# did: N=8 chaotic's slowest timestep needed depth 18 out of a ceiling of
# 20 -- only 2 layers of headroom to spare. Given depth requirements have
# been growing sharply with N (N=6 max=5 -> N=8 max=18), N=10 could
# plausibly need well beyond the old ceiling of 22. This is a genuine
# extrapolation, not a guarantee -- if N=10 hits even this new ceiling,
# that is itself worth reporting honestly rather than chasing indefinitely.
CEILING_OFFSET_BY_N = {4: 12, 6: 12, 8: 12, 10: 30}
CEILING_OFFSET_DEFAULT = 12
MAXITER        = 200

# [PATCH v6] N=10 given a much larger, dedicated budget -- previously
# 10000s wasn't enough to get past t=2 (t=1 converged at D=11; t=2 alone
# consumed the rest of the budget without finishing). This is meant to be
# run as its own separate, focused session (e.g. overnight), not
# alongside a routine N=4/6/8 rerun. N=4/6/8 budgets are left as before
# since those are already complete and will restore from checkpoint
# almost instantly regardless of budget.
TIME_BUDGET_TABLE = {4: 500.0, 6: 1500.0, 8: 6000.0, 10: 43200.0}  # N=10: 12h
TIME_BUDGET_DEFAULT = 43200.0

# Set to False to skip N=10 (fast CI run):
TRY_N10 = True

# ── [PATCH] Disk-backed checkpoint, replacing the old hardcoded PRECOMPUTED
# dict. Keyed on the full run config so results computed under a different
# N_RESTARTS (or eps_opt / n_steps / ceiling_offset) are never silently
# reused -- if the saved config doesn't match, the checkpoint is ignored
# and that (N, k) recomputes fresh. Saved incrementally after every (N, k)
# so a crash partway through (e.g. during N=10) doesn't lose the results
# already computed for smaller N.
#
# [PROVENANCE] Filename now includes the machine tag by default, so running
# this on a different computer writes to a DIFFERENT checkpoint file rather
# than reading/writing the same one your primary-hardware results live in.
# Override with SCALING_CHECKPOINT_PATH if you deliberately want a specific
# shared path (e.g. to merge results by hand later with full awareness of
# what came from where).
CHECKPOINT_PATH = os.environ.get(
    "SCALING_CHECKPOINT_PATH",
    f"figures/depth_scaling_checkpoint__{_MACHINE_TAG}.json",
)


def _checkpoint_config():
    """[PATCH v6] ceiling_offset removed from here -- it's now tracked
    PER (N,k) ENTRY (see _load_checkpoint/_save_checkpoint), not as one
    global value for the whole file. Everything still listed here applies
    uniformly across all N and SHOULD invalidate everything if changed
    (a different gradient method, restart count, or machine really does
    make every cached result incomparable) -- ceiling_offset is the one
    knob we specifically want to vary per-N without that blast radius.
    """
    return {
        "eps_opt": EPS_OPT,
        "n_steps": N_STEPS,
        "n_restarts": N_RESTARTS,
        "machine": _MACHINE_TAG,   # [PROVENANCE] a checkpoint file computed
        "cpu": _CPU_TAG,           # on a different machine now fails this
                                    # equality check and is treated as stale,
                                    # even if someone manually copies/renames
                                    # the file to look like a match.
        "gradient_method": "analytic_adjoint",
    }


def _load_checkpoint():
    """Load per-(N,k) results already computed under the CURRENT global
    config AND the CURRENT per-N ceiling_offset for that specific N. If no
    checkpoint file exists, or its global config doesn't match, returns an
    empty cache. If the global config matches but a given entry's
    ceiling_offset doesn't match CEILING_OFFSET_BY_N[N], only THAT entry is
    dropped -- other entries (e.g. N=4/6/8 when only N=10's ceiling
    changed) remain valid and cached.

    [PATCH v6 -- migration] Old checkpoint files (written before this
    per-entry-ceiling change) store ceiling_offset once, globally, under
    "config", with flat [m,s,mx,ne,conv] result values. Such a file is
    auto-migrated on load: its global ceiling_offset is treated as the
    recorded per-entry offset for every entry it contains, then the same
    per-entry validity check applies -- so an old N=4/6/8/10 checkpoint
    (all computed under one old global offset) correctly keeps N=4/6/8
    valid while dropping N=10 the moment CEILING_OFFSET_BY_N[10] is raised
    above that old value, with no manual intervention needed.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        return {}, {}
    try:
        with open(CHECKPOINT_PATH) as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError):
        print(f"[CHECKPOINT] {CHECKPOINT_PATH} unreadable -- starting fresh.",
              flush=True)
        return {}, {}

    is_old_format = "config" in raw and "global_config" not in raw

    if is_old_format:
        old_config = dict(raw.get("config", {}))
        old_ceiling_offset = old_config.pop("ceiling_offset", None)
        if old_config != _checkpoint_config():
            print(f"[CHECKPOINT] Found {CHECKPOINT_PATH} (old format) but its "
                  f"non-ceiling config {old_config} does not match the current "
                  f"config {_checkpoint_config()} -- ignoring stale checkpoint, "
                  f"recomputing everything.", flush=True)
            return {}, {}
        print(f"[CHECKPOINT] Migrating old-format checkpoint (global "
              f"ceiling_offset={old_ceiling_offset}) to per-entry format.",
              flush=True)
        raw_results = {
            key: {"ceiling_offset": old_ceiling_offset, "value": val}
            for key, val in raw.get("results", {}).items()
        }
    else:
        if raw.get("global_config") != _checkpoint_config():
            print(f"[CHECKPOINT] Found {CHECKPOINT_PATH} but its global config "
                  f"{raw.get('global_config')} does not match the current "
                  f"config {_checkpoint_config()} -- ignoring stale checkpoint, "
                  f"recomputing everything.", flush=True)
            return {}, {}
        raw_results = raw.get("results", {})

    cache = {}
    entry_ceiling_offsets = {}
    dropped = []
    for key_str, entry in raw_results.items():
        N_str, k_str = key_str.split(",")
        N_key = int(N_str)
        expected_offset = CEILING_OFFSET_BY_N.get(N_key, CEILING_OFFSET_DEFAULT)
        if entry.get("ceiling_offset") != expected_offset:
            dropped.append((key_str, entry.get("ceiling_offset"), expected_offset))
            continue
        key = (N_key, float(k_str))
        cache[key] = tuple(entry["value"])
        entry_ceiling_offsets[key] = expected_offset

    if cache:
        print(f"[CHECKPOINT] Loaded {len(cache)} completed (N,k) result(s) "
              f"from {CHECKPOINT_PATH} (config + per-entry ceiling match).",
              flush=True)
    for key_str, old_off, new_off in dropped:
        print(f"[CHECKPOINT] Dropping cached entry {key_str}: computed under "
              f"ceiling_offset={old_off}, current setting for its N is "
              f"{new_off} -- will recompute.", flush=True)
    return cache, entry_ceiling_offsets


def _save_checkpoint(cache, entry_ceiling_offsets):
    """[PATCH v6] entry_ceiling_offsets: dict mapping (N,k) -> the
    ceiling_offset actually used to compute that entry, stored alongside
    each result so future runs can tell whether it's still valid under
    whatever CEILING_OFFSET_BY_N says for that N at load time."""
    os.makedirs(os.path.dirname(CHECKPOINT_PATH) or ".", exist_ok=True)
    serializable = {
        f"{N},{k}": {
            "ceiling_offset": entry_ceiling_offsets.get((N, k)),
            "value": list(v),
        }
        for (N, k), v in cache.items()
    }
    with open(CHECKPOINT_PATH, "w") as fh:
        json.dump({"global_config": _checkpoint_config(), "results": serializable},
                  fh, indent=2)


# ── Core search ─────────────────────────────────────────────────────────────

def _run_one_restart(args):
    """[PERF] Module-level (hence picklable) worker for a single restart --
    runs in its own process. Reconstructs Ansatz(N, D) locally rather than
    pickling/sending an existing instance, to sidestep any non-picklable
    cached state inside Ansatz; this reconstruction cost is negligible next
    to the optimization itself. Returns (r, infidelity, success) -- the
    infidelity is returned too (not just success) in case you later want to
    log/report success fractions the way Appendix C already does for the
    Dmax sweep in plot_dmax_vs_k.py.

    [PERF -- analytic gradient] Previously this called minimize(cost, ...)
    with no jac= argument, so SciPy estimated the gradient by finite
    differences: ~n_params extra calls to ans.state() per gradient, every
    L-BFGS-B iteration. At N=10, D=16 that's ~337 extra state simulations
    per gradient step. Now uses infidelity_and_grad (adjoint
    differentiation, vqs.py) via jac=True: the objective function itself
    returns (value, gradient) computed together in ~2 full-circuit-passes
    total, independent of n_params. Measured speedup at N=10, D=16: ~158x
    per gradient evaluation, with gradient values matching finite
    differences to ~5e-11 (validated across N=2..5, D=1..3 before being
    wired in here). This is the dominant fix for the N=8/N=10 sampling gap
    -- more CPU cores alone could not move those points, because each
    individual restart was inherently this expensive regardless of
    parallelism.
    """
    N, D, t, r, psi_t, psi0, fid_target, maxiter = args
    ans = Ansatz(N, D)

    def cost_and_grad(th):
        return infidelity_and_grad(ans, th, psi0, psi_t)

    rng = np.random.default_rng((N, t, D, r))
    x0 = rng.uniform(0, 2 * np.pi, ans.n_params)
    res = minimize(cost_and_grad, x0, method="L-BFGS-B", jac=True,
                   options={"maxiter": maxiter})
    success = (1.0 - res.fun) >= fid_target
    return r, float(res.fun), bool(success)


def min_sufficient_depth(N, k, psi0, U_F, t, fid_target, max_depth, n_restarts,
                         time_budget_s=None, t_start=None,
                         executor=None, n_workers=None):
    """[PATCH] time_budget_s/t_start: if given, checked before every new
    depth and while restarts are streaming in, not just between t's.
    Without this, a single non-converging t (e.g. deep in a high-N search)
    can run unbounded, since the caller's between-t check never gets a
    chance to fire until this whole function returns. Returns
    (D, converged, timed_out).

    [PERF v2 -- streaming dispatch] Previously this fired restarts in
    fixed-size batches of n_workers and waited for the WHOLE batch (or an
    early success) before submitting the next. That's fine on a homogeneous
    CPU, but on a hybrid P-core/E-core chip (e.g. Intel 200-series "Ultra"
    parts) a fixed batch's completion time is gated by whichever restart
    the OS scheduler happens to land on the slowest core -- fast cores
    finish early and then sit idle waiting for stragglers before the next
    batch can even start. This version instead keeps up to n_workers
    restarts continuously in flight: the instant any one finishes, the next
    pending restart (if any) is submitted immediately. Fast cores naturally
    pull through more restarts per unit time; nothing waits on a straggler.
    This is a strict improvement on homogeneous hardware too (removes
    synchronization stalls from restart-to-restart variance in how many
    L-BFGS-B iterations it takes to converge or fail), and specifically
    fixes the P-core/E-core imbalance on hybrid CPUs.

    NOTE: because only "was D sufficient" is returned (not which restart
    index proved it), this is exactly equivalent to the original serial
    semantics for every quantity actually used downstream (depth_stats,
    the figure, the confound check) -- it does NOT change which restart
    happens to be credited with success, only how fast/efficiently we find
    out, regardless of core count or core heterogeneity.

    Pass an existing `executor` to reuse one pool across many calls
    (STRONGLY recommended -- spawning a new process pool per depth/timestep
    is itself expensive). If none is given, one is created and torn down
    locally, which is safe but slower if called many times in a loop.
    """
    n_workers = n_workers or N_WORKERS
    psi_t = psi0.copy()
    for _ in range(t):
        psi_t = normalize(U_F @ psi_t)

    owns_executor = executor is None
    if owns_executor:
        executor = ProcessPoolExecutor(max_workers=n_workers)

    try:
        for D in range(1, max_depth + 1):
            if time_budget_s is not None and t_start is not None:
                if time.time() - t_start > time_budget_s:
                    return D, False, True

            next_r = 0
            in_flight = {}          # future -> restart index
            success_found = False
            timed_out_here = False

            while next_r < n_restarts or in_flight:
                # Keep the in-flight window topped up to n_workers.
                while next_r < n_restarts and len(in_flight) < n_workers:
                    fut = executor.submit(
                        _run_one_restart,
                        (N, D, t, next_r, psi_t, psi0, fid_target, MAXITER))
                    in_flight[fut] = next_r
                    next_r += 1

                if not in_flight:
                    break  # nothing left to submit and nothing pending

                done, _ = wait(list(in_flight.keys()),
                                return_when=FIRST_COMPLETED)
                for fut in done:
                    in_flight.pop(fut, None)
                    _, _, success = fut.result()
                    if success:
                        success_found = True
                # Don't bother launching new restarts once we know D is
                # sufficient -- fall through and return below. Any other
                # still-in-flight futures from this depth are abandoned
                # (not cancelled forcibly, since a worker mid-computation
                # can't be interrupted anyway) -- same negligible-waste
                # tradeoff as the previous batched version.
                if success_found:
                    break

                if time_budget_s is not None and t_start is not None:
                    if time.time() - t_start > time_budget_s:
                        for f in in_flight:
                            f.cancel()  # best-effort: only cancels futures
                        timed_out_here = True    # that haven't started yet
                        break

            if success_found:
                return D, True, False
            if timed_out_here:
                return D, False, True
    finally:
        if owns_executor:
            executor.shutdown(wait=False, cancel_futures=True)

    return max_depth, False, False


def depth_stats(N, k,
                steps       = N_STEPS,
                eps_opt     = EPS_OPT,
                ceiling_offset = None,
                n_restarts  = N_RESTARTS,
                time_budget_s = None,
                executor    = None):
    """[PERF] Accepts an optional shared `executor` (ProcessPoolExecutor),
    passed straight through to min_sufficient_depth for every t. Creating a
    process pool has real overhead (process spawn/fork), so reusing one
    pool across all ~10 timesteps in this function -- and ideally across
    all (N,k) pairs in main() -- matters much more than it would for a
    single call.

    [PATCH v6] ceiling_offset defaults to None, meaning "look up
    CEILING_OFFSET_BY_N[N]" -- this is what lets N=10 use a different
    (larger) ceiling than N=4/6/8 without every caller needing to know
    about the per-N table.
    """
    if ceiling_offset is None:
        ceiling_offset = CEILING_OFFSET_BY_N.get(N, CEILING_OFFSET_DEFAULT)
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
            time_budget_s=time_budget_s, t_start=t_start, executor=executor)
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
                mean_reg2=None, std_reg2=None, k_reg2=None, neval_reg2=None):
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
    # [FIX] Previously drawn as a single errorbar() call with one uniform
    # marker style for every point -- meaning an undersampled point (e.g.
    # N=10 at 4/10 steps) rendered identically to a fully-converged one
    # (e.g. N=4 at 10/10 steps), with no visual distinction at all. This
    # silently contradicted the manuscript text, which explicitly
    # describes the N=10 regular-alt point as a partial, 4-of-10-step
    # average. Now split into solid/open groups exactly like the regular
    # series above, using the same UNDERSAMPLED_THRESHOLD.
    if mean_reg2:
        ns_reg2 = ns_done[:len(mean_reg2)]
        mean_reg2_arr = np.array(mean_reg2)
        std_reg2_arr = np.array(std_reg2)
        reg2_open_mask = np.array([
            (neval_reg2[i] if (neval_reg2 is not None and i < len(neval_reg2)) else 999)
            < UNDERSAMPLED_THRESHOLD
            for i in range(len(ns_reg2))
        ])
        lbl = f"Regular-alt ($k={k_reg2}$)" if k_reg2 else "Regular-alt"
        if reg2_open_mask.any():
            solid_idx2 = ~reg2_open_mask
            if solid_idx2.any():
                ax.errorbar(ns_reg2[solid_idx2], mean_reg2_arr[solid_idx2],
                            yerr=std_reg2_arr[solid_idx2], fmt="^", ls="none",
                            color="#009E73", capsize=4, label=lbl)
            ax.errorbar(ns_reg2[reg2_open_mask], mean_reg2_arr[reg2_open_mask],
                        yerr=std_reg2_arr[reg2_open_mask], fmt="^", ls="none",
                        color="#009E73", capsize=4, markerfacecolor="none",
                        label=None if solid_idx2.any() else lbl)
            ax.plot(ns_reg2, mean_reg2, "--", color="#009E73", lw=1.5, zorder=0)
        else:
            ax.errorbar(ns_reg2, mean_reg2, yerr=std_reg2, fmt="^--",
                        color="#009E73", capsize=4, label=lbl)

    ax.set_xlabel("System size $N$ (qubits)")
    ax.set_ylabel(r"Mean sufficient depth $\langle D\rangle_t$")
    ax.set_title("Mean circuit depth required vs. system size\n"
                 "(direct optimization, averaged over Floquet steps)",
                 pad=14)
    ax.set_xticks(ns_done)
    ax.grid(True, ls=":")
    # [FIX] The previous version set ylim from the bare MEANS of only two of
    # the three plotted series (mean_reg, mean_cha) -- it never looked at
    # error-bar extents, and never looked at mean_reg2 (regular-alt) at all.
    # This silently clipped whichever error bar happened to reach highest:
    # at the current data, chaotic N=10 (mean=14.50, std=4.95) has a true
    # upper whisker of 19.45, but the old logic set ylim top to just 16.50
    # (14.50 + 2.0 headroom) -- cutting off nearly 3 units, about a fifth,
    # of that error bar's true extent with no visual indication it happened.
    # Fixed to use the actual max upper-whisker value (mean + std) across
    # ALL THREE series, so no error bar is ever silently truncated.
    _upper_whiskers = []
    for means, stds in ((mean_reg, std_reg), (mean_cha, std_cha)):
        _upper_whiskers += [m + s for m, s in zip(means[:len(ns_done)], stds[:len(ns_done)])]
    if mean_reg2:
        _upper_whiskers += [m + s for m, s in zip(mean_reg2, std_reg2)]
    _ymax = max(_upper_whiskers) if _upper_whiskers else 6.0
    ax.set_ylim(0, _ymax + 1.5)

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
    changing save_figure()'s styling (e.g. this open-marker patch, or the
    y-limit clipping fix) instead of rerunning main().

    [FIX] Previously did not read or pass along the "regular_alt" (k=1.5)
    series saved in the JSON, even though _main_body() always writes it
    when RUN_REGULAR_ALT is True -- so running --replot would silently
    drop the green regular-alt series entirely from the regenerated
    figure. Now reads it when present.
    """
    with open(path) as fh:
        r = json.load(fh)
    reg_alt = r.get("regular_alt")
    save_figure(
        r["system_sizes"],
        r["regular"]["mean_depth"], r["regular"]["std_depth"],
        r["chaotic"]["mean_depth"], r["chaotic"]["std_depth"],
        neval_cha=r["chaotic"]["n_steps_evaluated"],
        neval_reg=r["regular"]["n_steps_evaluated"],
        mean_reg2=reg_alt["mean_depth"] if reg_alt else None,
        std_reg2=reg_alt["std_depth"] if reg_alt else None,
        k_reg2=reg_alt["k"] if reg_alt else None,
        neval_reg2=reg_alt["n_steps_evaluated"] if reg_alt else None,
    )
    print("Replotted from saved JSON -- no computation was rerun.")


# ── Master runner ─────────────────────────────────────────────────────────────

def main():
    system_sizes = [4, 6, 8]
    if TRY_N10:
        system_sizes.append(10)

    checkpoint, entry_ceiling_offsets = _load_checkpoint()

    mean_cha, std_cha, max_cha = [], [], []
    mean_reg, std_reg, max_reg = [], [], []
    neval_cha, neval_reg       = [], []
    mean_reg2, std_reg2, max_reg2, neval_reg2 = [], [], [], []

    os.makedirs("figures", exist_ok=True)

    # [PERF] One process pool for the ENTIRE run, reused across every
    # (N, k) pair and every timestep within it. Spawning a fresh pool per
    # depth_stats() call (or worse, per timestep) would waste most of the
    # parallelization benefit on process-startup overhead instead of on
    # your actual restarts. With N_WORKERS=6 physical cores and
    # OMP/OPENBLAS/MKL_NUM_THREADS pinned to 1 (top of file), this should
    # not oversubscribe the CPU.
    print(f"[PROVENANCE] Running on machine='{_MACHINE_TAG}' cpu='{_CPU_TAG}'",
          flush=True)
    print(f"[PROVENANCE] Checkpoint file for this run: {CHECKPOINT_PATH}",
          flush=True)
    print(f"[PERF] Starting shared process pool with N_WORKERS={N_WORKERS} "
          f"(override via SCALING_N_WORKERS env var).", flush=True)
    _executor = ProcessPoolExecutor(max_workers=N_WORKERS)
    try:
        _main_body(system_sizes, checkpoint, entry_ceiling_offsets,
                   mean_cha, std_cha, max_cha,
                   mean_reg, std_reg, max_reg, neval_cha, neval_reg,
                   mean_reg2, std_reg2, max_reg2, neval_reg2, _executor)
    finally:
        _executor.shutdown(wait=True)
        print("[PERF] Process pool shut down cleanly.", flush=True)


def _main_body(system_sizes, checkpoint, entry_ceiling_offsets,
               mean_cha, std_cha, max_cha,
               mean_reg, std_reg, max_reg, neval_cha, neval_reg,
               mean_reg2, std_reg2, max_reg2, neval_reg2, executor):

    valid_sizes = []
    for N in system_sizes:
        this_n_ceiling = CEILING_OFFSET_BY_N.get(N, CEILING_OFFSET_DEFAULT)
        print(f"\n{'='*55}")
        print(f"N={N} | chaotic (k={K_CHAOTIC}) …  [ceiling_offset={this_n_ceiling}, "
              f"max_depth={N + this_n_ceiling}]")
        key_cha = (N, K_CHAOTIC)
        if key_cha in checkpoint:
            m, s, mx, ne, _ = checkpoint[key_cha]
            print(f"  [RESTORED FROM CHECKPOINT, NOT RECOMPUTED "
                  f"-- n_restarts={N_RESTARTS} matches saved config]")
        else:
            try:
                m, s, mx, ne, conv = depth_stats(N, K_CHAOTIC, executor=executor)
            except MemoryError:
                print(f"  N={N} chaotic: MemoryError — skipping N={N}.", flush=True)
                if N == 10:
                    print("  N=10 skipped (insufficient RAM).  "
                          "Set TRY_N10=False to suppress.", flush=True)
                system_sizes = system_sizes[:system_sizes.index(N)]
                break
            checkpoint[key_cha] = (m, s, mx, ne, conv)
            entry_ceiling_offsets[key_cha] = this_n_ceiling
            _save_checkpoint(checkpoint, entry_ceiling_offsets)
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
                m, s, mx, ne, conv = depth_stats(N, K_REGULAR, executor=executor)
            except MemoryError:
                print(f"  N={N} regular: MemoryError — using placeholder.", flush=True)
                m, s, mx, ne, conv = (mean_reg[-1] if mean_reg else 1.0), 0.0, 1, 0, False
            checkpoint[key_reg] = (m, s, mx, ne, conv)
            entry_ceiling_offsets[key_reg] = this_n_ceiling
            _save_checkpoint(checkpoint, entry_ceiling_offsets)
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
                    m2, s2, mx2, ne2, conv2 = depth_stats(N, K_REGULAR_ALT, executor=executor)
                except MemoryError:
                    print(f"  N={N} regular-alt: MemoryError — using placeholder.", flush=True)
                    m2, s2, mx2, ne2, conv2 = (mean_reg2[-1] if mean_reg2 else 1.0), 0.0, 1, 0, False
                checkpoint[key_reg2] = (m2, s2, mx2, ne2, conv2)
                entry_ceiling_offsets[key_reg2] = this_n_ceiling
                _save_checkpoint(checkpoint, entry_ceiling_offsets)
                print(f"  [COMPUTED FRESH -- checkpointed to {CHECKPOINT_PATH}]")
            print(f"  → mean depth={m2:.2f} ± {s2:.2f}, max={mx2}, over {ne2} steps")
            mean_reg2.append(m2); std_reg2.append(s2); max_reg2.append(mx2); neval_reg2.append(ne2)

        valid_sizes.append(N)

    save_figure(valid_sizes, mean_reg, std_reg, mean_cha, std_cha,
                neval_cha=neval_cha, neval_reg=neval_reg,
                mean_reg2=mean_reg2 if RUN_REGULAR_ALT else None,
                std_reg2=std_reg2 if RUN_REGULAR_ALT else None,
                k_reg2=K_REGULAR_ALT if RUN_REGULAR_ALT else None,
                neval_reg2=neval_reg2 if RUN_REGULAR_ALT else None)

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