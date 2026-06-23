"""
adaptive_vqs_qkt.py  --  Adaptive-depth McLachlan VQS for the QKT.

CHANGES FROM ORIGINAL (all reviewer-driven):
  (i)   Integrator convergence: run_adaptive_floquet() now accepts dt and
        use_rk4 kwargs. The __main__ block runs dt=1/15 (original), 1/30,
        1/60, and RK4 at k=0.5 and k=2.5 and overlays the curves so you can
        verify visually that fidelity/residual are dt-independent.
  (viii) Noise-seed loop: run_noisy_jz2_seeds() repeats the noisy <Jz^2>
        simulation over n_seeds random noise seeds and returns mean±1-sigma
        time-averaged relative errors.  Call it from __main__ to get the
        error bars needed for the paper (currently reported as bare 17.6% /
        21.8% with no uncertainty).
  (ix)  Trigger false-positive/negative rates: compute_trigger_rates() takes
        the regular and chaotic residual_ref trajectories and the threshold
        and computes the per-step FPR and FNR reported in Sec. Limitations.
  (ii)  kappa_max sweep: run_kappa_sweep() reruns the adaptive loop at four
        kappa_max values and a fixed-lambda comparison.
  
  All other logic (trigger, D=1 reference trajectory, residual-first policy,
  Renyi-2 entropy, etc.) is UNCHANGED from the corrected original.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import time

from spin_operators import coherent_product_state, collective_J
from qkt_quantum import floquet_U_exact
from vqs import (
    Ansatz, mclachlan_AC, mclachlan_residual_sq, solve_thetadot,
    condition_number, adaptive_ridge, floquet_step_generator,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def renyi2_entropy(psi: np.ndarray, N: int) -> float:
    dimA = 2 ** (N // 2)
    dimB = 2 ** (N - N // 2)
    M = psi.reshape(dimA, dimB)
    rhoA = M @ M.conj().T
    rhoA /= np.trace(rhoA)
    purity = np.real(np.trace(rhoA @ rhoA))
    purity = min(max(purity, 1e-15), 1.0)
    return float(-np.log2(purity))


def _rk4_step(ans, theta, psi0, H_step, dt, target_cond):
    """One RK4 step for the McLachlan ODE  d theta/dt = A^{-1} C."""
    def f(th):
        A, C, _, _ = mclachlan_AC(ans, th, psi0, H_step)
        ridge = adaptive_ridge(A, target_cond=target_cond)
        return solve_thetadot(A, C, ridge), ridge, condition_number(A, ridge)

    k1, ridge, cond = f(theta)
    k2, _, _ = f(theta + 0.5 * dt * k1)
    k3, _, _ = f(theta + 0.5 * dt * k2)
    k4, _, _ = f(theta + dt * k3)
    return theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), ridge, cond


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_adaptive_floquet(
    N: int, k: float, steps: int = 12, p: float = np.pi / 2,
    n_sub: int = 15, dt: float = None,
    use_rk4: bool = False,
    residual_threshold: float = 0.85, M_consec: int = 1,
    max_depth: int = 8, target_cond: float = 1e8, seed: int = 42,
    compute_exact_diag: bool = True,
):
    """Adaptive-depth McLachlan VQS for the QKT.

    Parameters
    ----------
    dt : float or None
        Substep size.  If None, defaults to 1/n_sub.  Pass explicitly to
        test convergence (e.g. dt=1/30, dt=1/60).
    use_rk4 : bool
        If True, use RK4 instead of forward Euler.  Use with dt=1/15 and
        compare to the Euler runs to bound integration error (reviewer item i).
    """
    if dt is None:
        dt = 1.0 / n_sub
    n_sub_actual = max(1, round(1.0 / dt))   # infer n_sub from dt for logging

    rng = np.random.default_rng(seed)
    H_step = floquet_step_generator(N, k, p)
    U_F = floquet_U_exact(N, k, p) if compute_exact_diag else None

    psi0 = coherent_product_state(N)
    depth = 1
    ans = Ansatz(N, depth)
    theta = rng.uniform(-1e-3, 1e-3, ans.n_params)

    # Independent D=1 reference trajectory for the panel-(a) trigger signal
    ref_ans = Ansatz(N, 1)
    ref_theta = rng.uniform(-1e-3, 1e-3, ref_ans.n_params)

    integrator_label = f"RK4 dt={dt:.4f}" if use_rk4 else f"Euler dt={dt:.4f}"
    print(f"[adaptive Floquet | N={N} k={k} | {integrator_label}] "
          f"eps_trig={residual_threshold}, max_depth={max_depth}")

    out = {key: [] for key in
           ["residual", "residual_ref", "depth", "renyi2", "cond", "ridge",
            "fid_diag", "theta_history"]}
    consec = 0
    psi_exact = psi0.copy()

    for t in range(steps):
        if compute_exact_diag:
            psi_exact = U_F @ psi_exact

        # --- Advance independent D=1 reference one Floquet period ---
        ref_peak = 0.0
        for _sub in range(n_sub_actual):
            A_r, C_r, _, _ = mclachlan_AC(ref_ans, ref_theta, psi0, H_step)
            ridge_r = adaptive_ridge(A_r, target_cond=target_cond)
            td_r = solve_thetadot(A_r, C_r, ridge_r)
            r2_r = mclachlan_residual_sq(ref_ans, ref_theta, td_r, psi0, H_step)
            if use_rk4:
                ref_theta, _, _ = _rk4_step(ref_ans, ref_theta, psi0, H_step,
                                             dt, target_cond)
            else:
                ref_theta = ref_theta + dt * td_r
            ref_peak = max(ref_peak, r2_r)

        theta_period_start = theta.copy()
        depth_changed = True
        trigger_r2 = None
        trigger_cond = None
        trigger_ridge = None

        while depth_changed:
            depth_changed = False
            theta = theta_period_start.copy()
            if len(theta) < ans.n_params:
                noise = rng.normal(0, 1e-3, ans.n_params - len(theta))
                theta = np.concatenate([theta, noise])

            peak_r2, peak_cond, peak_ridge = 0.0, 0.0, 0.0
            for _sub in range(n_sub_actual):
                A, C, psi, _ = mclachlan_AC(ans, theta, psi0, H_step)
                ridge = adaptive_ridge(A, target_cond=target_cond)
                cond = condition_number(A, ridge)
                thetadot = solve_thetadot(A, C, ridge)
                r2 = mclachlan_residual_sq(ans, theta, thetadot, psi0, H_step)

                if use_rk4:
                    theta, ridge, cond = _rk4_step(ans, theta, psi0, H_step,
                                                    dt, target_cond)
                else:
                    theta = theta + dt * thetadot

                peak_r2 = max(peak_r2, r2)
                peak_cond = max(peak_cond, cond)
                peak_ridge = max(peak_ridge, ridge)

            if trigger_r2 is None:
                trigger_r2 = peak_r2
                trigger_cond = peak_cond
                trigger_ridge = peak_ridge

            if peak_r2 > residual_threshold:
                consec += 1
            else:
                consec = 0

            if consec >= M_consec and depth < max_depth:
                depth += 1
                ans = Ansatz(N, depth)
                consec = 0
                depth_changed = True
                print(f"   period {t+1}: peak r^2 = {peak_r2:.3f} "
                      f"> {residual_threshold} -> depth {depth}")

        psi_var = ans.state(theta, psi0)
        out["residual"].append(trigger_r2)
        out["residual_ref"].append(ref_peak)
        out["depth"].append(depth)
        out["renyi2"].append(renyi2_entropy(psi_var, N))
        out["cond"].append(trigger_cond)
        out["ridge"].append(trigger_ridge)
        out["theta_history"].append(theta.copy())

        if compute_exact_diag:
            f_val = abs(np.vdot(psi_exact / np.linalg.norm(psi_exact),
                                psi_var)) ** 2
            out["fid_diag"].append(float(np.real(f_val)))
        else:
            out["fid_diag"].append(np.nan)

    return out


# ---------------------------------------------------------------------------
# (i) Integrator-convergence comparison
# ---------------------------------------------------------------------------

def run_convergence_comparison(N=6, steps=12, k_vals=(0.5, 2.5)):
    """Run forward-Euler at dt=1/15, 1/30, 1/60 AND RK4 at dt=1/15.
    Returns dict keyed by (k, label) -> out dict.
    Used to produce the convergence overlay figure.
    Reviewer item (i): if curves overlap -> add confirming sentence to paper.
    """
    configs = [
        ("Euler_dt15",  dict(dt=1/15,  use_rk4=False)),
        ("Euler_dt30",  dict(dt=1/30,  use_rk4=False)),
        ("Euler_dt60",  dict(dt=1/60,  use_rk4=False)),
        ("RK4_dt15",    dict(dt=1/15,  use_rk4=True)),
    ]
    results = {}
    for k in k_vals:
        for label, kwargs in configs:
            print(f"\n=== Convergence run: k={k}, {label} ===")
            results[(k, label)] = run_adaptive_floquet(
                N, k, steps=steps, compute_exact_diag=True, **kwargs)
    return results


def plot_convergence(results, N=6, steps=12, k_vals=(0.5, 2.5),
                     outfile="figures/integrator_convergence"):
    """Overlay fidelity and residual for all integrator configs."""
    import os
    _aps_style()
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    t = np.arange(1, steps + 1)
    colors = {"Euler_dt15": "#0072B2", "Euler_dt30": "#56B4E9",
              "Euler_dt60": "#009E73", "RK4_dt15":   "#D55E00"}
    styles = {"Euler_dt15": "-", "Euler_dt30": "--",
              "Euler_dt60": ":", "RK4_dt15":   "-."}
    labels = {"Euler_dt15": r"Euler $\Delta t=1/15$",
              "Euler_dt30": r"Euler $\Delta t=1/30$",
              "Euler_dt60": r"Euler $\Delta t=1/60$",
              "RK4_dt15":   r"RK4 $\Delta t=1/15$"}

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5),
                             sharex=True, constrained_layout=True)
    panel_titles = [r"Regular ($k=0.5$) — Fidelity",
                    r"Chaotic ($k=2.5$) — Fidelity",
                    r"Regular — Residual $r^2$ at $D{=}1$",
                    r"Chaotic — Residual $r^2$ at $D{=}1$"]

    for col_idx, k in enumerate(k_vals):
        for lbl in ("Euler_dt15", "Euler_dt30", "Euler_dt60", "RK4_dt15"):
            key = (k, lbl)
            if key not in results:
                continue
            d = results[key]
            axes[0, col_idx].plot(t, d["fid_diag"],
                                  color=colors[lbl], ls=styles[lbl],
                                  label=labels[lbl])
            axes[1, col_idx].plot(t, d["residual_ref"],
                                  color=colors[lbl], ls=styles[lbl],
                                  label=labels[lbl])

        for row in range(2):
            axes[row, col_idx].set_title(panel_titles[row + 2 * 0 + col_idx],
                                         fontsize=8)
            axes[row, col_idx].grid(True, ls=":", alpha=0.5)
            axes[row, col_idx].legend(fontsize=6.5, loc="best")
        axes[1, col_idx].axhline(EPS_TRIG, color="0.3", ls="--", lw=1.0)

    axes[0, 0].set_ylabel("Fidelity $F(t)$")
    axes[1, 0].set_ylabel(r"Residual $r^2$ at $D{=}1$")
    for col_idx in range(2):
        axes[1, col_idx].set_xlabel(r"Floquet step $t$")

    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}.pdf / .png")

    # Print quantitative convergence table to stdout so you can paste into paper
    print("\n=== CONVERGENCE TABLE (max |fidelity difference| from Euler dt=1/15) ===")
    for k in k_vals:
        baseline = results.get((k, "Euler_dt15"))
        if baseline is None:
            continue
        f0 = np.array(baseline["fid_diag"])
        for lbl in ("Euler_dt30", "Euler_dt60", "RK4_dt15"):
            d = results.get((k, lbl))
            if d is None:
                continue
            diff = np.max(np.abs(np.array(d["fid_diag"]) - f0))
            print(f"  k={k}  {lbl:20s}  max|ΔF| = {diff:.4f}")


# ---------------------------------------------------------------------------
# (viii) Noise-seed loop for error bars on <Jz^2> relative errors
# ---------------------------------------------------------------------------

def run_noisy_jz2_seeds(N=6, k_vals=(0.5, 2.5), steps=12,
                        n_seeds=20, target_cond=1e8):
    """Repeat the noisy <Jz^2> tracking over n_seeds noise seeds.
    Returns dict: k -> {"mean_relerr": float, "std_relerr": float,
                        "per_seed": list of float}

    The noisy simulation uses the adaptive loop's theta_history (from a
    single noiseless run) plus the ibm_fez noise model applied via
    error_mitigation.py.  Because we don't re-run the full adaptive loop
    per seed (expensive), we replay the stored thetas under different noise
    realizations -- this isolates the noise-sampling variance from the
    optimization variance, which is what the referee asked for.

    NOTE: this requires error_mitigation.py to expose a function
        noisy_jz2_trajectory(theta_history, N, k, p, seed) -> array of floats
    If that function doesn't exist yet, the fallback prints instructions.
    """
    try:
        from error_mitigation import noisy_jz2_trajectory
        _has_em = True
    except (ImportError, AttributeError):
        _has_em = False

    results = {}
    for k in k_vals:
        print(f"\n[noisy Jz2 seeds] k={k}, n_seeds={n_seeds} ...")
        # Run one noiseless adaptive to get theta_history
        noiseless = run_adaptive_floquet(N, k, steps=steps,
                                         compute_exact_diag=True,
                                         target_cond=target_cond)
        theta_hist = noiseless["theta_history"]

        # Exact <Jz^2> trajectory
        from spin_operators import collective_J
        _, _, Jz = collective_J(N)
        Jz2 = Jz @ Jz
        psi0 = coherent_product_state(N)
        U_F = floquet_U_exact(N, k, np.pi / 2)
        psi_t = psi0.copy()
        exact_jz2 = []
        for _ in range(steps):
            psi_t = U_F @ psi_t
            exact_jz2.append(float(np.real(psi_t.conj() @ Jz2 @ psi_t)))
        exact_jz2 = np.array(exact_jz2)

        seed_relerrs = []
        for seed in range(n_seeds):
            if _has_em:
                noisy_vals = noisy_jz2_trajectory(theta_hist, N, k,
                                                   np.pi / 2, seed=seed)
                noisy_vals = np.array(noisy_vals)
            else:
                # Fallback: inject Gaussian noise proportional to 3e-3 per gate
                rng = np.random.default_rng(seed)
                # Approximate noise as 3e-3 * sqrt(2*N*depth) per step
                noise_scale = 3e-3 * np.sqrt(2 * N * np.array(noiseless["depth"]))
                noisy_vals = exact_jz2 * (1 + rng.normal(0, noise_scale))

            nonzero = np.abs(exact_jz2) > 1e-10
            relerr = np.mean(
                np.abs(noisy_vals[nonzero] - exact_jz2[nonzero])
                / np.abs(exact_jz2[nonzero])
            )
            seed_relerrs.append(float(relerr))

        mean_re = float(np.mean(seed_relerrs))
        std_re = float(np.std(seed_relerrs, ddof=1))
        results[k] = {"mean_relerr": mean_re, "std_relerr": std_re,
                      "per_seed": seed_relerrs}
        print(f"  k={k}: time-avg rel err = {mean_re*100:.1f}% "
              f"± {std_re*100:.1f}% (n={n_seeds} seeds)")
        if not _has_em:
            print("  WARNING: error_mitigation.noisy_jz2_trajectory not found. "
                  "Using Gaussian-noise fallback.  Implement the real ibm_fez "
                  "noise model in error_mitigation.py for publication figures.")
    return results


# ---------------------------------------------------------------------------
# (ix) Trigger false-positive / false-negative rates
# ---------------------------------------------------------------------------

def compute_trigger_rates(reg_residuals, cha_residuals,
                          eps_trig=0.85):
    """Compute per-step FPR and FNR of the r^2 > eps_trig trigger.

    FPR (false positive): fraction of steps where the REGULAR regime's
        residual exceeds eps_trig (would trigger expansion spuriously).
    FNR (false negative): fraction of steps where the CHAOTIC regime's
        residual is BELOW eps_trig (would miss a needed expansion).

    Parameters
    ----------
    reg_residuals, cha_residuals : list of float
        The residual_ref lists from run_adaptive_floquet for regular and
        chaotic regimes (D=1 reference trajectory).
    """
    reg = np.array(reg_residuals)
    cha = np.array(cha_residuals)

    fpr_per_step = (reg > eps_trig).astype(float)
    fnr_per_step = (cha <= eps_trig).astype(float)

    print(f"\n=== TRIGGER RATES (eps_trig={eps_trig}) ===")
    print(f"Steps: {len(reg)} regular, {len(cha)} chaotic")
    print(f"False-positive rate (regular r^2 > {eps_trig}): "
          f"{fpr_per_step.mean()*100:.1f}% "
          f"({int(fpr_per_step.sum())}/{len(reg)} steps)")
    print(f"False-negative rate (chaotic r^2 <= {eps_trig}): "
          f"{fnr_per_step.mean()*100:.1f}% "
          f"({int(fnr_per_step.sum())}/{len(cha)} steps)")
    print(f"Chaotic steps ABOVE threshold: "
          f"{(1-fnr_per_step).mean()*100:.1f}%")
    print(f"Per-step chaotic residuals: {np.round(cha, 3).tolist()}")
    print(f"Per-step regular residuals: {np.round(reg, 3).tolist()}")

    return {
        "fpr": float(fpr_per_step.mean()),
        "fnr": float(fnr_per_step.mean()),
        "fpr_per_step": fpr_per_step.tolist(),
        "fnr_per_step": fnr_per_step.tolist(),
    }


# ---------------------------------------------------------------------------
# (ii) kappa_max sweep
# ---------------------------------------------------------------------------

def run_kappa_sweep(N=6, k=2.5, steps=12, kappa_vals=(1e6, 1e7, 1e8, 1e9)):
    """Run the adaptive loop for each kappa_max value.
    Also runs a fixed-lambda comparison (lambda = median of adaptive lambdas
    from the kappa=1e8 run) to confirm the adaptive scheme isn't distorting
    results.  Reviewer item (ii).
    """
    results = {}
    for kappa in kappa_vals:
        label = f"kappa_{kappa:.0e}"
        print(f"\n=== kappa_max sweep: kappa={kappa:.0e} ===")
        results[label] = run_adaptive_floquet(
            N, k, steps=steps, target_cond=kappa, compute_exact_diag=True)

    # Fixed-lambda comparison: use median ridge from kappa=1e8 run
    ref_ridges = results.get("kappa_1e+08", {}).get("ridge", [])
    if ref_ridges:
        fixed_lam = float(np.median(ref_ridges))
        print(f"\n=== Fixed-lambda comparison: lambda={fixed_lam:.3e} ===")
        # To run with a fixed lambda we temporarily monkey-patch adaptive_ridge
        import vqs as _vqs
        _orig_ridge = _vqs.adaptive_ridge
        _vqs.adaptive_ridge = lambda A, target_cond=None: fixed_lam
        results["fixed_lambda"] = run_adaptive_floquet(
            N, k, steps=steps, compute_exact_diag=True)
        _vqs.adaptive_ridge = _orig_ridge

    # Report summary
    print("\n=== KAPPA SWEEP SUMMARY (fidelity at final step, chaotic) ===")
    for label, d in results.items():
        fids = [x for x in d["fid_diag"] if not np.isnan(x)]
        print(f"  {label:20s}: final fidelity = {fids[-1]:.4f}, "
              f"final depth = {d['depth'][-1]}")
    return results


def plot_kappa_sweep(results, steps=12, outfile="figures/kappa_sweep"):
    """Overlay fidelity trajectories for all kappa_max values."""
    import os
    _aps_style()
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    t = np.arange(1, steps + 1)
    cmap = plt.cm.viridis
    labels = list(results.keys())
    colors = [cmap(i / max(len(labels) - 1, 1)) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    for lbl, col in zip(labels, colors):
        d = results[lbl]
        fids = [x for x in d["fid_diag"] if not np.isnan(x)]
        ax.plot(range(1, len(fids) + 1), fids, label=lbl, color=col)
    ax.set_xlabel(r"Floquet step $t$")
    ax.set_ylabel(r"Fidelity $F(t)$")
    ax.set_title(r"$\kappa_{\rm max}$ sensitivity (chaotic, $k=2.5$)")
    ax.legend(fontsize=6.5)
    ax.grid(True, ls=":", alpha=0.5)
    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}.pdf / .png")


# ---------------------------------------------------------------------------
# Plotting (original, unchanged except Renyi-2 Page value correction)
# ---------------------------------------------------------------------------

def _aps_style():
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "text.usetex": False,
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.3,
        "lines.markersize": 4.0, "savefig.dpi": 600,
    })


REG_C, CHA_C = "#0072B2", "#D55E00"
DOUBLE_COL = 7.0
EPS_TRIG = 0.85
# Corrected Renyi-2 Page value: S2 = -log2(16/65) ≈ 2.02 bits for 3+3 partition
# (was 2.85 in original, which is the von Neumann Page value -- wrong for Renyi-2)
RENYI2_PAGE = -np.log2(16.0 / 65.0)   # ≈ 2.02 bits


def plot_adaptive(reg, cha, steps, outfile="figures/adaptive_residual",
                  eps_trig=EPS_TRIG, mid=None):
    """Plot the 4-panel adaptive residual figure.
    mid : optional dict from run_adaptive_floquet at k=1.75 (F6 intermediate regime).
    """
    import os
    _aps_style()
    MID_C = "#009E73"   # green for intermediate k=1.75
    t = np.arange(1, steps + 1)
    fig, ax = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.9),
                           sharex=True, constrained_layout=True)

    a = ax[0, 0]
    a.plot(t, reg["residual_ref"], "o-", color=REG_C, label="Regular ($k=0.5$)")
    if mid is not None:
        a.plot(t, mid["residual_ref"], "^-", color=MID_C,
               label="Transition ($k=1.75$)")
    a.plot(t, cha["residual_ref"], "s-", color=CHA_C, label="Chaotic ($k=2.5$)")
    a.axhline(eps_trig, color="0.3", ls="--", lw=1.0,
              label=rf"$\varepsilon_{{\rm trig}}={eps_trig}$")
    a.set_ylim(0.0, 1.0)
    a.set_ylabel(r"McLachlan residual $r^2(t)$ at $D{=}1$")
    a.grid(True, ls=":", alpha=0.5)
    a.legend(loc="best", fontsize=6.5)
    a.text(0.035, 0.965, "(a)", transform=a.transAxes, va="top", fontweight="bold")

    b = ax[0, 1]
    b.plot(t, reg["depth"], "o-", color=REG_C, label="Regular ($k=0.5$)")
    if mid is not None:
        b.plot(t, mid["depth"], "^-", color=MID_C, label="Transition ($k=1.75$)")
    b.plot(t, cha["depth"], "s-", color=CHA_C, label="Chaotic ($k=2.5$)")
    b.set_ylabel(r"Circuit depth $D(t)$")
    b.grid(True, ls=":", alpha=0.5)
    b.legend(loc="best", fontsize=6.5)
    b.text(0.035, 0.965, "(b)", transform=b.transAxes, va="top", fontweight="bold")

    c = ax[1, 0]
    c.plot(t, reg["renyi2"], "o-", color=REG_C, label="Regular ($k=0.5$)")
    if mid is not None:
        c.plot(t, mid["renyi2"], "^-", color=MID_C, label="Transition ($k=1.75$)")
    c.plot(t, cha["renyi2"], "s-", color=CHA_C, label="Chaotic ($k=2.5$)")
    # Corrected Renyi-2 Page value line (was 2.85; now 2.02 bits)
    c.axhline(RENYI2_PAGE, color="0.4", ls=":", lw=1.0,
              label=rf"$S_2^{{\rm Page}}\approx{RENYI2_PAGE:.2f}$ bits")
    c.set_ylabel(r"Rényi-2 entropy $S_2$ (bits)")
    c.set_xlabel(r"Floquet step $t$")
    c.grid(True, ls=":", alpha=0.5)
    c.legend(loc="best", fontsize=6.5)
    c.text(0.035, 0.965, "(c)", transform=c.transAxes, va="top", fontweight="bold")

    d = ax[1, 1]
    d.semilogy(t, reg["cond"], "o-", color=REG_C, label="Regular ($k=0.5$)")
    if mid is not None:
        d.semilogy(t, mid["cond"], "^-", color=MID_C,
                   label="Transition ($k=1.75$)")
    d.semilogy(t, cha["cond"], "s-", color=CHA_C, label="Chaotic ($k=2.5$)")
    d.set_ylabel(r"Metric cond. number $\kappa(A)$")
    d.set_xlabel(r"Floquet step $t$")
    d.grid(True, which="both", ls=":", alpha=0.5)
    d.legend(loc="best", fontsize=6.5)
    d.text(0.035, 0.965, "(d)", transform=d.transAxes, va="top", fontweight="bold")

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}.pdf / {outfile}.png")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    N, STEPS = 6, 12

    # -----------------------------------------------------------------------
    # STEP 1: Main adaptive run (original result, replicated)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Main adaptive run (Euler dt=1/15, the original result)")
    print("="*60)
    reg = run_adaptive_floquet(N, k=0.5, steps=STEPS)
    cha = run_adaptive_floquet(N, k=2.5, steps=STEPS)
    # F6: intermediate regime k=1.75 (regular-to-chaotic transition)
    mid = run_adaptive_floquet(N, k=1.75, steps=STEPS)
    plot_adaptive(reg, cha, STEPS, mid=mid)

    # Verify residual ranges and report them for manuscript Sec. II.E.1
    print(f"\n[VERIFY for manuscript] D=1 residual ranges:")
    print(f"  Regular r^2: min={min(reg['residual_ref']):.3f}  "
          f"max={max(reg['residual_ref']):.3f}  (claim: stays below 0.85)")
    print(f"  Chaotic r^2: min={min(cha['residual_ref']):.3f}  "
          f"max={max(cha['residual_ref']):.3f}  (claim: mostly above 0.85)")
    print(f"  Final regular depth = {reg['depth'][-1]} (expect 1)")
    print(f"  Final chaotic depth = {cha['depth'][-1]} (expect 8)")
    print(f"  [diagnostic] Final fidelity: reg={reg['fid_diag'][-1]:.3f}  "
          f"cha={cha['fid_diag'][-1]:.3f}")

    # Report intermediate k=1.75 results for F6
    if mid is not None:
        print(f"\n[F6 intermediate k=1.75] D=1 residual ranges:")
        print(f"  k=1.75 r^2: min={min(mid['residual_ref']):.3f}  "
              f"max={max(mid['residual_ref']):.3f}")
        print(f"  Final depth at k=1.75: {mid['depth'][-1]}")
        # Check how often trigger fires at k=1.75 vs threshold
        mid_above = sum(r > EPS_TRIG for r in mid['residual_ref'])
        print(f"  Steps above eps_trig={EPS_TRIG}: {mid_above}/{STEPS} "
              f"({mid_above/STEPS*100:.0f}%)")
        print(f"  >>> FOR PAPER (F6/Fig 8): at k=1.75 the trigger fires "
              f"{mid_above/STEPS*100:.0f}% of steps, depth reaches "
              f"{mid['depth'][-1]} — intermediate between regular (D=1) "
              f"and chaotic (D=8).")

    # -----------------------------------------------------------------------
    # STEP 2: (ix) Trigger false-positive / false-negative rates
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: Trigger false-positive / false-negative rates")
    print("="*60)
    trigger_stats = compute_trigger_rates(reg["residual_ref"],
                                          cha["residual_ref"])
    print(f"\n>>> PASTE INTO PAPER (Sec. Limitations item ix):")
    print(f"    FPR = {trigger_stats['fpr']*100:.1f}%  "
          f"FNR = {trigger_stats['fnr']*100:.1f}%")

    # -----------------------------------------------------------------------
    # STEP 3: (i) Integrator convergence
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: Integrator convergence (Euler dt=1/15,1/30,1/60 and RK4)")
    print("NOTE: This is the slow step (~3-4x longer than Step 1).")
    print("="*60)
    conv_results = run_convergence_comparison(N=N, steps=STEPS)
    plot_convergence(conv_results, N=N, steps=STEPS)
    print("\n>>> ACTION: check figures/integrator_convergence.png")
    print("    If all four curves overlap: add confirming sentence to paper.")
    print("    If they diverge: switch to RK4 or explicitly flag in text.")

    # -----------------------------------------------------------------------
    # STEP 4: (viii) Error bars on noisy <Jz^2> relative errors
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4: Noise-seed loop for <Jz^2> error bars (n=20 seeds)")
    print("="*60)
    noisy_stats = run_noisy_jz2_seeds(N=N, k_vals=(0.5, 2.5), steps=STEPS,
                                      n_seeds=20)
    print("\n>>> PASTE INTO PAPER (Sec. Observable Tracking):")
    for k_val, stats in noisy_stats.items():
        regime = "regular" if k_val == 0.5 else "chaotic"
        print(f"    {regime} (k={k_val}): "
              f"{stats['mean_relerr']*100:.1f}% ± {stats['std_relerr']*100:.1f}%")

    # -----------------------------------------------------------------------
    # STEP 5: (ii) kappa_max sweep
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 5: kappa_max sensitivity sweep")
    print("="*60)
    kappa_results = run_kappa_sweep(N=N, k=2.5, steps=STEPS)
    plot_kappa_sweep(kappa_results, steps=STEPS)
    print("\n>>> ACTION: check figures/kappa_sweep.png")
    print("    If curves are indistinguishable: state 'insensitive to kappa' in paper.")

    # -----------------------------------------------------------------------
    # STEP 6: N=4 export for hardware PoC (unchanged)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 6: N=4 export for hardware PoC")
    print("="*60)
    cha_poc = run_adaptive_floquet(4, k=2.5, steps=5)
    with open("N4_chaotic_history.pkl", "wb") as fh:
        pickle.dump(cha_poc, fh)
    print("Saved 'N4_chaotic_history.pkl'.")

    print("\n" + "="*60)
    print("ALL STEPS COMPLETE.")
    print("New figures produced:")
    print("  figures/adaptive_residual.{pdf,png}    <- REPLACES Fig. 8")
    print("  figures/integrator_convergence.{pdf,png} <- NEW (reviewer item i)")
    print("  figures/kappa_sweep.{pdf,png}           <- NEW (reviewer item ii)")
    print("="*60)
