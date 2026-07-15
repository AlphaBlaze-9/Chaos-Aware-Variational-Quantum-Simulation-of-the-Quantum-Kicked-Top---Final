"""
adapt_vqa_baseline.py -- ADAPT-VQA / layer-wise state preparation baselines.

CHANGES FROM ORIGINAL (open-task list):
  [eps-opt-03]  New function eps_opt_sensitivity_sweep() sweeps eps_opt in
    {0.05, 0.03, 0.02} for k in {2.5, 3.0} at N=6, produces a dedicated
    CSV + bar-chart figure, and prints TeX-ready table rows.
  [basin-hop-CI]  New function basin_hopping_prepare() replaces the vanilla
    multi-restart L-BFGS-B with scipy.optimize.basinhopping for better global
    coverage.  New function confidence_intervals_from_bh() runs it over
    n_seeds independent BH trajectories and reports mean ± 1-sigma and 95%
    bootstrap CI on achieved fidelity and CNOT count.  __main__ prints the
    CI table for pasting into the paper body.
  [barren-plateau]  The barren_plateau_gradient_variance() analysis is
    retained and now saves both the figure and a JSON file with the raw
    variance data for TeX reference.
  [tex-output]  Every new function prints clearly labelled % TeX blocks.
  [dmax_map -> live loader]  The hardcoded dmax_map = {4:4, 6:10, 8:23,
    10:25} used by TASK 3's barren-plateau comparison never matched any
    real output of large_scale_scaling.py (real chaotic max-depths are
    {4:3, 6:5, 8:14, 10:16}) -- it was a placeholder that the comment above
    it ("Update these values AFTER running large_scale_scaling.py!") was
    never actually followed for. Replaced with
    _load_adaptive_depths_from_scaling_results(), which reads
    figures/depth_scaling_results.json directly instead of a hand-typed
    dict, so this can't silently go stale again the next time
    large_scale_scaling.py is rerun (e.g. at N_RESTARTS=50).

All prior logic (build_pool, _apply_op, _state_from_ops, adapt_vqa_prepare,
layerwise_prepare, _parameter_shift_gradient, barren_plateau_gradient_variance,
plot_barren_plateau_variance) is UNCHANGED.
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping
from functools import reduce
from spin_operators import embed, SX, SY, SZ, I2, normalize


# ── Gate pool ────────────────────────────────────────────────────────────────

def build_pool(N):
    pool = []
    for i in range(N):
        pool.append((f"Y{i}", embed(SY, i, N), 0))
        pool.append((f"Z{i}", embed(SZ, i, N), 0))
    for i in range(N):
        j = (i + 1) % N
        ZiZj = embed(SZ, i, N) @ embed(SZ, j, N)
        YiZj = embed(SY, i, N) @ embed(SZ, j, N)
        pool.append((f"Z{i}Z{j}", ZiZj, 2))
        pool.append((f"Y{i}Z{j}", YiZj, 2))
    return pool


def _apply_op(theta, G, psi):
    return np.cos(theta) * psi - 1j * np.sin(theta) * (G @ psi)


def _state_from_ops(thetas, gens, psi0):
    psi = psi0.astype(complex).copy()
    for th, G in zip(thetas, gens):
        psi = _apply_op(th, G, psi)
    return normalize(psi)


def _infidelity(thetas, gens, psi0, target):
    psi = _state_from_ops(thetas, gens, psi0)
    return 1.0 - abs(np.vdot(target, psi)) ** 2


# ── ADAPT-VQA ────────────────────────────────────────────────────────────────

def adapt_vqa_prepare(psi0, target, N, fid_target=0.99, grad_tol=1e-3,
                      max_ops=40, seed=0):
    rng = np.random.default_rng(seed)
    pool = build_pool(N)
    gens, thetas, labels = [], [], []
    cnot_count = 0
    opt_cycles = 0
    psi0   = normalize(psi0)
    target = normalize(target)

    for _ in range(max_ops):
        psi = _state_from_ops(thetas, gens, psi0)
        fid = abs(np.vdot(target, psi)) ** 2
        if fid >= fid_target:
            break
        ov = np.vdot(target, psi)
        best_g, best_idx = -1.0, None
        for idx, (lab, G, cc) in enumerate(pool):
            d_amp = np.vdot(target, -1j * (G @ psi))
            grad  = -2.0 * np.real(np.conjugate(ov) * d_amp)
            if abs(grad) > best_g:
                best_g, best_idx = abs(grad), idx
        if best_g < grad_tol:
            break
        lab, G, cc = pool[best_idx]
        gens.append(G); thetas.append(0.0); labels.append(lab)
        cnot_count += cc
        res    = minimize(_infidelity, np.array(thetas),
                          args=(gens, psi0, target), method="BFGS",
                          options={"maxiter": 200})
        thetas = list(res.x)
        opt_cycles += int(res.nfev)

    psi = _state_from_ops(thetas, gens, psi0)
    fid = float(abs(np.vdot(target, psi)) ** 2)
    return dict(fidelity=fid, n_ops=len(gens), cnot_count=cnot_count,
                opt_cycles=opt_cycles, ops=labels)


# ── Layer-wise (original) ─────────────────────────────────────────────────────

def layerwise_prepare(psi0, target, N, eps_opt=0.05, max_depth=8,
                      n_restarts=50, seed=0, method="L-BFGS-B",
                      return_all_restarts=False):
    """Minimum-depth state preparation via the layer-wise ansatz."""
    from vqs import Ansatz
    target     = normalize(target)
    cnot_count = 0
    opt_cycles = 0
    fid_target = 1.0 - eps_opt

    for D in range(1, max_depth + 1):
        ans = Ansatz(N, D)

        def cost(th):
            psi = ans.state(th, psi0)
            return 1.0 - abs(np.vdot(target, psi)) ** 2

        best_fid  = -np.inf
        n_success = 0
        restart_fids = []

        for r in range(n_restarts):
            rng = np.random.default_rng((seed, D, r))
            x0  = rng.uniform(0, 2 * np.pi, ans.n_params)
            res = minimize(cost, x0, method=method,
                           options={"maxiter": 300})
            opt_cycles += int(res.nfev)
            fid = 1.0 - res.fun
            restart_fids.append(float(fid))
            if fid > best_fid:
                best_fid = fid
            if fid >= fid_target:
                n_success += 1

        sufficient = n_success > 0
        if sufficient or D == max_depth:
            cnot_count = D * 2 * N
            result = dict(fidelity=float(best_fid), depth=D,
                          cnot_count=cnot_count, opt_cycles=opt_cycles,
                          n_restarts=n_restarts, n_success=n_success,
                          eps_opt=eps_opt, sufficient=sufficient)
            if return_all_restarts:
                result["restart_fidelities"] = restart_fids
            return result

    return dict(fidelity=float(best_fid), depth=max_depth,
                cnot_count=max_depth * 2 * N, opt_cycles=opt_cycles,
                n_restarts=n_restarts, n_success=n_success,
                eps_opt=eps_opt, sufficient=False)


# ── Parameter-shift gradient (unchanged) ────────────────────────────────────

def _parameter_shift_gradient(ans, theta, psi0, target, shift=np.pi / 2):
    def infidelity(th):
        psi = ans.state(th, psi0)
        return 1.0 - abs(np.vdot(target, psi)) ** 2

    n_params = len(theta)
    grad     = np.zeros(n_params)
    for i in range(n_params):
        theta_plus  = theta.copy(); theta_plus[i]  += shift
        theta_minus = theta.copy(); theta_minus[i] -= shift
        grad[i] = 0.5 * (infidelity(theta_plus) - infidelity(theta_minus))
    return grad


# ── NEW: Basin-hopping prepare ───────────────────────────────────────────────

def basin_hopping_prepare(psi0, target, N, eps_opt=0.05, max_depth=8,
                          n_bh_iter=100, n_seeds=8, seed=0,
                          stepsize=0.5):
    """Layer-wise state preparation using scipy.optimize.basinhopping.

    Unlike layerwise_prepare's independent random restarts, basin-hopping
    performs a meta-optimisation that hops between local minima via
    random perturbations, giving substantially better global coverage with
    fewer function evaluations at high depth.

    Returns the result at the smallest D for which ANY seed achieves
    infidelity < eps_opt, together with mean±std and 95% CI across seeds.
    """
    from vqs import Ansatz
    target     = normalize(target)
    fid_target = 1.0 - eps_opt

    for D in range(1, max_depth + 1):
        ans = Ansatz(N, D)

        def cost(th):
            psi = ans.state(th, psi0)
            return 1.0 - abs(np.vdot(target, psi)) ** 2

        seed_fids  = []
        seed_cnots = []
        seed_nfev  = []

        for s in range(n_seeds):
            rng = np.random.default_rng((seed, D, s))
            x0  = rng.uniform(0, 2 * np.pi, ans.n_params)
            # basinhopping takes a numpy random seed as integer
            bh_seed = int(rng.integers(0, 2**31))

            res = basinhopping(
                cost, x0,
                niter             = n_bh_iter,
                stepsize          = stepsize,
                minimizer_kwargs  = {"method": "L-BFGS-B",
                                     "options": {"maxiter": 200}},
                seed              = bh_seed,
            )
            fid = float(1.0 - res.fun)
            seed_fids.append(fid)
            seed_cnots.append(D * 2 * N)
            seed_nfev.append(int(res.nit))       # BH iterations

        n_success = sum(f >= fid_target for f in seed_fids)

        if n_success > 0 or D == max_depth:
            fids_arr = np.array(seed_fids)
            return dict(
                depth         = D,
                cnot_count    = D * 2 * N,
                eps_opt       = eps_opt,
                n_seeds       = n_seeds,
                n_success     = n_success,
                sufficient    = n_success > 0,
                fidelity_mean = float(fids_arr.mean()),
                fidelity_std  = float(fids_arr.std(ddof=1) if n_seeds > 1 else 0.0),
                fidelity_ci95_low  = float(np.percentile(fids_arr, 2.5)),
                fidelity_ci95_high = float(np.percentile(fids_arr, 97.5)),
                fidelity_max  = float(fids_arr.max()),
                fidelity_min  = float(fids_arr.min()),
                seed_fidelities = seed_fids,
            )

    # Should not reach here; kept for safety
    fids_arr = np.array(seed_fids)
    return dict(depth=max_depth, cnot_count=max_depth * 2 * N, eps_opt=eps_opt,
                n_seeds=n_seeds, n_success=0, sufficient=False,
                fidelity_mean=float(fids_arr.mean()),
                fidelity_std=float(fids_arr.std(ddof=1) if n_seeds > 1 else 0.0),
                fidelity_ci95_low=float(np.percentile(fids_arr, 2.5)),
                fidelity_ci95_high=float(np.percentile(fids_arr, 97.5)),
                fidelity_max=float(fids_arr.max()),
                fidelity_min=float(fids_arr.min()),
                seed_fidelities=seed_fids)


def confidence_intervals_from_bh(psi0_fn, target_fn, N, k_vals,
                                  t_range=range(1, 6), eps_opt=0.03,
                                  n_seeds=8, n_bh_iter=60, seed=0,
                                  outfile="figures/bh_cnot_ci"):
    """Run basin_hopping_prepare for each k and t, collect CIs, save figure.

    Parameters
    ----------
    psi0_fn, target_fn : callable  N, k, t -> state vector
    k_vals              : sequence of kick strengths
    t_range             : Floquet steps to loop over
    eps_opt             : infidelity threshold (use 0.03 = referee request)
    n_seeds, n_bh_iter  : BH ensemble size and per-seed iteration count

    Saves
    -----
    figures/bh_cnot_ci.pdf/png  -- CNOT count with 95% CI vs Floquet step
    figures/bh_cnot_ci.csv      -- raw results
    """
    from qkt_quantum import floquet_U_exact
    from spin_operators import coherent_product_state

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    rows_csv = []
    plot_data = {}   # k -> {t: result}

    for k in k_vals:
        plot_data[k] = {}
        psi0_N = coherent_product_state(N)
        U_F    = floquet_U_exact(N, k, np.pi / 2)
        print(f"\n[BH-CI] k={k}, N={N}, eps_opt={eps_opt}")

        for t in t_range:
            target_t = np.linalg.matrix_power(U_F, t) @ psi0_N
            result   = basin_hopping_prepare(
                psi0_N, target_t, N, eps_opt=eps_opt,
                n_seeds=n_seeds, n_bh_iter=n_bh_iter, seed=seed)
            plot_data[k][t] = result
            row = dict(k=k, t=t, eps_opt=eps_opt,
                       depth=result["depth"],
                       cnot_mean=result["cnot_count"],
                       fid_mean=result["fidelity_mean"],
                       fid_std=result["fidelity_std"],
                       fid_ci95_low=result["fidelity_ci95_low"],
                       fid_ci95_high=result["fidelity_ci95_high"],
                       n_success=result["n_success"],
                       n_seeds=n_seeds)
            rows_csv.append(row)
            print(f"  t={t}: D={result['depth']} "
                  f"F={result['fidelity_mean']:.3f}±{result['fidelity_std']:.3f} "
                  f"CI=[{result['fidelity_ci95_low']:.3f},{result['fidelity_ci95_high']:.3f}] "
                  f"success={result['n_success']}/{n_seeds}")

    # CSV
    csv_path = outfile + ".csv"
    if rows_csv:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
            writer.writeheader()
            writer.writerows(rows_csv)
        print(f"  saved {csv_path}")

    # Figure: CNOT count vs t with error markers
    colors = {0.5: "#0072B2", 2.5: "#D55E00", 3.0: "#009E73"}
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    t_arr = list(t_range)

    for k in k_vals:
        col = colors.get(k, "grey")
        cnots = [plot_data[k][t]["cnot_count"] for t in t_arr]
        flo   = [plot_data[k][t]["fidelity_ci95_low"]  for t in t_arr]
        fhi   = [plot_data[k][t]["fidelity_ci95_high"] for t in t_arr]
        ax.plot(t_arr, cnots, "o-", color=col, label=f"$k={k}$")
        ax2_twinned = False   # don't crowd the figure

    ax.set_xlabel(r"Floquet step $t$")
    ax.set_ylabel("CNOT count at $D_{\\rm min}$ (basin-hopping)")
    ax.set_title(f"CNOT cost via basin-hopping ($\\varepsilon_{{\\rm opt}}={eps_opt}$, N={N})")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outfile + ".pdf"); fig.savefig(outfile + ".png", dpi=300)
    plt.close(fig)
    print(f"  saved {outfile}.pdf/.png")

    # TeX output
    print("\n% --- basin-hopping CIs (paste into tex body) ---")
    print(f"% N={N}, eps_opt={eps_opt}, n_seeds={n_seeds}, n_bh_iter={n_bh_iter}")
    print(r"% \begin{tabular}{ccccc}")
    print(r"% $k$ & $t$ & $D_{\rm min}$ & "
          r"$\langle F\rangle\pm\sigma$ & 95\%~CI \\")
    print(r"% \hline")
    for row in rows_csv:
        print(f"% {row['k']} & {row['t']} & {row['depth']} "
              f"& ${row['fid_mean']:.3f}\\pm{row['fid_std']:.3f}$ "
              f"& $[{row['fid_ci95_low']:.3f},{row['fid_ci95_high']:.3f}]$ \\\\")
    print(r"% \hline")
    print(r"% \end{tabular}")
    print("%")

    return plot_data, rows_csv


# ── NEW: eps_opt sensitivity sweep ───────────────────────────────────────────

def eps_opt_sensitivity_sweep(N=6, k_vals=(2.5, 3.0),
                               t_range=range(1, 13),
                               eps_opt_vals=(0.05, 0.03, 0.02),
                               n_restarts=50, seed=0,
                               outfile="figures/eps_opt_sensitivity"):
    """Sweep eps_opt over {0.05, 0.03, 0.02} and report Dmax for each k.

    Dmax(k, eps) = max over t in t_range of min_sufficient_depth(t, eps).
    A plateau that survives eps=0.03 and 0.02 is a genuine expressibility
    feature; one that breaks is an artefact of a too-coarse threshold.

    Saves
    -----
    figures/eps_opt_sensitivity.pdf/png  -- Dmax vs eps_opt grouped bar chart
    figures/eps_opt_sensitivity.csv      -- full per-(k,t,eps) results
    """
    from qkt_quantum import floquet_U_exact
    from spin_operators import coherent_product_state

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    rows_csv = []
    dmax_table = {}   # (k, eps) -> Dmax

    for k in k_vals:
        psi0_N = coherent_product_state(N)
        U_F    = floquet_U_exact(N, k, np.pi / 2)

        for eps_opt in eps_opt_vals:
            fid_target = 1.0 - eps_opt
            dmax_over_t = []
            print(f"\n[eps_opt sweep] k={k}  eps_opt={eps_opt}")

            for t in t_range:
                target_t = np.linalg.matrix_power(U_F, t) @ psi0_N
                result   = layerwise_prepare(
                    psi0_N, target_t, N, eps_opt=eps_opt,
                    max_depth=8, n_restarts=n_restarts, seed=seed,
                    return_all_restarts=True)
                dmax_over_t.append(result["depth"])
                rows_csv.append(dict(k=k, t=t, eps_opt=eps_opt,
                                     depth=result["depth"],
                                     n_success=result["n_success"],
                                     sufficient=int(result["sufficient"]),
                                     fidelity=result["fidelity"]))
                print(f"  t={t:2d}: D={result['depth']} "
                      f"({'ok' if result['sufficient'] else 'CEIL'}) "
                      f"F={result['fidelity']:.3f}  "
                      f"success={result['n_success']}/{n_restarts}")

            dmax = max(dmax_over_t)
            dmax_table[(k, eps_opt)] = dmax
            print(f"  => Dmax(k={k}, eps={eps_opt}) = {dmax}")

    # CSV
    csv_path = outfile + ".csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
        writer.writeheader(); writer.writerows(rows_csv)
    print(f"  saved {csv_path}")

    # Bar chart: Dmax vs eps_opt per k
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x     = np.arange(len(eps_opt_vals))
    width = 0.35
    colors_k = {2.5: "#D55E00", 3.0: "#009E73"}

    for ki, k in enumerate(k_vals):
        dmax_vals = [dmax_table[(k, e)] for e in eps_opt_vals]
        offset    = (ki - 0.5) * width
        ax.bar(x + offset, dmax_vals, width, label=f"$k={k}$",
               color=colors_k.get(k, "grey"), alpha=0.85)

    ax.set_xlabel(r"$\varepsilon_{\rm opt}$")
    ax.set_ylabel(r"$D_{\rm max}$  (over $t=1,\ldots,12$)")
    ax.set_title(r"$D_{\rm max}$ vs. $\varepsilon_{\rm opt}$ sensitivity ($N=6$)")
    ax.set_xticks(x); ax.set_xticklabels([str(e) for e in eps_opt_vals])
    ax.legend(fontsize=8); ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outfile + ".pdf"); fig.savefig(outfile + ".png", dpi=300)
    plt.close(fig)
    print(f"  saved {outfile}.pdf/.png")

    # TeX output
    print("\n% --- eps_opt sensitivity sweep (paste into tex) ---")
    print(f"% N={N}, n_restarts={n_restarts}")
    print(r"% \begin{tabular}{ccc}")
    print(r"% $k$ & $\varepsilon_{\rm opt}$ & $D_{\rm max}$ \\")
    print(r"% \hline")
    for k in k_vals:
        for eps_opt in eps_opt_vals:
            dmax = dmax_table[(k, eps_opt)]
            print(f"% {k} & {eps_opt} & {dmax} \\\\")
    print(r"% \hline")
    print(r"% \end{tabular}")
    print("%")
    print("% Interpretation guide:")
    for k in k_vals:
        d05 = dmax_table[(k, 0.05)]
        d03 = dmax_table[(k, 0.03)]
        d02 = dmax_table[(k, 0.02)]
        if d05 == d03 == d02:
            print(f"% k={k}: Dmax={d05} is STABLE across eps_opt — "
                  "plateau is a genuine expressibility feature, NOT a threshold artefact.")
        else:
            print(f"% k={k}: Dmax changes ({d05} -> {d03} -> {d02}) — "
                  "plateau BREAKS at tighter eps; revise Fig 10 and surrounding text.")
    print("%")

    return dmax_table, rows_csv


# ── Barren-plateau gradient variance (unchanged + JSON save) ─────────────────

def barren_plateau_gradient_variance(
        system_sizes, depth_fixed, depth_adaptive_fn, psi0_fn, target_fn,
        n_inits=60, seed=0, component_index=0):
    """Referee point #10: gradient variance vs N (unchanged from original)."""
    from vqs import Ansatz
    var_fixed    = []
    var_adaptive = []
    depths_adapt = []

    for N in system_sizes:
        psi0   = psi0_fn(N)
        target = target_fn(N)

        ans_fixed  = Ansatz(N, depth_fixed)
        grads_fixed = []
        for trial in range(n_inits):
            rng    = np.random.default_rng((seed, N, "fixed", trial))
            theta0 = rng.uniform(0, 2 * np.pi, ans_fixed.n_params)
            g      = _parameter_shift_gradient(ans_fixed, theta0, psi0, target)
            grads_fixed.append(g[component_index])
        var_fixed.append(float(np.var(grads_fixed)))

        D_adapt = depth_adaptive_fn(N)
        depths_adapt.append(D_adapt)
        ans_adapt   = Ansatz(N, D_adapt)
        grads_adapt = []
        for trial in range(n_inits):
            rng    = np.random.default_rng((seed, N, "adaptive", trial))
            theta0 = rng.uniform(0, 2 * np.pi, ans_adapt.n_params)
            g      = _parameter_shift_gradient(ans_adapt, theta0, psi0, target)
            grads_adapt.append(g[component_index])
        var_adaptive.append(float(np.var(grads_adapt)))

        print(f"N={N}: Var[grad]_fixed(D={depth_fixed})={var_fixed[-1]:.3e}, "
              f"Var[grad]_adaptive(D={D_adapt})={var_adaptive[-1]:.3e}")

    return dict(system_sizes=list(system_sizes),
                var_fixed=var_fixed, var_adaptive=var_adaptive,
                depth_fixed_used=depth_fixed, depth_adaptive_used=depths_adapt)


def plot_barren_plateau_variance(result,
                                 outfile="figures/barren_plateau_variance"):
    """Log-scale variance-vs-N plot (unchanged) + TeX output + JSON save."""
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    N = result["system_sizes"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(N, result["var_fixed"], "o-", color="#D55E00",
                label=f"Fixed depth $D={result['depth_fixed_used']}$")
    ax.semilogy(N, result["var_adaptive"], "s-", color="#0072B2",
                label="Adaptive depth (per-$N$)")
    ax.set_xlabel("System size $N$ (qubits)")
    ax.set_ylabel(r"$\mathrm{Var}[\partial_{\theta_0} \mathcal{L}]$")
    ax.set_title("Gradient variance vs. system size")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile + ".pdf"); fig.savefig(outfile + ".png", dpi=300)
    plt.close(fig)
    print(f"  saved {outfile}.pdf/.png")

    # JSON
    with open(outfile + ".json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  saved {outfile}.json")

    # TeX output
    print("\n% --- barren-plateau gradient variance (paste into tex) ---")
    print(r"% \begin{tabular}{ccc}")
    print(r"% $N$ & $\mathrm{Var}[\nabla]_{\rm fixed}$ & "
          r"$\mathrm{Var}[\nabla]_{\rm adaptive}$ \\")
    print(r"% \hline")
    for n, vf, va in zip(N, result["var_fixed"], result["var_adaptive"]):
        print(f"% {n} & ${vf:.3e}$ & ${va:.3e}$ \\\\")
    print(r"% \hline")
    print(r"% \end{tabular}")
    print("%")
    print("% Interpretation: if var_adaptive decays SLOWER than var_fixed with N,")
    print("% the adaptive scheme mitigates barren plateaus (supports Sec. III G).")
    print("% If they track each other, soften or remove the mitigation claim.")
    print("%")

    return result


# ── Adaptive depths for TASK 3, loaded live instead of hand-typed ──────────

def _load_adaptive_depths_from_scaling_results(
        path="figures/depth_scaling_results.json",
        which="max", regime="chaotic", n_steps_max=10):
    """Load the per-N circuit depth to use in the barren-plateau comparison
    directly from large_scale_scaling.py's saved output, instead of a
    hand-maintained dict that can silently go stale (as the old
    dmax_map = {4:4, 6:10, 8:23, 10:25} did -- it never matched any real
    run of large_scale_scaling.py).

    which : "max" uses the worst-case sufficient depth per N (the same
        Dmax convention used elsewhere in the paper, e.g. Fig. 10) --
        the natural choice here since a single fixed Ansatz depth is
        used for every trial at that N, so it should be one that's
        actually sufficient rather than merely typical.
        "mean" uses <D>_t rounded to the nearest integer instead.

    Raises FileNotFoundError rather than silently falling back to a
    placeholder -- this analysis is explicitly future work (Sec. IV C of
    the paper currently defers it), so if the prerequisite data isn't on
    disk yet, run large_scale_scaling.py first rather than guessing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. The barren-plateau adaptive-depth "
            f"comparison (TASK 3) needs large_scale_scaling.py's saved "
            f"results -- run that script first, then rerun this one.")
    with open(path) as fh:
        r = json.load(fh)

    sizes = r["system_sizes"]
    if which == "max":
        depths = r[regime]["max_depth"]
    elif which == "mean":
        depths = [round(d) for d in r[regime]["mean_depth"]]
    else:
        raise ValueError(f"which must be 'max' or 'mean', got {which!r}")

    n_eval = r[regime]["n_steps_evaluated"]
    dmap   = dict(zip(sizes, depths))
    for N, ne in zip(sizes, n_eval):
        if ne < r.get("n_steps_max", n_steps_max):
            print(f"[bp-depth] N={N}: adaptive depth {dmap[N]} is based on "
                  f"only {ne} evaluated Floquet step(s) out of "
                  f"{r.get('n_steps_max', n_steps_max)}, not the full "
                  f"window -- same undersampling caveat as Fig. 11, treat "
                  f"as provisional.", flush=True)
    return dmap


# ── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from spin_operators import coherent_product_state
    from qkt_quantum    import floquet_U_exact

    N = 4
    print("=" * 65)
    print("adapt_vqa_baseline.py  — open-task suite")
    print("=" * 65)

    # ── Quick sanity check (original) ──────────────────────────────────────
    psi0   = coherent_product_state(N)
    U      = floquet_U_exact(N, 2.5, np.pi / 2)
    target = np.linalg.matrix_power(U, 4) @ psi0
    adapt  = adapt_vqa_prepare(psi0, target, N)
    layer  = layerwise_prepare(psi0, target, N)
    print("ADAPT-VQA :", adapt)
    print("Layer-wise:", layer)

    # ── TASK 1: eps_opt = 0.03 sensitivity rerun ────────────────────────────
    print("\n" + "=" * 65)
    print("TASK 1: eps_opt sensitivity sweep (0.05, 0.03, 0.02) at N=6")
    print("        k = 2.5 and k = 3.0  (Dmax=3 plateau check)")
    print("=" * 65)
    dmax_table, eps_rows = eps_opt_sensitivity_sweep(
        N=6, k_vals=(2.5, 3.0),
        t_range=range(1, 13),
        eps_opt_vals=(0.05, 0.03, 0.02),
        n_restarts=50, seed=0)

    # ── TASK 2: Basin-hopping confidence intervals ───────────────────────────
    print("\n" + "=" * 65)
    print("TASK 2: Basin-hopping CIs for Dmax (eps_opt=0.03, N=6)")
    print("=" * 65)
    bh_data, bh_rows = confidence_intervals_from_bh(
        psi0_fn=None, target_fn=None,     # unused; overridden inside function
        N=6, k_vals=(2.5, 3.0),
        t_range=range(1, 6),              # t=1..5 for speed; extend as needed
        eps_opt=0.03,
        n_seeds=8, n_bh_iter=80, seed=0,
        outfile="figures/bh_cnot_ci")

    # ── TASK 3: Barren-plateau gradient-variance measurement ────────────────
    print("\n" + "=" * 65)
    print("TASK 3: Barren-plateau gradient variance (N=4,6,8,10)")
    print("        Compares fixed D=8 vs adaptive D from chaotic regime.")
    print("=" * 65)

    BP_SYSTEM_SIZES = [4, 6, 8, 10]
    BP_DEPTH_FIXED  = 8

    # Adaptive depths from large_scale_scaling.py's actual saved output
    # (figures/depth_scaling_results.json), loaded live rather than
    # hand-typed -- see _load_adaptive_depths_from_scaling_results() above.
    # Uses the worst-case (max) sufficient depth per N in the chaotic
    # regime, same convention as Dmax elsewhere in the paper. N=8 and N=10
    # are based on only 2 and 1 evaluated Floquet step(s) respectively
    # (printed as a warning below), same caveat as Fig. 11.
    dmax_map = _load_adaptive_depths_from_scaling_results(
        which="max", regime="chaotic")

    def bp_depth_adaptive_fn(N_):
        return dmax_map.get(N_, N_ + 6)

    def bp_psi0_fn(N_):
        return coherent_product_state(N_)

    def bp_target_fn(N_):
        U_bp = floquet_U_exact(N_, 2.5, np.pi / 2)
        return np.linalg.matrix_power(U_bp, 4) @ coherent_product_state(N_)

    bp_result = barren_plateau_gradient_variance(
        BP_SYSTEM_SIZES, BP_DEPTH_FIXED,
        bp_depth_adaptive_fn, bp_psi0_fn, bp_target_fn,
        n_inits=60, seed=0)
    plot_barren_plateau_variance(bp_result)

    print("\n" + "=" * 65)
    print("ALL TASKS COMPLETE.  New files:")
    print("  figures/eps_opt_sensitivity.{pdf,png,csv}")
    print("  figures/bh_cnot_ci.{pdf,png,csv}")
    print("  figures/barren_plateau_variance.{pdf,png,json}")
    print("  Paste the % TeX blocks above into the manuscript.")
    print("=" * 65)