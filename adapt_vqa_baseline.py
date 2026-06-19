import numpy as np
from scipy.optimize import minimize
from functools import reduce
from spin_operators import embed, SX, SY, SZ, I2, normalize





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


def adapt_vqa_prepare(psi0, target, N, fid_target=0.99, grad_tol=1e-3,
                      max_ops=40, seed=0):
    rng = np.random.default_rng(seed)
    pool = build_pool(N)
    gens, thetas, labels = [], [], []
    cnot_count = 0
    opt_cycles = 0

    psi0 = normalize(psi0)
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
            grad = -2.0 * np.real(np.conjugate(ov) * d_amp)
            if abs(grad) > best_g:
                best_g, best_idx = abs(grad), idx

        if best_g < grad_tol:
            break

        lab, G, cc = pool[best_idx]
        gens.append(G); thetas.append(0.0); labels.append(lab)
        cnot_count += cc


        res = minimize(_infidelity, np.array(thetas),
                       args=(gens, psi0, target), method="BFGS",
                       options={"maxiter": 200})
        thetas = list(res.x)
        opt_cycles += int(res.nfev)

    psi = _state_from_ops(thetas, gens, psi0)
    fid = float(abs(np.vdot(target, psi)) ** 2)
    return dict(fidelity=fid, n_ops=len(gens), cnot_count=cnot_count,
                opt_cycles=opt_cycles, ops=labels)





def layerwise_prepare(psi0, target, N, eps_opt=0.05, max_depth=8,
                      n_restarts=50, seed=0, method="L-BFGS-B",
                      return_all_restarts=False):
    """Minimum-depth state preparation via the layer-wise ansatz, matching
    the direct-optimization Dmax procedure described in the manuscript
    (Sec. II.E.4): for each depth D, attempt n_restarts independent random
    initializations and declare D sufficient if ANY restart reaches
    infidelity below eps_opt.

    NOTE on referee point #8 (Dmax=3 plateau at k=2.5 and k=3.0): the
    previous version of this function used a single fixed-seed restart per
    depth (`rng = np.random.default_rng(seed)` re-created identically
    inside the D-loop), so it never actually tested whether a different
    initialization could break a depth's failure to converge. That is a
    likely confound for an apparent Dmax plateau: a single unlucky
    initialization at a given depth can masquerade as a genuine
    expressibility limit. This version restores genuine restart diversity
    (a different, deterministic sub-seed per restart) and exposes eps_opt
    directly so the threshold used here matches whatever eps_opt value the
    manuscript or a follow-up sensitivity check (e.g. eps_opt = 0.03 or
    0.02 at k=2.5 and k=3.0) actually requires.

    Parameters
    ----------
    eps_opt : float
        Target infidelity threshold (manuscript's varepsilon_opt). Default
        0.05 matches the value used for Fig. 10 in the manuscript. To test
        whether the k=2.5/k=3.0 Dmax=3 plateau survives a tighter
        threshold, call this with eps_opt=0.03 or eps_opt=0.02.
    n_restarts : int
        Number of independent random restarts per depth. Manuscript uses
        50; depth D is declared sufficient if ANY restart succeeds, and
        insufficient only if ALL n_restarts restarts fail.
    return_all_restarts : bool
        If True, also return the full list of per-restart infidelities at
        the depth where the loop terminates, so you can check (as the
        manuscript robustness check does) how many of the n_restarts
        independently reached the target -- a single lucky restart out of
        50 should be treated with more suspicion than, say, 10+ successes.
    """
    from vqs import Ansatz
    target = normalize(target)
    cnot_count = 0
    opt_cycles = 0
    fid_target = 1.0 - eps_opt

    for D in range(1, max_depth + 1):
        ans = Ansatz(N, D)

        def cost(th):
            psi = ans.state(th, psi0)
            return 1.0 - abs(np.vdot(target, psi)) ** 2

        best_fid = -np.inf
        n_success = 0
        restart_fids = []
        for r in range(n_restarts):
            # Each restart gets its own sub-seed, deterministically derived
            # from (seed, D, r), so results are reproducible but restarts
            # are no longer identical across the loop.
            rng = np.random.default_rng((seed, D, r))
            x0 = rng.uniform(0, 2 * np.pi, ans.n_params)
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

    # Unreachable given the D == max_depth fallback above, kept for clarity.
    return dict(fidelity=float(best_fid), depth=max_depth,
               cnot_count=max_depth * 2 * N, opt_cycles=opt_cycles,
               n_restarts=n_restarts, n_success=n_success, eps_opt=eps_opt,
               sufficient=False)


def _parameter_shift_gradient(ans, theta, psi0, target, shift=np.pi / 2):
    """Parameter-shift-rule gradient of the infidelity cost
    1 - |<target|psi(theta)>|^2 with respect to every entry of theta.

    Uses only ans.state(theta, psi0), which is already the interface this
    module relies on elsewhere (see layerwise_prepare), so it makes no
    additional assumptions about the Ansatz class beyond what is already
    exercised in this file. This avoids depending on an autodiff capability
    of Ansatz that may or may not exist, at the cost of 2 circuit
    evaluations per parameter (standard parameter-shift cost).
    """
    def infidelity(th):
        psi = ans.state(th, psi0)
        return 1.0 - abs(np.vdot(target, psi)) ** 2

    n_params = len(theta)
    grad = np.zeros(n_params)
    for i in range(n_params):
        theta_plus = theta.copy()
        theta_plus[i] += shift
        theta_minus = theta.copy()
        theta_minus[i] -= shift
        grad[i] = 0.5 * (infidelity(theta_plus) - infidelity(theta_minus))
    return grad


def barren_plateau_gradient_variance(
    system_sizes, depth_fixed, depth_adaptive_fn, psi0_fn, target_fn,
    n_inits=60, seed=0, component_index=0):
    """Referee point #10 (barren plateau mitigation claim unsupported).

    For each system size N, draws n_inits independent random parameter
    initializations, computes the parameter-shift gradient of the
    infidelity cost at each, and reports the variance of one gradient
    component (component_index, default the first parameter) across those
    initializations -- the standard barren-plateau diagnostic (McClean et
    al. 2018): an exponential decay of this variance with N is the
    signature of a barren plateau. This is done separately for a FIXED
    circuit depth (depth_fixed, representing what the manuscript calls a
    circuit "initialized deep enough to capture chaotic dynamics from the
    outset") and for the ADAPTIVE algorithm's typical depth at each N
    (depth_adaptive_fn(N), representing the depth the adaptive scheme
    would actually be sitting at, e.g. D=1 or whatever the residual
    trigger settles on for that N). Comparing the two variance-vs-N curves
    is what actually substantiates (or refutes) the claim in Sec. III G
    that adaptive depth expansion mitigates barren plateaus, rather than
    asserting it without evidence.

    Parameters
    ----------
    system_sizes : sequence of int
        Qubit counts N to sweep.
    depth_fixed : int
        A single, fixed circuit depth applied at every N, representing the
        "circuit of a fixed depth initialized deep enough to capture
        chaotic dynamics from the outset" comparison case.
    depth_adaptive_fn : callable
        Function N -> int giving the depth the adaptive algorithm should be
        evaluated at for that system size (e.g. D=1, or whatever depth the
        residual trigger settles at for the regime being studied).
    psi0_fn, target_fn : callable
        Functions N -> state vector giving the initial state and target
        state for that system size (e.g. wrap coherent_product_state and a
        Floquet-evolved target from qkt_quantum for consistency with the
        rest of the pipeline).
    n_inits : int
        Number of random initializations per (N, depth) pair. The
        manuscript-style robustness checks elsewhere in this codebase use
        50; 60 is used here as a comparable default within reach of a
        single overnight run.
    component_index : int
        Which entry of the gradient vector to use for the variance
        statistic. McClean et al. use a single fixed gradient component
        (not the full vector norm) because the variance of any single
        component already exhibits the barren-plateau scaling, and using a
        single component avoids artificially inflating the variance
        estimate by aggregating over a parameter count that itself grows
        with N and depth.

    Returns
    -------
    dict with keys 'system_sizes', 'var_fixed', 'var_adaptive',
    'depth_fixed_used', 'depth_adaptive_used' for direct use in a
    log-variance-vs-N plot.
    """
    from vqs import Ansatz

    var_fixed    = []
    var_adaptive = []
    depths_adapt = []

    for N in system_sizes:
        psi0   = psi0_fn(N)
        target = target_fn(N)

        # --- fixed, large depth case ---
        ans_fixed = Ansatz(N, depth_fixed)
        grads_fixed = []
        for trial in range(n_inits):
            rng = np.random.default_rng((seed, N, "fixed", trial))
            theta0 = rng.uniform(0, 2 * np.pi, ans_fixed.n_params)
            g = _parameter_shift_gradient(ans_fixed, theta0, psi0, target)
            grads_fixed.append(g[component_index])
        var_fixed.append(float(np.var(grads_fixed)))

        # --- adaptive-algorithm-typical depth case ---
        D_adapt = depth_adaptive_fn(N)
        depths_adapt.append(D_adapt)
        ans_adapt = Ansatz(N, D_adapt)
        grads_adapt = []
        for trial in range(n_inits):
            rng = np.random.default_rng((seed, N, "adaptive", trial))
            theta0 = rng.uniform(0, 2 * np.pi, ans_adapt.n_params)
            g = _parameter_shift_gradient(ans_adapt, theta0, psi0, target)
            grads_adapt.append(g[component_index])
        var_adaptive.append(float(np.var(grads_adapt)))

        print(f"N={N}: Var[grad]_fixed(D={depth_fixed})={var_fixed[-1]:.3e}, "
              f"Var[grad]_adaptive(D={D_adapt})={var_adaptive[-1]:.3e}")

    return dict(system_sizes=list(system_sizes),
              var_fixed=var_fixed, var_adaptive=var_adaptive,
              depth_fixed_used=depth_fixed, depth_adaptive_used=depths_adapt)


def plot_barren_plateau_variance(result, outfile="figures/barren_plateau_variance"):
    """Log-scale variance-vs-N plot, the standard way barren-plateau
    results are presented (McClean et al. 2018, Fig. 2-style): an
    exponential decay of gradient variance with N appears as a straight
    line on this semi-log plot, while a mitigated (non-exponential) decay
    visibly bends away from that line.
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)

    N = result["system_sizes"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(N, result["var_fixed"], "o-", color="#D55E00",
              label=f"Fixed depth $D={result['depth_fixed_used']}$")
    ax.semilogy(N, result["var_adaptive"], "s-", color="#0072B2",
              label="Adaptive depth (per-$N$, see legend depths)")
    ax.set_xlabel("System size $N$ (qubits)")
    ax.set_ylabel(r"$\mathrm{Var}[\partial_{\theta_0} \mathcal{L}]$")
    ax.set_title("Gradient variance vs. system size")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile + ".pdf")
    fig.savefig(outfile + ".png", dpi=300)
    plt.close(fig)
    print(f"Saved {outfile}.pdf / .png")
    print(
        "\nInterpretation guide: if var_fixed decays roughly exponentially "
        "with N (a straight downward line on this semi-log plot) while "
        "var_adaptive decays much more slowly or stays roughly flat, that "
        "is genuine evidence for the manuscript's barren-plateau "
        "mitigation claim (Sec. III G). If the two curves track each "
        "other closely, the claim is NOT supported by this data and the "
        "manuscript text should be softened or the claim removed, per "
        "referee point #10.")


if __name__ == "__main__":
    from spin_operators import coherent_product_state
    from qkt_quantum import floquet_U_exact

    N = 4
    psi0 = coherent_product_state(N)
    U = floquet_U_exact(N, 2.5, np.pi / 2)
    target = np.linalg.matrix_power(U, 4) @ psi0 

    adapt = adapt_vqa_prepare(psi0, target, N)
    layer = layerwise_prepare(psi0, target, N)
    print("ADAPT-VQA :", adapt)
    print("Layer-wise:", layer)

    print("\n" + "=" * 60)
    print("Referee point #8: Dmax=3 plateau at k=2.5 and k=3.0")
    print("=" * 60)
    print("Re-running the direct-optimization Dmax procedure at the\n"
          "manuscript's N=6, with eps_opt swept tighter than the default\n"
          "0.05, to test whether the plateau survives higher resolution.\n"
          "NOTE: this uses layerwise_prepare's max-over-t convention to\n"
          "match Fig. 10's definition of Dmax (smallest D achieving the\n"
          "target infidelity at ANY t in {1,...,12}), so it loops over t\n"
          "and takes the max depth needed across that range.")

    N_PLATEAU = 6
    T_RANGE = range(1, 13)
    K_PLATEAU = [2.5, 3.0]
    EPS_OPT_VALUES = [0.05, 0.03, 0.02]

    psi0_p = coherent_product_state(N_PLATEAU)

    for k in K_PLATEAU:
        U_p = floquet_U_exact(N_PLATEAU, k, np.pi / 2)
        print(f"\n--- k={k} ---")
        for eps_opt in EPS_OPT_VALUES:
            dmax_over_t = []
            for t in T_RANGE:
                target_t = np.linalg.matrix_power(U_p, t) @ psi0_p
                result = layerwise_prepare(
                    psi0_p, target_t, N_PLATEAU, eps_opt=eps_opt,
                    max_depth=8, n_restarts=50, seed=0,
                    return_all_restarts=True)
                dmax_over_t.append(result["depth"])
                n_succ = result["n_success"]
                print(f"  t={t:2d}: eps_opt={eps_opt:.2f} -> "
                      f"D={result['depth']} "
                      f"(succeeded in {n_succ}/50 restarts, "
                      f"sufficient={result['sufficient']})")
            dmax_k = max(dmax_over_t)
            print(f"  => Dmax(k={k}, eps_opt={eps_opt:.2f}) = {dmax_k}")

    print(
        "\nInterpretation guide: if Dmax(k=2.5) and Dmax(k=3.0) remain\n"
        "equal at eps_opt=0.03 and 0.02 (not just 0.05), the plateau is\n"
        "likely a genuine feature of the expressibility landscape rather\n"
        "than an artifact of a too-coarse infidelity threshold, and the\n"
        "manuscript should say so explicitly (referee point #8). If the\n"
        "plateau breaks at a tighter eps_opt, report the new Dmax values\n"
        "and revise Fig. 10 / the surrounding text accordingly -- do not\n"
        "keep both the old plateau claim and the new numbers in the\n"
        "manuscript at the same time.")

    print("\n" + "=" * 60)
    print("Referee point #10: barren plateau mitigation claim")
    print("=" * 60)
    print("Computing parameter-shift gradient variance vs. system size N,\n"
          "for a fixed large depth vs. the depth the adaptive algorithm\n"
          "would actually use at each N, in the chaotic regime (k=2.5).\n"
          "This is the standard McClean et al. 2018 diagnostic and is what\n"
          "actually substantiates (or refutes) the Sec. III G claim that\n"
          "adaptive depth expansion mitigates barren plateaus.")

    BP_SYSTEM_SIZES = [4, 6, 8, 10]
    BP_DEPTH_FIXED  = 8   # matches the manuscript's D_ceil for the chaotic
                          # regime; represents "initialized deep enough to
                          # capture chaotic dynamics from the outset"

    def bp_depth_adaptive_fn(N):
        # These are the converged adaptive depths in the chaotic regime
        # that we just obtained from large_scale_scaling.py
        dmax_map = {4: 4, 6: 10, 8: 23, 10: 25}
        return dmax_map[N]

    def bp_psi0_fn(N):
        return coherent_product_state(N)

    def bp_target_fn(N):
        U_bp = floquet_U_exact(N, 2.5, np.pi / 2)
        return np.linalg.matrix_power(U_bp, 4) @ coherent_product_state(N)

    bp_result = barren_plateau_gradient_variance(
        BP_SYSTEM_SIZES, BP_DEPTH_FIXED, bp_depth_adaptive_fn,
        bp_psi0_fn, bp_target_fn, n_inits=60, seed=0)
    plot_barren_plateau_variance(bp_result)
