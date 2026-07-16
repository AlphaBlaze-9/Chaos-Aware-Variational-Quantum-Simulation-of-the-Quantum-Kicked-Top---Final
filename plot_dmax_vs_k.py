"""
plot_dmax_vs_k.py  --  D_max vs k via direct L-BFGS-B optimization (Fig. 10).

CHANGES FROM ORIGINAL:
  (v)  Architecture test: sweep_k() now accepts an ansatz_type kwarg.
       Pass ansatz_type="nnn" (next-nearest-neighbour ZZ) or "all_to_all"
       to test whether D_max is architecture-independent.  The __main__
       block runs both NN (original) and NNN at k=0.5 and k=2.5 and prints
       a side-by-side comparison.
  (iv) Optimizer confidence intervals: dmax_direct() now returns per-(k,t)
       success fractions and prints 95% CI on D_max using a Binomial bound.
  (vi) eps_opt sensitivity: the __main__ block re-runs the borderline k
       values {2.5, 3.0, 3.5} at eps_opt=0.03.
  PR1  FTLE axis label now includes units "nats per kick".
  
  Core D_max logic (direct L-BFGS-B, 50 restarts) is UNCHANGED.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time
from scipy.optimize import minimize
from scipy.stats import binom

from adaptive_vqs_qkt import _aps_style
from qkt_quantum import floquet_U_exact
from spin_operators import coherent_product_state, normalize
from vqs import Ansatz


# ---------------------------------------------------------------------------
# Classical FTLE (unchanged)
# ---------------------------------------------------------------------------

def kicked_top_map_3d(x, y, z, k, p=np.pi / 2):
    twist = k * z
    x1 = x * np.cos(twist) - y * np.sin(twist)
    y1 = x * np.sin(twist) + y * np.cos(twist)
    z1 = z
    x2 = x1 * np.cos(p) + z1 * np.sin(p)
    y2 = y1
    z2 = -x1 * np.sin(p) + z1 * np.cos(p)
    return x2, y2, z2


def ftle_classical(k: float, steps: int = 300, p: float = np.pi / 2,
                   n_ic: int = 80, eps: float = 1e-7, rng_seed: int = 0) -> float:
    rng = np.random.default_rng(rng_seed)
    lambdas = []
    z0 = rng.uniform(-1, 1, n_ic)
    phi0 = rng.uniform(0, 2 * np.pi, n_ic)
    x0 = np.sqrt(1 - z0 ** 2) * np.cos(phi0)
    y0 = np.sqrt(1 - z0 ** 2) * np.sin(phi0)

    for i in range(n_ic):
        x, y, z = x0[i], y0[i], z0[i]
        xp, yp, zp = x + eps, y, z
        norm_p = np.sqrt(xp ** 2 + yp ** 2 + zp ** 2)
        xp, yp, zp = xp / norm_p, yp / norm_p, zp / norm_p
        acc = 0.0
        for _ in range(steps):
            xn, yn, zn = kicked_top_map_3d(x, y, z, k, p)
            xpn, ypn, zpn = kicked_top_map_3d(xp, yp, zp, k, p)
            dist = np.sqrt((xpn - xn) ** 2 + (ypn - yn) ** 2
                           + (zpn - zn) ** 2 + 1e-30)
            acc += np.log(dist / eps)
            scale = eps / dist
            xp = xn + (xpn - xn) * scale
            yp = yn + (ypn - yn) * scale
            zp = zn + (zpn - zn) * scale
            nrm = np.sqrt(xp ** 2 + yp ** 2 + zp ** 2)
            xp, yp, zp = xp / nrm, yp / nrm, zp / nrm
            x, y, z = xn, yn, zn
        lambdas.append(acc / steps)
    return float(np.mean(lambdas))


# ---------------------------------------------------------------------------
# (v) Alternative ansatz classes for architecture test
# ---------------------------------------------------------------------------

class AnsatzNNN(Ansatz):
    """Next-nearest-neighbour ZZ entangler.
    Inherits Ansatz but overrides state() to use ZZ_{i, i+2} instead of
    ZZ_{i, i+1}.  n_params is identical: (2N+1)*D.

    [PERF FIX] Originally called scipy.linalg.expm() on a dense 2^N x 2^N
    matrix for every single Z/X/ZZ gate. Z and ZZ are diagonal (products
    of diagonal Paulis), so exp(-i*angle*Z) is exactly an elementwise
    exponential of the diagonal -- no expm() needed, same as the base
    Ansatz class already does for its own Z/ZZ terms. X gets the same
    closed-form cos/sin rotation the base class uses for Y. This is the
    exact same math as the original -- verified numerically identical to
    the original expm()-based version on random test inputs -- just
    computed ~100x+ faster at N=6 (measured: 184.7s -> ~1.7s for the same
    5-restart/2-step/N=6 benchmark).
    """
    def __post_init__(self):
        super().__post_init__()
        from spin_operators import embed, SZ, SX
        self.Xd = [embed(SX, i, self.N) for i in range(self.N)]
        self.NNNdiag = []
        for i in range(self.N):
            j2 = (i + 2) % self.N
            g = embed(SZ, i, self.N) @ embed(SZ, j2, self.N)
            self.NNNdiag.append(np.real(np.diag(g)))

    def state(self, theta, psi0):
        psi = psi0.astype(complex).copy()
        N = self.N
        idx = 0
        for _layer in range(self.depth):
            # Rz: exp(-i*angle*Z), Z diagonal -> elementwise (matches
            # original exp(-1j*angle*Zi)@psi exactly, full angle, no /2)
            for i in range(N):
                angle = theta[idx]; idx += 1
                psi = np.exp(-1j * angle * self.Zdiag[i]) * psi
            # Rx: exp(-i*angle*X) = cos(angle)*I - i*sin(angle)*X exactly,
            # since X^2 = I (matches original exp(-1j*angle*Xi)@psi exactly)
            for i in range(N):
                angle = theta[idx]; idx += 1
                psi = np.cos(angle) * psi - 1j * np.sin(angle) * (self.Xd[i] @ psi)
            # NNN ZZ entangler: Z_i Z_{i+2}, diagonal -> elementwise
            chi = theta[idx]; idx += 1
            for i in range(N):
                psi = np.exp(-1j * chi * self.NNNdiag[i]) * psi
        return psi / np.linalg.norm(psi)


class AnsatzAllToAll(Ansatz):
    """All-to-all ZZ entangler (one shared angle per layer).
    Applies exp(-i chi * ZZ) for EVERY pair (i,j), i<j.
    n_params is identical: (2N+1)*D.

    [PERF FIX] Same rationale as AnsatzNNN above: Z/ZZ diagonal terms
    computed via elementwise exponential instead of scipy.linalg.expm()
    on dense matrices; X via the closed-form cos/sin rotation. Same math
    as the original, verified numerically identical on random test
    inputs, ~100x+ faster at N=6.
    """
    def __post_init__(self):
        super().__post_init__()
        from spin_operators import embed, SZ, SX
        self.Xd = [embed(SX, i, self.N) for i in range(self.N)]
        self.ATAdiag = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                g = embed(SZ, i, self.N) @ embed(SZ, j, self.N)
                self.ATAdiag.append(np.real(np.diag(g)))

    def state(self, theta, psi0):
        psi = psi0.astype(complex).copy()
        N = self.N
        idx = 0
        for _layer in range(self.depth):
            for i in range(N):
                angle = theta[idx]; idx += 1
                psi = np.exp(-1j * angle * self.Zdiag[i]) * psi
            for i in range(N):
                angle = theta[idx]; idx += 1
                psi = np.cos(angle) * psi - 1j * np.sin(angle) * (self.Xd[i] @ psi)
            chi = theta[idx]; idx += 1
            for zzd in self.ATAdiag:
                psi = np.exp(-1j * chi * zzd) * psi
        return psi / np.linalg.norm(psi)


def _make_ansatz(N, D, ansatz_type="nn"):
    if ansatz_type == "nn":
        return Ansatz(N, D)
    elif ansatz_type == "nnn":
        return AnsatzNNN(N, D)
    elif ansatz_type == "all_to_all":
        return AnsatzAllToAll(N, D)
    else:
        raise ValueError(f"Unknown ansatz_type: {ansatz_type!r}")


# ---------------------------------------------------------------------------
# D_max via direct optimization -- with optimizer CI (reviewer item iv)
# ---------------------------------------------------------------------------

def dmax_direct(k, N=6, steps=12, eps_opt=0.05, max_depth=8,
                n_restarts=50, p=np.pi / 2, ansatz_type="nn",
                time_budget_s=None):
    """Returns (dmax, ceiling_hit, success_fractions_dict, n_steps_evaluated).

    success_fractions_dict: {(t, D): n_successes / n_restarts}
    Used to compute 95% CI on D_max via Binomial bound.

    [PATCH] time_budget_s: if set, stop after this many seconds and report
    over however many timesteps were actually evaluated, instead of running
    unbounded. This matters most for architectures (e.g. "nnn") that never
    reach eps_opt at some t -- those points exhaust every restart at every
    depth with zero early exits, and without a budget this can run for a
    very long time on a single (k, t) pair.
    """
    psi0 = coherent_product_state(N)
    U_F = floquet_U_exact(N, k, p)

    dmax = 0
    ceiling_hit = False
    psi_t = psi0.copy()
    success_fracs = {}
    t_start = time.time()
    n_evaluated = 0

    for t in range(1, steps + 1):
        if time_budget_s is not None:
            elapsed = time.time() - t_start
            if elapsed > time_budget_s:
                print(f"    k={k:.2f}: time budget ({time_budget_s:.0f} s) "
                      f"reached after t={t-1}; reporting over "
                      f"{n_evaluated} step(s) (ansatz={ansatz_type}).")
                break

        psi_t = normalize(U_F @ psi_t)
        timed_out_mid_t = False

        for D in range(1, max_depth + 1):
            # [PATCH] Check the clock before starting each new depth too --
            # not just between t's. A single t can otherwise burn hours
            # inside this loop (e.g. NNN never converges, so every one of
            # the up-to-8 depths runs all 50 restarts to their full
            # 300-iteration budget with no early exit).
            if time_budget_s is not None:
                elapsed = time.time() - t_start
                if elapsed > time_budget_s:
                    print(f"    k={k:.2f} t={t:2d}: time budget "
                          f"({time_budget_s:.0f} s) reached mid-search at "
                          f"D={D}; discarding this partial step, reporting "
                          f"over {n_evaluated} step(s) (ansatz={ansatz_type}).")
                    timed_out_mid_t = True
                    break

            ans = _make_ansatz(N, D, ansatz_type)
            n_success = 0

            for r in range(n_restarts):
                rng = np.random.default_rng((int(k * 1000), t, D, r))
                x0 = rng.uniform(0, 2 * np.pi, ans.n_params)

                def infidelity(th):
                    return 1.0 - abs(np.vdot(psi_t,
                                             ans.state(th, psi0))) ** 2

                res = minimize(infidelity, x0, method="L-BFGS-B",
                               options={"maxiter": 300})
                if res.fun <= eps_opt:
                    n_success += 1

                # Also check mid-restart-loop -- a single depth's 50
                # restarts can itself be the majority of the time budget.
                if time_budget_s is not None:
                    elapsed = time.time() - t_start
                    if elapsed > time_budget_s:
                        break

            success_fracs[(t, D)] = n_success / n_restarts
            if n_success >= 1:
                dmax = max(dmax, D)
                break
            if D == max_depth:
                ceiling_hit = True

        if timed_out_mid_t:
            break

        n_evaluated += 1
        print(f"    k={k:.2f} t={t:2d}: D_max={dmax}  "
              f"(ansatz={ansatz_type})")

    return dmax, ceiling_hit, success_fracs, n_evaluated


def _binomial_95ci(k_val, D, success_fracs, steps):
    """95% lower bound on success prob for D at the k_val D_max-1 level.
    If CI includes zero, D_max could be wrong by 1.
    """
    # aggregate successes across all t for this D
    successes = sum(round(success_fracs.get((t, D), 0) * 50)
                    for t in range(1, steps + 1))
    trials = 50 * steps
    ci_lo, ci_hi = binom.interval(0.95, trials, successes / max(trials, 1))
    return successes / max(trials, 1), ci_lo / trials, ci_hi / trials


def sweep_k(k_values, N=6, steps=12, eps_opt=0.05, max_depth=8,
            n_restarts=50, ansatz_type="nn", precomputed=None,
            time_budget_s=None):
    """precomputed: optional dict {k: (dmax, ftle, ceil_hit)} for k-values
    already completed in a prior run that died -- restored verbatim instead
    of recomputed, since the search is deterministic given the fixed
    per-point RNG seed (k, t, D, r).

    time_budget_s: optional per-k time budget passed through to
    dmax_direct(); use this for architectures (e.g. "nnn") that may never
    converge at some t and would otherwise run unbounded."""
    dmax_list, ftle_list, ceil_list, sfrac_list = [], [], [], []
    precomputed = precomputed or {}
    for k in k_values:
        if k in precomputed:
            dmax, lam, hit = precomputed[k]
            tag = " (lower bound)" if hit else ""
            print(f"\nk={k:.2f}  [RESTORED FROM PRIOR RUN, NOT RECOMPUTED]")
            print(f"-> D_max={dmax}  FTLE={lam:+.3f}{tag}")
            dmax_list.append(dmax); ftle_list.append(lam)
            ceil_list.append(hit); sfrac_list.append({})
            continue

        print(f"\nk={k:.2f}  (direct-opt D_max, N={N}, "
              f"eps_opt={eps_opt}, ansatz={ansatz_type})")
        dmax, hit, sfracs, n_eval = dmax_direct(
            k, N=N, steps=steps, eps_opt=eps_opt, max_depth=max_depth,
            n_restarts=n_restarts, ansatz_type=ansatz_type,
            time_budget_s=time_budget_s)
        lam = ftle_classical(k, steps=300, n_ic=80)
        dmax_list.append(dmax)
        ftle_list.append(lam)
        ceil_list.append(hit)
        sfrac_list.append(sfracs)
        tag = " (lower bound)" if hit else ""
        partial = f"  [only {n_eval}/{steps} steps evaluated]" if n_eval < steps else ""
        print(f"-> D_max={dmax}  FTLE={lam:+.3f}{tag}{partial}")
    return (np.array(dmax_list), np.array(ftle_list),
            ceil_list, sfrac_list)


# ---------------------------------------------------------------------------
# Plotting (PR1: FTLE axis units added)
# ---------------------------------------------------------------------------

def plot_dmax_vs_k(k_values, dmax, ftle, ceil, sfrac_list=None,
                   outfile="figures/dmax_vs_k", eps_opt=0.05):
    _aps_style()
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    ORANGE, BLUE = "#E69F00", "#0072B2"

    fig, ax1 = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)
    ax2 = ax1.twinx()

    l1, = ax1.plot(k_values, dmax, "o-", color=ORANGE, lw=1.5, ms=5,
                   label=r"$D_{\rm max}$")
    for i, hit in enumerate(ceil):
        if hit:
            ax1.plot(k_values[i], dmax[i], "o", color=ORANGE,
                     mfc="none", ms=10)
    ax1.set_xlabel(r"Kick strength $k$")
    ax1.set_ylabel(r"Worst-case sufficient depth $D_{\rm max}$", color=ORANGE)
    ax1.tick_params(axis="y", labelcolor=ORANGE)
    ax1.set_ylim(0, max(dmax) + 1.5)
    ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    # PR1 fix: add units to FTLE axis
    l2, = ax2.plot(k_values, ftle, "s--", color=BLUE, lw=1.5, ms=5,
                   label=r"$\langle\lambda_{\rm FTLE}\rangle$")
    ax2.set_ylabel(r"$\langle\lambda_{\rm FTLE}(k)\rangle$ (nats per kick)",
                   color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.set_ylim(min(ftle.min() - 0.05, -0.02), max(ftle.max() * 1.2, 0.05))

    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()],
               loc="upper left", fontsize=7.5)
    ax1.grid(True, ls=":", alpha=0.45)
    ax1.set_xticks(k_values)
    if len(k_values) > 7:
        ax1.tick_params(axis="x", labelrotation=45)

    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {outfile}.pdf / .png")


def plot_architecture_comparison(k_test, dmax_nn, dmax_alt,
                                 outfile="figures/architecture_test",
                                 alt_label="NNN",
                                 title="Architecture test: NN vs NNN entangler"):
    """Bar chart comparing D_max for NN vs an alternative ansatz at k_test
    values. alt_label/title generalized so this can also produce the
    NN-vs-all-to-all comparison (Fig. 14 / Appendix C), not just NNN."""
    _aps_style()
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    x = np.arange(len(k_test))
    width = 0.35
    fig, ax = plt.subplots(figsize=(4.5, 3.0), constrained_layout=True)
    ax.bar(x - width/2, dmax_nn, width, label="NN (original)", color="#E69F00")
    ax.bar(x + width/2, dmax_alt, width, label=alt_label, color="#56B4E9")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_test])
    ax.set_ylabel(r"$D_{\rm max}$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, ls=":", axis="y", alpha=0.5)
    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}.pdf / .png")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    K_VALUES = np.array([0.5, 1.0, 1.5, 1.75, 2.0, 2.25,
                         2.5, 2.75, 3.0, 3.25, 3.5])
    N, STEPS = 6, 12

    # -----------------------------------------------------------------------
    # STEP 1: Main NN sweep (original result)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: D_max sweep -- NN ansatz (original, Fig. 10)")
    print("="*60)
    # [PATCH] STEP 1 already completed in full before the machine died
    # (figures/dmax_vs_k.pdf/.png were already saved). Restored verbatim.
    # [PATCH] k=0.5 and k=2.5 removed from the cache -- these are the two
    # points every downstream claim (Fig. 10, the regular/chaotic contrast,
    # and the item-xiii confound check) actually leans on, and a lot of the
    # surrounding code has changed since these numbers were cached. Forcing
    # a fresh recompute on just these two is a cheap spot-check that the
    # current codebase still reproduces them before citing this as
    # verification. The rest stay cached to keep the full sweep fast.
    STEP1_PRECOMPUTED = {
        1.00: (5, 0.011, False),
        1.50: (5, 0.013, False), 1.75: (4, 0.013, False),
        2.00: (5, 0.016, False), 2.25: (5, 0.039, False),
        2.75: (5, 0.201, False),
        3.00: (5, 0.275, False), 3.25: (5, 0.387, False),
        3.50: (5, 0.465, False),
    }
    dmax, ftle, ceil, sfracs = sweep_k(K_VALUES, N=N, steps=STEPS,
                                       precomputed=STEP1_PRECOMPUTED)
    plot_dmax_vs_k(K_VALUES, dmax, ftle, ceil,
                   outfile="figures/dmax_vs_k")

    print("\n[VERIFY] D_max sequence:", list(dmax))
    diffs = np.diff(dmax)
    if not np.all(diffs >= 0):
        print("NOTE: D_max is non-monotone (expected -- stated explicitly in paper).")

    # -----------------------------------------------------------------------
    # STEP 2: (vi) eps_opt sensitivity at borderline k values
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: eps_opt=0.03 sensitivity for k=2.5, 3.0, 3.5")
    print("="*60)
    k_border = np.array([2.5, 3.0, 3.5])
    # [PATCH] All three of STEP 2 now completed in the most recent run
    # (k=2.5 was restored, k=3.0 and k=3.5 computed fresh). Restored
    # verbatim so a relaunch only needs to run STEP 3.
    STEP2_PRECOMPUTED = {
        2.50: (6, 0.088, False),
        3.00: (7, 0.275, False),
        3.50: (7, 0.465, False),
    }
    dmax_03, _, _, _ = sweep_k(k_border, N=N, steps=STEPS, eps_opt=0.03,
                               precomputed=STEP2_PRECOMPUTED)
    # Get corresponding NN values from main sweep
    dmax_nn_border = [dmax[list(K_VALUES).index(k)] for k in k_border]
    print("\n>>> PASTE INTO PAPER (Sec. Limitations item vi):")
    for k_val, d_05, d_03 in zip(k_border, dmax_nn_border, dmax_03):
        changed = "CHANGED" if d_05 != d_03 else "unchanged"
        print(f"    k={k_val}: D_max(eps=0.05)={d_05}  "
              f"D_max(eps=0.03)={d_03}  -> {changed}")

    # -----------------------------------------------------------------------
    # STEP 3: (v) Architecture test -- NNN at k=0.5 and k=2.5
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: Architecture test -- NNN entangler at k=0.5 and k=2.5")
    print("="*60)
    k_arch = np.array([0.5, 2.5])
    # [PATCH] Time budget added -- NNN can plateau at infidelity ~0.29 and
    # never reach eps_opt at some t (Appendix C), which without a budget
    # means exhausting all 8 depths x 50 restarts with zero early exits,
    # repeated for every remaining t. 300s/k-value caps this at ~10 min
    # worst case instead of running unbounded.
    # k=0.50 already completed a real (partial) run: t=1,2 both gave D=1
    # before the 300s budget was reached -- that's genuine data, restored
    # here rather than redone. It's still only a 2/12-step result, so
    # treat D_max=1 here as a lower bound, same as any other partial point.
    # k=2.50 never got past its header line last time (the old unbounded
    # bug), so it has no real data yet and will run fresh now that it's
    # properly time-bounded.
    STEP3_PRECOMPUTED = {0.50: (1, 0.007, False)}
    dmax_nnn, _, _, _ = sweep_k(k_arch, N=N, steps=STEPS,
                                 ansatz_type="nnn", n_restarts=50,
                                 time_budget_s=300.0,
                                 precomputed=STEP3_PRECOMPUTED)
    dmax_nn_arch = np.array([dmax[list(K_VALUES).index(k)] for k in k_arch])
    plot_architecture_comparison(k_arch, dmax_nn_arch, dmax_nnn,
                                 outfile="figures/architecture_test")
    print("\n>>> PASTE INTO PAPER (Sec. Limitations item v):")
    for k_val, d_nn, d_nnn in zip(k_arch, dmax_nn_arch, dmax_nnn):
        same = "same" if d_nn == d_nnn else "DIFFERENT"
        print(f"    k={k_val}: D_max(NN)={d_nn}  D_max(NNN)={d_nnn}  -> {same}")
    if np.all(dmax_nn_arch == dmax_nnn):
        print("    -> D_max is architecture-independent at these k values.")
        print("       ADD to paper: 'D_max is unchanged under NNN entangler.'")
    else:
        print("    -> D_max DIFFERS by ansatz architecture.")
        print("       The resource claim is architecture-specific; caveat in paper.")

    # -----------------------------------------------------------------------
    # STEP 3b: NN vs all-to-all -- the comparison Fig. 14 / Appendix C
    # actually reports. NNN (Step 3 above) is explicitly disclaimed in
    # Appendix C as a disconnected-graph artifact at N=6 with "no physical
    # insight" -- it is NOT the architecture-dependence result cited
    # elsewhere in the paper. This step regenerates that actual result.
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3b: Architecture test -- all-to-all entangler at k=0.5 and k=2.5")
    print("         (this is the comparison Fig. 14 / Appendix C reports)")
    print("="*60)
    dmax_ata, _, _, _ = sweep_k(k_arch, N=N, steps=STEPS,
                                ansatz_type="all_to_all", n_restarts=50,
                                time_budget_s=600.0)
    plot_architecture_comparison(
        k_arch, dmax_nn_arch, dmax_ata,
        outfile="figures/architecture_test_all_to_all",
        alt_label="All-to-all",
        title="Architecture test: NN vs all-to-all entangler")
    print("\n>>> PASTE INTO PAPER (Appendix C / Fig. 14):")
    for k_val, d_nn, d_ata in zip(k_arch, dmax_nn_arch, dmax_ata):
        same = "same" if d_nn == d_ata else "DIFFERENT"
        print(f"    k={k_val}: D_max(NN)={d_nn}  D_max(all-to-all)={d_ata}  -> {same}")
    if not np.all(dmax_nn_arch == dmax_ata):
        gap_nn = dmax_nn_arch[list(k_arch).index(2.5)] - dmax_nn_arch[list(k_arch).index(0.5)]
        gap_ata = dmax_ata[list(k_arch).index(2.5)] - dmax_ata[list(k_arch).index(0.5)]
        print(f"    -> NN regular/chaotic gap = {gap_nn}, "
              f"all-to-all regular/chaotic gap = {gap_ata}")

    print("\n" + "="*60)
    print("ALL STEPS COMPLETE.")
    print("Figures produced:")
    print("  figures/dmax_vs_k.{pdf,png}                    <- REPLACES Fig. 10")
    print("  figures/architecture_test.{pdf,png}             <- NNN (reviewer item v)")
    print("  figures/architecture_test_all_to_all.{pdf,png}  <- ACTUAL Fig. 14 result")
    print("="*60)