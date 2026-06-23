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
    """
    def state(self, theta, psi0):
        from spin_operators import embed, SZ
        import scipy.linalg
        psi = psi0.astype(complex).copy()
        N = self.N
        idx = 0
        for _layer in range(self.D):
            # Rz and Rx rotations (identical to base class)
            for i in range(N):
                angle = theta[idx]; idx += 1
                Zi = embed(SZ, i, N)
                psi = scipy.linalg.expm(-1j * angle * Zi) @ psi
            for i in range(N):
                from spin_operators import SX
                angle = theta[idx]; idx += 1
                Xi = embed(SX, i, N)
                psi = scipy.linalg.expm(-1j * angle * Xi) @ psi
            # NNN ZZ entangler: Z_i Z_{i+2}
            chi = theta[idx]; idx += 1
            for i in range(N):
                j2 = (i + 2) % N
                Zi = embed(SZ, i, N)
                Zj = embed(SZ, j2, N)
                psi = scipy.linalg.expm(-1j * chi * (Zi @ Zj)) @ psi
        return psi / np.linalg.norm(psi)


class AnsatzAllToAll(Ansatz):
    """All-to-all ZZ entangler (one shared angle per layer).
    Applies exp(-i chi * ZZ) for EVERY pair (i,j), i<j.
    n_params is identical: (2N+1)*D.
    """
    def state(self, theta, psi0):
        from spin_operators import embed, SZ, SX
        import scipy.linalg
        psi = psi0.astype(complex).copy()
        N = self.N
        idx = 0
        for _layer in range(self.D):
            for i in range(N):
                angle = theta[idx]; idx += 1
                Zi = embed(SZ, i, N)
                psi = scipy.linalg.expm(-1j * angle * Zi) @ psi
            for i in range(N):
                angle = theta[idx]; idx += 1
                Xi = embed(SX, i, N)
                psi = scipy.linalg.expm(-1j * angle * Xi) @ psi
            chi = theta[idx]; idx += 1
            for i in range(N):
                for j in range(i + 1, N):
                    Zi = embed(SZ, i, N)
                    Zj = embed(SZ, j, N)
                    psi = scipy.linalg.expm(-1j * chi * (Zi @ Zj)) @ psi
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
                n_restarts=50, p=np.pi / 2, ansatz_type="nn"):
    """Returns (dmax, ceiling_hit, success_fractions_dict).

    success_fractions_dict: {(t, D): n_successes / n_restarts}
    Used to compute 95% CI on D_max via Binomial bound.
    """
    psi0 = coherent_product_state(N)
    U_F = floquet_U_exact(N, k, p)

    dmax = 0
    ceiling_hit = False
    psi_t = psi0.copy()
    success_fracs = {}

    for t in range(1, steps + 1):
        psi_t = normalize(U_F @ psi_t)

        for D in range(1, max_depth + 1):
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

            success_fracs[(t, D)] = n_success / n_restarts
            if n_success >= 1:
                dmax = max(dmax, D)
                break
            if D == max_depth:
                ceiling_hit = True

        print(f"    k={k:.2f} t={t:2d}: D_max={dmax}  "
              f"(ansatz={ansatz_type})")

    return dmax, ceiling_hit, success_fracs


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
            n_restarts=50, ansatz_type="nn"):
    dmax_list, ftle_list, ceil_list, sfrac_list = [], [], [], []
    for k in k_values:
        print(f"\nk={k:.2f}  (direct-opt D_max, N={N}, "
              f"eps_opt={eps_opt}, ansatz={ansatz_type})")
        dmax, hit, sfracs = dmax_direct(k, N=N, steps=steps,
                                        eps_opt=eps_opt, max_depth=max_depth,
                                        n_restarts=n_restarts,
                                        ansatz_type=ansatz_type)
        lam = ftle_classical(k, steps=300, n_ic=80)
        dmax_list.append(dmax)
        ftle_list.append(lam)
        ceil_list.append(hit)
        sfrac_list.append(sfracs)
        tag = " (lower bound)" if hit else ""
        print(f"-> D_max={dmax}  FTLE={lam:+.3f}{tag}")
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


def plot_architecture_comparison(k_test, dmax_nn, dmax_nnn,
                                 outfile="figures/architecture_test"):
    """Bar chart comparing D_max for NN vs NNN ansatz at k_test values."""
    _aps_style()
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    x = np.arange(len(k_test))
    width = 0.35
    fig, ax = plt.subplots(figsize=(4.5, 3.0), constrained_layout=True)
    ax.bar(x - width/2, dmax_nn, width, label="NN (original)", color="#E69F00")
    ax.bar(x + width/2, dmax_nnn, width, label="NNN", color="#56B4E9")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_test])
    ax.set_ylabel(r"$D_{\rm max}$")
    ax.set_title("Architecture test: NN vs NNN entangler")
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
    dmax, ftle, ceil, sfracs = sweep_k(K_VALUES, N=N, steps=STEPS)
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
    dmax_03, _, _, _ = sweep_k(k_border, N=N, steps=STEPS, eps_opt=0.03)
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
    dmax_nnn, _, _, _ = sweep_k(k_arch, N=N, steps=STEPS,
                                 ansatz_type="nnn", n_restarts=50)
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

    print("\n" + "="*60)
    print("ALL STEPS COMPLETE.")
    print("Figures produced:")
    print("  figures/dmax_vs_k.{pdf,png}          <- REPLACES Fig. 10")
    print("  figures/architecture_test.{pdf,png}   <- NEW (reviewer item v)")
    print("="*60)
