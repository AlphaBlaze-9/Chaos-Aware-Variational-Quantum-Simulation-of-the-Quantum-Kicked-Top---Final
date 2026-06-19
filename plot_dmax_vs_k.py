import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from adaptive_vqs_qkt import _aps_style
from adapt_vqa_baseline import layerwise_prepare
from qkt_quantum import floquet_U_exact
from spin_operators import coherent_product_state, normalize


# =====================================================================
# Classical FTLE (unchanged): used only for the right-hand axis curve.
# =====================================================================
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
            dist = np.sqrt((xpn - xn) ** 2 + (ypn - yn) ** 2 + (zpn - zn) ** 2 + 1e-30)
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


# =====================================================================
# D_max via DIRECT optimization (the manuscript's Fig. 10 / Sec. II.E.4).
#
# This is the CORRECTED pipeline. The previous plot_dmax_vs_k.py computed
# D_max from the adaptive McLachlan INTEGRATOR (run_adaptive_floquet), which
# is explicitly NOT what the Fig. 10 caption claims. The manuscript states
# D_max is the minimum depth that reaches infidelity eps_opt <= 0.05 in
# representing U_F^t|psi0> for ANY t in {1,...,12}, found by direct L-BFGS-B
# optimization with 50 random restarts, independent of any integrator.
#
# Definition used here (matching the text): for each k,
#     D_max(k) = max_t  [ min D such that some restart hits infidelity<eps ]
# i.e. the deepest "minimum sufficient depth" across the time window.
# =====================================================================
def dmax_direct(k, N=6, steps=12, eps_opt=0.05, max_depth=8,
                n_restarts=50, p=np.pi / 2):
    psi0 = coherent_product_state(N)
    U_F = floquet_U_exact(N, k, p)

    dmax = 0
    ceiling_hit = False
    psi_t = psi0.copy()
    for t in range(1, steps + 1):
        psi_t = normalize(U_F @ psi_t)
        res = layerwise_prepare(psi0, psi_t, N, eps_opt=eps_opt,
                                max_depth=max_depth, n_restarts=n_restarts)
        dmax = max(dmax, res["depth"])
        if not res["sufficient"]:
            ceiling_hit = True   # this t only reached a lower bound
        print(f"    k={k:.2f} t={t:2d}: depth={res['depth']} "
              f"sufficient={res['sufficient']} "
              f"(n_success={res['n_success']}/{res['n_restarts']})")
    return dmax, ceiling_hit


def sweep_k(k_values, N=6, steps=12, eps_opt=0.05, max_depth=8, n_restarts=50):
    dmax_list, ftle_list, ceil_list = [], [], []
    for k in k_values:
        print(f"\nk={k:.2f}  (direct-optimization D_max, N={N}, eps_opt={eps_opt})")
        dmax, hit = dmax_direct(k, N=N, steps=steps, eps_opt=eps_opt,
                                max_depth=max_depth, n_restarts=n_restarts)
        lam = ftle_classical(k, steps=300, n_ic=80)
        dmax_list.append(dmax)
        ftle_list.append(lam)
        ceil_list.append(hit)
        print(f"-> D_max={dmax}  <lambda_FTLE>={lam:+.3f}  ceiling_hit={hit}")
    return np.array(dmax_list), np.array(ftle_list), ceil_list


def plot_dmax_vs_k(k_values, dmax, ftle, ceil, outfile="figures/dmax_vs_k"):
    _aps_style()
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    ORANGE, BLUE = "#E69F00", "#0072B2"

    fig, ax1 = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)
    ax2 = ax1.twinx()

    l1, = ax1.plot(k_values, dmax, "o-", color=ORANGE, lw=1.5, ms=5,
                   label=r"$D_{\rm max}$")
    # mark any point that hit the depth ceiling as an open (lower-bound) marker
    for i, hit in enumerate(ceil):
        if hit:
            ax1.plot(k_values[i], dmax[i], "o", color=ORANGE,
                     mfc="none", ms=10)
    ax1.set_xlabel(r"Kick strength $k$")
    ax1.set_ylabel(r"Max adaptive depth $D_{\rm max}$", color=ORANGE)
    ax1.tick_params(axis="y", labelcolor=ORANGE)
    ax1.set_ylim(0, max(dmax) + 1.5)
    ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    l2, = ax2.plot(k_values, ftle, "s--", color=BLUE, lw=1.5, ms=5,
                   label=r"$\langle\lambda_{\rm FTLE}\rangle$")
    ax2.set_ylabel(r"$\langle\lambda_{\rm FTLE}(k)\rangle$", color=BLUE)
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


if __name__ == "__main__":
    # Finer k-grid (referee #2) around the FTLE rise and the k=2.5/3.0
    # plateau (referee #8). The 7 manuscript points are a subset of these.
    K_VALUES = np.array([0.5, 1.0, 1.5, 1.75, 2.0, 2.25,
                         2.5, 2.75, 3.0, 3.25, 3.5])
    N, STEPS = 6, 12

    print("Sweeping k via DIRECT optimization (this is the expensive run)...")
    dmax, ftle, ceil = sweep_k(K_VALUES, N=N, steps=STEPS)

    print("\nResults:")
    for k, d, l, h in zip(K_VALUES, dmax, ftle, ceil):
        tag = " (lower bound: ceiling hit)" if h else ""
        print(f"  k={k:.2f}  D_max={d}  lambda_FTLE={l:+.4f}{tag}")

    diffs = np.diff(dmax)
    nondec = bool(np.all(diffs >= 0))
    strict = bool(np.all(diffs > 0))
    print(f"\n[VERIFY] D_max sequence: {list(dmax)}")
    print(f"[VERIFY] non-decreasing: {nondec}   strictly increasing: {strict}")
    print("[VERIFY] manuscript states the 7-point grid gives "
          "D_max = [1,1,1,2,3,3,4] at k=[0.5..3.5]; confirm the matching "
          "subset of points reproduces this, else update Sec. III H / "
          "Conclusion and the Fig. 10 in-text values.")
    if not nondec:
        bad = [(K_VALUES[i], K_VALUES[i + 1], dmax[i], dmax[i + 1])
               for i in range(len(diffs)) if diffs[i] < 0]
        print("WARNING: D_max DECREASES somewhere -- 'monotonic' claim fails:")
        for k1, k2, d1, d2 in bad:
            print(f"    k={k1:.2f}(D={d1}) -> k={k2:.2f}(D={d2})")

    plot_dmax_vs_k(K_VALUES, dmax, ftle, ceil)
