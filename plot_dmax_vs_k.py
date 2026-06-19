import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from adaptive_vqs_qkt import run_adaptive_floquet, _aps_style

def kicked_top_map_3d(x, y, z, k, p=np.pi/2):
    
    
    twist = k * z
    x1 = x * np.cos(twist) - y * np.sin(twist)
    y1 = x * np.sin(twist) + y * np.cos(twist)
    z1 = z

    
    x2 = x1 * np.cos(p) + z1 * np.sin(p)
    y2 = y1
    z2 = -x1 * np.sin(p) + z1 * np.cos(p)

    return x2, y2, z2

def ftle_classical(k: float, steps: int = 200, p: float = np.pi / 2,
                   n_ic: int = 50, eps: float = 1e-7, rng_seed: int = 0) -> float:
    rng = np.random.default_rng(rng_seed)
    lambdas = []

    
    z0 = rng.uniform(-1, 1, n_ic)
    phi0 = rng.uniform(0, 2 * np.pi, n_ic)
    x0 = np.sqrt(1 - z0**2) * np.cos(phi0)
    y0 = np.sqrt(1 - z0**2) * np.sin(phi0)

    for i in range(n_ic):
        x, y, z = x0[i], y0[i], z0[i]

        
        xp, yp, zp = x + eps, y, z
        norm_p = np.sqrt(xp**2 + yp**2 + zp**2)
        xp, yp, zp = xp / norm_p, yp / norm_p, zp / norm_p

        acc = 0.0
        for _ in range(steps):
            x_next, y_next, z_next = kicked_top_map_3d(x, y, z, k, p)
            xp_next, yp_next, zp_next = kicked_top_map_3d(xp, yp, zp, k, p)

            
            dist = np.sqrt((xp_next - x_next)**2 + (yp_next - y_next)**2 + (zp_next - z_next)**2 + 1e-30)
            acc += np.log(dist / eps)

            
            scale = eps / dist
            xp = x_next + (xp_next - x_next) * scale
            yp = y_next + (yp_next - y_next) * scale
            zp = z_next + (zp_next - z_next) * scale

            norm = np.sqrt(xp**2 + yp**2 + zp**2)
            xp, yp, zp = xp / norm, yp / norm, zp / norm
            x, y, z = x_next, y_next, z_next

        lambdas.append(acc / steps)

    return float(np.mean(lambdas))

def sweep_k(
    k_values,
    N: int = 6,
    steps: int = 12,
    residual_threshold: float = 0.80,  
    M_consec: int = 2,                 
    max_depth: int = 8,
):
    # IMPORTANT: the Dmax computed in this function is max(depth history) of
    # the *adaptive McLachlan integrator* (run_adaptive_floquet), triggered
    # by residual_threshold and capped at max_depth. This is NOT the same
    # quantity as the direct-optimization Dmax (Sec. II.E.4 of the
    # manuscript / Fig. 10), which is obtained independently via L-BFGS-B
    # over circuit parameters with no integrator involved at all. The two
    # can disagree, and the manuscript's central monotonicity claim is
    # about the direct-optimization Dmax, not this integrator-based one.
    # If you are using the output of this script to investigate referee
    # points #2 (k-resolution) or #8 (Dmax=3 plateau at k=2.5, k=3.0), make
    # sure you are looking at the same Dmax definition the manuscript text
    # actually claims monotonicity for -- cross-check against the
    # direct-optimization pipeline (adapt_vqa_baseline.layerwise_prepare or
    # equivalent) before drawing conclusions from this script alone.
    #
    # BUG FIX: this previously called run_adaptive_floquet with
    # compute_exact_diag=True. Looking at adaptive_vqs_qkt.py's actual
    # trigger logic:
    #
    #     if compute_exact_diag:
    #         trigger_metric = 1.0 - fidelity   # exact-state infidelity
    #         active_thresh = 0.45              # HARDCODED, ignores
    #                                            # residual_threshold entirely
    #     else:
    #         trigger_metric = peak_r2          # the McLachlan residual
    #         active_thresh = residual_threshold
    #
    # compute_exact_diag=True silently switches the trigger from the
    # McLachlan residual (what residual_threshold=0.80 was meant to
    # control) to exact-state infidelity against a HARDCODED 0.45
    # threshold that residual_threshold has no effect on at all. An
    # infidelity threshold of 0.45 is aggressive enough that a D=1 ansatz
    # trips it almost every period even in the regular regime, which is
    # why the previous run produced D_max=8 (the ceiling) at every single
    # k, including k=0.5. Setting compute_exact_diag=False restores the
    # actual residual-threshold-driven trigger this sweep is supposed to
    # be testing. Exact fidelity is no longer computed at all in this mode
    # (out["fid_diag"] will be NaN); if you want exact-fidelity diagnostics
    # alongside a residual-driven trigger, that requires editing
    # adaptive_vqs_qkt.run_adaptive_floquet itself so the two are
    # decoupled (diagnostic-only fidelity tracking that does NOT also
    # override active_thresh), which is out of scope for this file.
    dmax_list  = []
    ftle_list  = []

    for k in k_values:
        print(f"\nk={k:.2f}, threshold={residual_threshold}, M_consec={M_consec}, max_depth={max_depth}")

        result = run_adaptive_floquet(
            N=N, k=k, steps=steps,
            residual_threshold=residual_threshold,
            M_consec=M_consec,
            max_depth=max_depth,
            compute_exact_diag=False,
        )
        
        print(f"  depth history: {result['depth']}")
        
        dmax = max(result["depth"])
        dmax_list.append(dmax)

        lam = ftle_classical(k, steps=300, n_ic=80)
        ftle_list.append(lam)

        print(f"-> D_max={dmax}  <lambda_FTLE>={lam:+.3f}")

    return np.array(dmax_list), np.array(ftle_list)

def plot_dmax_vs_k(k_values, dmax, ftle,
                   outfile: str = "figures/dmax_vs_k"):
    _aps_style()
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)

    ORANGE = "#E69F00"
    BLUE   = "#0072B2"

    fig, ax1 = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)
    ax2 = ax1.twinx()

    l1, = ax1.plot(k_values, dmax, "o-", color=ORANGE, lw=1.5, ms=5,
                   label=r"$D_{\rm max}$")
    ax1.set_xlabel(r"Kick strength $k$")
    ax1.set_ylabel(r"Max adaptive depth $D_{\rm max}$", color=ORANGE)
    ax1.tick_params(axis="y", labelcolor=ORANGE)
    ax1.set_ylim(0, max(dmax) + 1.5)
    ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    l2, = ax2.plot(k_values, ftle, "s--", color=BLUE, lw=1.5, ms=5,
                   label=r"$\langle\lambda_{\rm FTLE}\rangle$")
    ax2.set_ylabel(r"$\langle\lambda_{\rm FTLE}(k)\rangle$", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)

    ftle_max = max(ftle.max() * 1.2, 0.05)
    ftle_min = min(ftle.min() - 0.05, -0.02)
    ax2.set_ylim(ftle_min, ftle_max)

    lines  = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=7.5)

    ax1.grid(True, ls=":", alpha=0.45)
    ax1.set_xticks(k_values)
    # With a finer k-grid (referee point #2) there can be more than 7
    # tick labels; rotate them so they stay legible instead of overlapping.
    if len(k_values) > 7:
        ax1.tick_params(axis="x", labelrotation=45)

    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {outfile}.pdf / .png")

if __name__ == "__main__":
    # Higher-resolution kick-strength grid (referee point #2): the original
    # 7-point grid [0.5, 1.0, ..., 3.5] produced a coarse Dmax staircase
    # that left it unclear whether Dmax(k) is genuinely monotonic or just
    # non-decreasing at low resolution. The added points (1.75, 2.25, 2.75,
    # 3.25) sit inside the regions of fastest FTLE growth and around the
    # k=2.5/k=3.0 Dmax=3 plateau (referee point #8), where extra resolution
    # is most informative.
    K_VALUES = np.array([0.5, 1.0, 1.5, 1.75, 2.0, 2.25,
                          2.5, 2.75, 3.0, 3.25, 3.5])
    N        = 6
    STEPS    = 12

    print("Sweeping k ...")
    dmax, ftle = sweep_k(K_VALUES, N=N, steps=STEPS)

    print("\nResults:")
    for k, d, l in zip(K_VALUES, dmax, ftle):
        print(f"  k={k:.2f}  D_max={d}  lambda_FTLE={l:+.4f}")

    # Explicit monotonicity check, so the manuscript claim ("Dmax grows
    # monotonically with the Lyapunov exponent") is verified against the
    # actual data rather than asserted from a quick visual read of the plot.
    diffs = np.diff(dmax)
    is_monotonic_nondecreasing = bool(np.all(diffs >= 0))
    is_strictly_increasing = bool(np.all(diffs > 0))
    print(f"\nDmax sequence: {list(dmax)}")
    print(f"Non-decreasing (weakly monotonic): {is_monotonic_nondecreasing}")
    print(f"Strictly increasing at every step: {is_strictly_increasing}")
    if not is_monotonic_nondecreasing:
        bad = [(K_VALUES[i], K_VALUES[i+1], dmax[i], dmax[i+1])
               for i in range(len(diffs)) if diffs[i] < 0]
        print("WARNING: Dmax DECREASES at the following k transitions "
              "-- the monotonicity claim does NOT hold as stated and the "
              "manuscript text needs to be revised:")
        for k1, k2, d1, d2 in bad:
            print(f"    k={k1:.2f} (Dmax={d1}) -> k={k2:.2f} (Dmax={d2})")
    elif not is_strictly_increasing:
        flat = [(K_VALUES[i], K_VALUES[i+1], dmax[i])
                for i in range(len(diffs)) if diffs[i] == 0]
        print("Dmax is non-decreasing but has plateaus (ties) at:")
        for k1, k2, d in flat:
            print(f"    k={k1:.2f} -> k={k2:.2f}, both Dmax={d}")

    plot_dmax_vs_k(K_VALUES, dmax, ftle)