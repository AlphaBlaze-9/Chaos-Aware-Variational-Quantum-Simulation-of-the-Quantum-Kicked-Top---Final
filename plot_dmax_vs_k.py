

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
    dmax_list  = []
    ftle_list  = []

    for k in k_values:
        print(f"\nk={k:.2f}, threshold={residual_threshold}, M_consec={M_consec}, max_depth={max_depth}")

        result = run_adaptive_floquet(
            N=N, k=k, steps=steps,
            residual_threshold=residual_threshold,
            M_consec=M_consec,
            max_depth=max_depth,
            compute_exact_diag=True,
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

    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {outfile}.pdf / .png")

if __name__ == "__main__":
    K_VALUES = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    N        = 6
    STEPS    = 12

    print("Sweeping k ...")
    dmax, ftle = sweep_k(K_VALUES, N=N, steps=STEPS)

    print("\nResults:")
    for k, d, l in zip(K_VALUES, dmax, ftle):
        print(f"  k={k:.1f}  D_max={d}  lambda_FTLE={l:+.4f}")

    plot_dmax_vs_k(K_VALUES, dmax, ftle)