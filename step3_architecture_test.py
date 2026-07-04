"""
step3_architecture_test.py  --  Architecture test (reviewer item v).

Compares D_max under the layer-wise NN ansatz against an ALL-TO-ALL
entangler ansatz at k=0.5 and k=2.5.

WHY ALL-TO-ALL (not NNN):
  A pure next-nearest-neighbour (NNN) ring Z_i Z_{i+2} on N=6 is DISCONNECTED
  -- it splits into two triangles {0,2,4} and {1,3,5} that cannot be entangled
  with each other. It therefore cannot represent generic states (infidelity
  plateaus at ~0.29 regardless of depth), making it a meaningless comparison.
  The scientifically correct test uses MORE connectivity than NN, i.e.
  all-to-all ZZ entanglers, which the manuscript already discusses as the
  natural richer-ansatz baseline. This tests whether D_max reflects the
  physics (entanglement barrier) or merely the NN connectivity limit.

Steps 1 and 2 of plot_dmax_vs_k.py already completed:
  STEP 1 D_max (NN, eps_opt=0.05): k=0.5->4, k=2.5->5

Speed: N=6, all-to-all has 15 ZZ pairs. n_restarts=10, max_depth=6.
Each ZZ commutes with the others (all diagonal in Z basis), so we combine
them into a single diagonal exponential per layer -- much faster than
15 separate expm calls. Runtime ~10-20 min.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from scipy.linalg import expm

from qkt_quantum import floquet_U_exact
from spin_operators import coherent_product_state, normalize, embed, SZ, SX


def _build_generators(N):
    """Single-qubit generators + diagonal of every all-to-all ZZ pair."""
    Rz_gens = [embed(SZ, i, N) for i in range(N)]
    Rx_gens = [embed(SX, i, N) for i in range(N)]
    pairs   = [(i, j) for i in range(N) for j in range(i + 1, N)]
    # Each Z_iZ_j is diagonal; store its diagonal for fast exponentiation
    zz_diags = []
    for (i, j) in pairs:
        d = np.real(np.diag(embed(SZ, i, N) @ embed(SZ, j, N)))
        zz_diags.append(d)
    zz_diags = np.array(zz_diags)          # shape (n_pairs, 2^N)
    return Rz_gens, Rx_gens, zz_diags, len(pairs)


def state_all2all(theta, psi0, N, D, Rz_gens, Rx_gens, zz_diags, n_zz):
    """
    D-layer all-to-all ansatz.
    Per layer: N Rz, N Rx, then n_zz all-to-all ZZ angles.
    Rz/Rx use the exact Pauli shortcut. The ZZ block is fully diagonal,
    so exp(-i * sum_m chi_m * ZZ_m) = diag(exp(-i * sum_m chi_m * diag_m)).
    n_params per layer = 2N + n_zz.
    """
    psi = psi0.astype(complex).copy()
    ppl = 2 * N + n_zz
    for layer in range(D):
        base = layer * ppl
        # Rz -- exact shortcut (single-qubit Z squares to I)
        for i in range(N):
            a = theta[base + i]
            psi = np.cos(a) * psi - 1j * np.sin(a) * (Rz_gens[i] @ psi)
        # Rx -- exact shortcut
        for i in range(N):
            a = theta[base + N + i]
            psi = np.cos(a) * psi - 1j * np.sin(a) * (Rx_gens[i] @ psi)
        # All-to-all ZZ -- combine all diagonal generators into one phase
        chis = theta[base + 2 * N: base + 2 * N + n_zz]      # (n_zz,)
        phase = zz_diags.T @ chis                            # (2^N,)
        psi = np.exp(-1j * phase) * psi
    return psi / np.linalg.norm(psi)


def dmax_direct_all2all(k, N=6, steps=12, eps_opt=0.05, max_depth=6,
                        n_restarts=10, p=np.pi / 2):
    psi0 = coherent_product_state(N)
    U_F  = floquet_U_exact(N, k, p)
    Rz_gens, Rx_gens, zz_diags, n_zz = _build_generators(N)

    dmax  = 0
    psi_t = psi0.copy()

    for t in range(1, steps + 1):
        psi_t = normalize(U_F @ psi_t)
        psi_target = psi_t.copy()          # avoid closure-by-reference bug

        found_D    = None
        best_infid = 1.0

        for D in range(1, max_depth + 1):
            n_params  = (2 * N + n_zz) * D
            n_success = 0

            for r in range(n_restarts):
                rng = np.random.default_rng((int(k * 1000), t, D, r))
                x0  = rng.uniform(0, 2 * np.pi, n_params)

                def infidelity(th, _tgt=psi_target, _D=D):
                    psi_v = state_all2all(th, psi0, N, _D,
                                          Rz_gens, Rx_gens, zz_diags, n_zz)
                    return float(1.0 - abs(np.vdot(_tgt, psi_v)) ** 2)

                res = minimize(infidelity, x0, method="L-BFGS-B",
                               options={"maxiter": 400})
                best_infid = min(best_infid, res.fun)
                if res.fun <= eps_opt:
                    n_success += 1

            if n_success >= 1:
                found_D = D
                break

        if found_D is not None:
            dmax = max(dmax, found_D)

        print(f"    k={k:.2f} t={t:2d}: D_max={dmax}  "
              f"(best_infid={best_infid:.4f}, ansatz=all2all)")

    return dmax


def plot_architecture_comparison(k_test, dmax_nn, dmax_ata,
                                  outfile="figures/architecture_test"):
    mpl.rcParams.update({"font.family": "serif", "font.size": 9,
                          "axes.labelsize": 9, "legend.fontsize": 7.5})
    os.makedirs("figures", exist_ok=True)
    x, width = np.arange(len(k_test)), 0.35
    fig, ax = plt.subplots(figsize=(4.5, 3.0), constrained_layout=True)
    ax.bar(x - width/2, dmax_nn,  width,
           label="NN (nearest-neighbour)", color="#E69F00")
    ax.bar(x + width/2, dmax_ata, width,
           label="All-to-all ZZ", color="#56B4E9")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_test])
    ax.set_ylabel(
        r"$D_{\rm max}$ ($\varepsilon_{\rm opt}=0.05$, $N=6$, 12 steps)")
    ax.set_title("Architecture test: NN vs all-to-all entangler")
    ax.legend()
    ax.grid(True, ls=":", axis="y", alpha=0.5)
    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}.pdf / .png")


if __name__ == "__main__":
    N, STEPS = 6, 12
    k_arch       = np.array([0.5, 2.5])
    dmax_nn_arch = np.array([4,   5  ])          # from completed STEP 1

    print("=" * 60)
    print("STEP 3: Architecture test -- all-to-all entangler, k=0.5 and k=2.5")
    print("(Steps 1 and 2 already completed)")
    print("=" * 60)

    dmax_ata_results = []
    for k in k_arch:
        print(f"\nk={k:.2f}  (N={N}, eps_opt=0.05, ansatz=all2all, "
              f"n_restarts=10, max_depth=6)")
        d = dmax_direct_all2all(k, N=N, steps=STEPS,
                                n_restarts=10, max_depth=6)
        dmax_ata_results.append(d)
        print(f"-> D_max(all2all)={d}")

    dmax_ata = np.array(dmax_ata_results)
    plot_architecture_comparison(k_arch, dmax_nn_arch, dmax_ata)

    print("\n>>> PASTE INTO PAPER (Sec. Limitations item v):")
    for k_val, d_nn, d_ata in zip(k_arch, dmax_nn_arch, dmax_ata):
        same = "same" if d_nn == d_ata else "DIFFERENT"
        print(f"    k={k_val}: D_max(NN)={d_nn}  "
              f"D_max(all-to-all)={d_ata}  -> {same}")

    if np.all(dmax_nn_arch == dmax_ata):
        print("    -> D_max UNCHANGED under all-to-all entangler.")
        print("       The depth requirement reflects the entanglement barrier,")
        print("       NOT merely the NN connectivity limit. Strong result.")
    else:
        print("    -> D_max DIFFERS: richer connectivity changes the requirement.")
        print("       The resource claim is partly connectivity-dependent;")
        print("       report both numbers in Limitations item v.")

    print("\n" + "=" * 60)
    print("COMPLETE. Figure: figures/architecture_test.{pdf,png}")
    print("=" * 60)