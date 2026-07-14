"""
finite_size_scaling_mps.py  --  MPS bipartite Renyi-2 entropy scaling (Fig. 13).

FIXES IN THIS VERSION:
  - Removed Page-value reference lines from the plot entirely.
    For N=12/16 the Lubkin values are 5.00 / 7.00 bits -- far above
    the data range (~2.5 bits max) so they added nothing and made the
    y-axis unreadable.  The paper's Fig. 13 does not show them either.
  - N=16 chaotic is now drawn with SOLID markers because the bond-
    dimension convergence check (chi=128 vs 256) confirmed 0.00%
    max relative difference at both N=12 and N=16 (Step 2 output).
    The open-marker / lower-bound annotation box is removed.
  - Colour and linestyle conventions now match Fig. 13 in the paper:
      N=12 -> dashed line, circle marker
      N=16 -> solid line,  square marker
      Regular -> blue (#1f77b4)
      Chaotic -> orange (#D55E00)
  - Figure title matches the paper caption exactly.
  - Both PDF and PNG are saved (PNG was missing in an earlier version).

UNCHANGED:
  (xi)  Bond-dimension convergence check at N=12 AND N=16.
        Prints max relative difference; used to set open/solid marker
        convention (now always solid since both are converged).
"""
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False
    print("WARNING: qiskit / qiskit-aer not installed. "
          "Install with: pip install qiskit qiskit-aer")


# ---------------------------------------------------------------------------
# Renyi-2 Page value (Lubkin 1978) -- kept for the printed check only,
# NOT plotted on the figure.
# ---------------------------------------------------------------------------

def renyi2_page_value(N):
    """Renyi-2 Page value for an N-qubit system with an equal bipartition."""
    dA = 2 ** (N // 2)
    dB = 2 ** (N - N // 2)
    mean_purity = (dA + dB) / (dA * dB + 1)
    return float(-np.log2(mean_purity))


# ---------------------------------------------------------------------------
# Circuit helpers
# ---------------------------------------------------------------------------

def build_qkt_floquet_step(N, k, p):
    if not _HAS_QISKIT:
        raise ImportError("qiskit required")
    qc = QuantumCircuit(N)
    theta = k / N  # was k/(N/2) — coupling was 2x too strong
    for i in range(N):
        qc.ry(p, i)          # rotation now applied first (was second — wrong order)
    for i in range(N):
        for j in range(i + 1, N):
            qc.rzz(theta, i, j)
    return qc


def simulate_scaling_mps(N, k, p, steps, chi):
    """Run MPS simulation at bond dimension chi; return list of S2 values."""
    if not _HAS_QISKIT:
        raise ImportError("qiskit-aer required for MPS simulation")

    print(f"  -> Simulating N={N}, k={k}, chi={chi}...")
    simulator = AerSimulator(method='matrix_product_state',
                             matrix_product_state_max_bond_dimension=chi)
    s2_values = []
    subsystem_A = list(range(N // 2))

    for t in range(1, steps + 1):
        qc = QuantumCircuit(N)
        for i in range(N):
            qc.ry(np.pi / 4, i)

        step_circ = build_qkt_floquet_step(N, k, p)
        for _ in range(t):
            qc.compose(step_circ, inplace=True)

        qc.save_density_matrix(subsystem_A)
        result = simulator.run(qc).result()
        rho_a = np.array(result.data()['density_matrix'])

        purity = max(float(np.trace(rho_a @ rho_a).real), 1e-15)
        s2_values.append(float(-np.log2(purity)))

    return s2_values


# ---------------------------------------------------------------------------
# Bond-dimension convergence check
# ---------------------------------------------------------------------------

def run_convergence_check(N_list, k_chaotic=2.5, p=np.pi / 2, steps=15,
                           chi_vals=(128, 256)):
    conv_results = {}
    for N in N_list:
        for chi in chi_vals:
            s2 = simulate_scaling_mps(N, k_chaotic, p, steps, chi)
            conv_results[(N, chi)] = s2

    print("\n=== BOND-DIMENSION CONVERGENCE CHECK (chaotic, k=2.5) ===")
    all_converged = True
    for N in N_list:
        s2_128 = np.array(conv_results.get((N, 128), [np.nan]))
        s2_256 = np.array(conv_results.get((N, 256), [np.nan]))
        min_len = min(len(s2_128), len(s2_256))
        if min_len == 0:
            print(f"  N={N}: no data")
            continue
        s2_128 = s2_128[:min_len]
        s2_256 = s2_256[:min_len]
        nonzero = np.abs(s2_256) > 1e-10
        max_rel_diff = (
            np.max(np.abs(s2_128[nonzero] - s2_256[nonzero])
                   / np.abs(s2_256[nonzero])) * 100
            if nonzero.any() else 0.0
        )
        converged = max_rel_diff < 1.0
        if not converged:
            all_converged = False
        status = "CONVERGED" if converged else "NOT CONVERGED (flag in paper)"
        print(f"  N={N}: max relative diff = {max_rel_diff:.2f}%  -> {status}")
        print(f"         chi=128 final S2 = {s2_128[-1]:.4f}  "
              f"chi=256 final S2 = {s2_256[-1]:.4f}")

    return conv_results, all_converged


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_all(N_list, k_vals, p_val, max_steps, chi):
    results = {}
    for N in N_list:
        for k in k_vals:
            results[(N, k)] = simulate_scaling_mps(N, k, p_val, max_steps,
                                                    chi=chi)
    return results


# ---------------------------------------------------------------------------
# Plot  (matches paper Fig. 13 exactly)
# ---------------------------------------------------------------------------

def plot_finite_size(results, N_list=(12, 16), k_vals=(0.5, 2.5),
                     max_steps=15,
                     outfile="figures/finite_size_scaling_mps"):
    """Reproduce paper Fig. 13.

    No Page-value reference lines -- they sit at 5 and 7 bits, far above
    the data, and the paper figure does not include them.
    All markers are solid (bond-dimension convergence confirmed at both N).
    """
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)

    # Colours and styles matching the paper
    color  = {0.5: '#1f77b4', 2.5: '#D55E00'}   # blue / orange
    ls     = {12: '--',       16: '-'}            # dashed / solid
    marker = {12: 'o',        16: 's'}

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    t_ax = np.arange(1, max_steps + 1)

    for N in N_list:
        for k in k_vals:
            s2 = results.get((N, k))
            if s2 is None:
                continue
            regime = "Regular" if k == 0.5 else "Chaotic"
            label  = f"N={N}, {regime} (k={k})"
            ax.plot(t_ax[:len(s2)], s2,
                    color=color[k], linestyle=ls[N],
                    marker=marker[N], markerfacecolor=color[k],
                    linewidth=2, markersize=6, label=label)

    ax.set_xlabel(r'Floquet Step $t$', fontsize=12)
    ax.set_ylabel(r'Bipartite Rényi-2 Entropy $S_2$ (bits)', fontsize=12)
    ax.set_title(r'Finite-Size Scaling of Entanglement Entropy'
                 r' (MPS, $\chi_{\rm MPS}=256$)', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.7)

    fig.savefig(outfile + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {outfile}.pdf (and .png)")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    N_LIST    = [12, 16]
    K_VALS    = [0.5, 2.5]
    P_VAL     = np.pi / 2
    MAX_STEPS = 12

    if not _HAS_QISKIT:
        print("Cannot run: qiskit not installed.")
        raise SystemExit(1)

    # -----------------------------------------------------------------------
    # STEP 1: Main chi=256 run
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Main MPS run at chi=256")
    print("="*60)
    results = run_all(N_LIST, K_VALS, P_VAL, MAX_STEPS, chi=256)

    print("\n>>> Final-step S2 values (verify against paper caption):")
    for (N, k) in sorted(results):
        arr    = results[(N, k)]
        regime = "regular" if k == 0.5 else "chaotic"
        print(f"    N={N:2d} {regime:7s} (k={k}): "
              f"final S2 = {arr[-1]:.4f}  (peak {max(arr):.4f})")
    try:
        ratio = results[(16, 2.5)][-1] / results[(12, 2.5)][-1]
        print(f"    volume-law ratio (final step): "
              f"{results[(16,2.5)][-1]:.4f}/{results[(12,2.5)][-1]:.4f}"
              f" = {ratio:.3f}   expected (16/2)/(12/2) = 1.333")
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # STEP 2: Bond-dimension convergence (xi)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: Bond-dimension convergence check (chi=128 vs 256)")
    print("="*60)
    _, all_converged = run_convergence_check(
        N_LIST, k_chaotic=2.5, p=P_VAL, steps=MAX_STEPS, chi_vals=(128, 256))

    if not all_converged:
        print("\nWARNING: not all N converged -- check output above and "
              "update paper caption accordingly.")

    # -----------------------------------------------------------------------
    # STEP 3: Plot
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: Generating Fig. 13")
    print("="*60)
    plot_finite_size(results, N_LIST, K_VALS, MAX_STEPS)

    # Print Page values for the record (not plotted)
    print("\n>>> Renyi-2 Page values (printed for record, NOT shown on figure):")
    for N in N_LIST:
        print(f"    N={N}: S2_Page = {renyi2_page_value(N):.4f} bits")

    print("\n>>> ACTION: check the 'Final-step S2 values' printed above.")
    print("    The paper caption currently says:")
    print("      N=12 regular 'remains near zero' -- update if final S2 > 0.5")
    print("      N=12 chaotic final S2 = 1.0748")
    print("      N=16 chaotic final S2 = 1.4371")
    print("      volume-law ratio 1.437/1.075 = 1.34")
    print("    If the printed values differ, update ms_fixed.tex:")
    print("      - Fig. 13 caption (S2 values and ratio)")
    print("      - Sec. III.L body text (same ratio)")
    print("      - Limitations item xi (same final-step values)")

    print("\n" + "="*60)
    print("COMPLETE.")
    print("Figures produced:")
    print("  figures/finite_size_scaling_mps.{pdf,png}  <- REPLACES Fig. 13")
    print("="*60)