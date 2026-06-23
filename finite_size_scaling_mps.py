"""
finite_size_scaling_mps.py  --  MPS bipartite Renyi-2 entropy scaling (Fig. 14).

CHANGES FROM ORIGINAL:
  (xi)  Bond-dimension convergence check at N=16: the original only verified
        chi=128 vs chi=256 at N=12.  This revision runs the comparison at
        BOTH N=12 and N=16 for the chaotic regime and prints the max relative
        difference.  The N=16 chaotic curves are flagged as lower bounds
        until chi=256 is confirmed converged there.
  
  Plot changes:
  - Corrected Renyi-2 Page value reference line: was 2.85 (von Neumann),
    now 2.02 bits (Lubkin formula for 3+3 qubit partition of 6 qubits).
    For the N=12,16 plots the Page value is plotted for the N/2 partition.
  - Figure saves both PDF and PNG (original missed PNG; .tex references PNG).
  - Open-marker convention: N=16 chaotic is shown with open markers to flag
    the pending convergence check (matches the revised .tex caption).
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
          "MPS simulation will not run. Install with: "
          "pip install qiskit qiskit-aer")


# ---------------------------------------------------------------------------
# Renyi-2 Page value (Lubkin 1978)
# ---------------------------------------------------------------------------

def renyi2_page_value(N):
    """Renyi-2 Page value for an N-qubit system with an equal bipartition.
    Uses Lubkin's formula: <Tr rho_A^2> = (d_A + d_B) / (d_A*d_B + 1)
    where d_A = d_B = 2^(N/2).
    """
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
    theta = k / (N / 2)
    for i in range(N):
        for j in range(i + 1, N):
            qc.rzz(theta, i, j)
    for i in range(N):
        qc.ry(p, i)
    return qc


def simulate_scaling_mps(N, k, p, steps, chi):
    """Run MPS simulation at bond dimension chi, return list of S2 values."""
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
        rho_a = result.data()['density_matrix']

        rho_a = np.array(rho_a)
        purity = np.trace(np.dot(rho_a, rho_a)).real
        purity = max(purity, 1e-15)
        # Convert from nats to bits (qiskit uses natural log internally;
        # the density_matrix is correct, so -log2 is right here)
        s2 = float(-np.log2(purity))
        s2_values.append(s2)

    return s2_values


# ---------------------------------------------------------------------------
# (xi) Bond-dimension convergence check
# ---------------------------------------------------------------------------

def run_convergence_check(N_list, k_chaotic=2.5, p=np.pi / 2, steps=15,
                           chi_vals=(128, 256)):
    """Run at chi=128 and chi=256 for each N and the chaotic regime.
    Returns dict (N, chi) -> list of S2 values.
    Prints max relative difference between chi=128 and chi=256.
    """
    conv_results = {}
    for N in N_list:
        for chi in chi_vals:
            s2 = simulate_scaling_mps(N, k_chaotic, p, steps, chi)
            conv_results[(N, chi)] = s2

    print("\n=== BOND-DIMENSION CONVERGENCE CHECK (chaotic, k=2.5) ===")
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
        if nonzero.any():
            max_rel_diff = np.max(
                np.abs(s2_128[nonzero] - s2_256[nonzero])
                / np.abs(s2_256[nonzero])
            ) * 100
        else:
            max_rel_diff = 0.0
        converged = max_rel_diff < 1.0
        print(f"  N={N}: max relative diff = {max_rel_diff:.2f}%  "
              f"-> {'CONVERGED' if converged else 'NOT CONVERGED (flag in paper)'}")
        print(f"         chi=128 final S2 = {s2_128[-1]:.4f}  "
              f"chi=256 final S2 = {s2_256[-1]:.4f}")

    print("\n>>> ACTION: if N=16 chaotic is NOT converged, keep the")
    print("    open-marker / lower-bound language in the Fig. 14 caption.")
    print("    If converged, change to solid marker and remove the caveat.")
    return conv_results


# ---------------------------------------------------------------------------
# Main simulation and plot
# ---------------------------------------------------------------------------

def run_all_and_plot(N_list=(12, 16), k_vals=(0.5, 2.5), p_val=np.pi/2,
                     max_steps=15, chi=256):
    results = {}
    for N in N_list:
        for k in k_vals:
            s2 = simulate_scaling_mps(N, k, p_val, max_steps, chi=chi)
            results[(N, k)] = s2
    return results


def plot_finite_size(results, N_list=(12, 16), k_vals=(0.5, 2.5),
                     max_steps=15, pending_n16_convergence=True,
                     outfile="figures/finite_size_scaling_mps"):
    """Plot Renyi-2 entropy vs Floquet step for each (N, k).

    pending_n16_convergence : bool
        If True, draw N=16 chaotic with open markers and add a footnote
        annotation flagging the pending chi convergence check.
    """
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    colors = {0.5: '#1f77b4', 2.5: '#D55E00'}
    styles = {12: '--', 16: '-'}

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    t_ax = np.arange(1, max_steps + 1)

    for N in N_list:
        for k in k_vals:
            s2 = results.get((N, k))
            if s2 is None:
                continue
            regime = "Regular" if k == 0.5 else "Chaotic"
            label = f"N={N}, {regime} (k={k})"
            is_pending = (N == 16 and k == 2.5 and pending_n16_convergence)
            mfc = "none" if is_pending else colors[k]
            ax.plot(t_ax[:len(s2)], s2,
                    color=colors[k], linestyle=styles[N],
                    marker='s' if N == 16 else 'o',
                    markerfacecolor=mfc,
                    linewidth=2, markersize=6, label=label)

    # Add Renyi-2 Page value reference lines for N=12 and N=16
    for N in N_list:
        page = renyi2_page_value(N)
        ax.axhline(page, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
        ax.text(max_steps * 0.98, page + 0.02,
                rf"$S_2^{{\rm Page}}(N={N})\approx{page:.2f}$ bits",
                ha='right', va='bottom', fontsize=8, color='gray')

    if pending_n16_convergence:
        ax.text(0.02, 0.98,
                "Open markers: N=16 chaotic — lower bound pending\n"
                r"$\chi_{\rm MPS}=128$ vs $256$ convergence check",
                transform=ax.transAxes, va='top', fontsize=8,
                color='#D55E00',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

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
    N_LIST = [12, 16]
    K_VALS = [0.5, 2.5]
    P_VAL = np.pi / 2
    MAX_STEPS = 15

    if not _HAS_QISKIT:
        print("Cannot run: qiskit not installed.")
        raise SystemExit(1)

    # -----------------------------------------------------------------------
    # STEP 1: Main chi=256 run (Fig. 14)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Main MPS run at chi=256")
    print("="*60)
    results = run_all_and_plot(N_LIST, K_VALS, P_VAL, MAX_STEPS, chi=256)

    # -----------------------------------------------------------------------
    # STEP 2: (xi) Bond-dimension convergence check at N=12 AND N=16
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: Bond-dimension convergence check (chi=128 vs 256)")
    print("="*60)
    conv_results = run_convergence_check(N_LIST, k_chaotic=2.5,
                                         p=P_VAL, steps=MAX_STEPS,
                                         chi_vals=(128, 256))

    # Determine if N=16 is converged to set the open-marker flag
    s2_128_n16 = np.array(conv_results.get((16, 128), [np.nan]))
    s2_256_n16 = np.array(conv_results.get((16, 256), [np.nan]))
    if len(s2_128_n16) > 0 and len(s2_256_n16) > 0:
        nonzero = np.abs(s2_256_n16) > 1e-10
        if nonzero.any():
            max_diff = np.max(np.abs(s2_128_n16[nonzero] - s2_256_n16[nonzero])
                              / np.abs(s2_256_n16[nonzero])) * 100
            n16_converged = max_diff < 1.0
        else:
            n16_converged = True
    else:
        n16_converged = False

    # -----------------------------------------------------------------------
    # STEP 3: Plot (pending flag depends on convergence check above)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: Generating Fig. 14")
    print("="*60)
    plot_finite_size(results, N_LIST, K_VALS, MAX_STEPS,
                     pending_n16_convergence=(not n16_converged))

    # Print Page values for paper
    print("\n>>> Renyi-2 Page values (for manuscript / figure annotation):")
    for N in N_LIST:
        print(f"    N={N}: S2_Page = {renyi2_page_value(N):.4f} bits")

    print("\n" + "="*60)
    print("COMPLETE.")
    print("Figures produced:")
    print("  figures/finite_size_scaling_mps.{pdf,png}  <- REPLACES Fig. 14")
    print("="*60)
