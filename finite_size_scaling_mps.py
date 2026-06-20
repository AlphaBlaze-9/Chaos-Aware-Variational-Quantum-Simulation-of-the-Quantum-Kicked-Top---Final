"""
Finite-size scaling of bipartite Renyi-2 entropy via Qiskit Aer's
matrix-product-state (MPS) simulator.

THIS SCRIPT HAD THREE BUGS THAT ARE FIXED HERE:

1. WRONG GATE ANGLE / ORDERING. The previous version applied RZZ(theta) with
   theta = k/(N/2) BEFORE RY(p), using a different angle convention than the
   rest of the codebase. The repo's own validated Floquet decomposition
   (qkt_quantum.floquet_gate_sequence, which floquet_U_trotter is built from
   and which is cross-checked against floquet_U_exact via trotter_error)
   applies RY(p) on every qubit FIRST (free precession), THEN RZZ(2a) on
   every pair with 2a = (k/(2j))/2 = k/(4j) = k/(2N) (the twist), plus an
   overall global phase that does not affect entropy/purity but is included
   for consistency. The old angle and ordering did not match this and so was
   NOT simulating the Hamiltonian in Eq. (1)/(2) of the manuscript.

2. NO BOND DIMENSION WAS EVER SET. AerSimulator(method='matrix_product_state')
   with no matrix_product_state_max_bond_dimension argument runs with NO
   truncation at all (Aer's documented default), i.e. it is exact MPS, not a
   bond-dimension-limited approximation. The manuscript caption claims
   chi_MPS=256 with convergence checked against chi=128 -- that comparison
   never actually happened. This version sets an explicit bond dimension and
   runs BOTH chi=128 and chi=256 so the convergence claim is genuinely
   checked (and reports the max relative difference, exactly as the caption
   says).

3. NUMERICALLY FRAGILE PURITY. tr(rho_A^2) via a dense product+trace on the
   returned density matrix can drift slightly outside [0,1] from floating
   point noise. This version clips purity into (0,1] before the log and
   warns if it is ever found meaningfully outside [1e-6, 1+1e-8], which
   would indicate a genuine numerical problem worth investigating rather
   than silently plotting a bad point (this is the suspected cause of the
   late-time entropy DROP seen in the previous broken run).

SELF-VALIDATION: before trusting the expensive N=12,16 MPS runs, this script
first reproduces small-N (N=6, N=8) Renyi-2 entropy curves with the EXACT
'statevector' method and compares them against the repo's own exact
diagonalization (qkt_quantum.floquet_U_exact + spin_operators.collective_J),
which is what every other figure in the paper is built from. If these don't
agree to high precision, the gate circuit is still wrong and the MPS numbers
should not be trusted -- the script says so explicitly and aborts rather
than silently proceeding.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spin_operators import coherent_product_state
from qkt_quantum import floquet_U_exact


def build_qkt_floquet_step(N, k, p):
    """One Floquet period, matching floquet_gate_sequence's PHYSICAL
    convention: RY(p) on every qubit (free precession), THEN a ZZ twist of
    exp(-i*(2a)*Z_i Z_j) on every pair, with 2a = (k/(2j))/2 = k/(2N).

    IMPORTANT: Qiskit's RZZ(theta) gate is defined as
    exp(-i*(theta/2)*Z(tensor)Z) (note the extra /2 -- see Qiskit's RZZGate
    docs), NOT exp(-i*theta*Z(tensor)Z). The repo's floquet_U_trotter builds
    the twist directly as expm(-i*(2a)*ZZ) with NO such factor of 2. To
    reproduce that exact unitary via qc.rzz(theta, ...), theta must satisfy
    theta/2 = 2a, i.e. theta = 4a = k/N. Using theta = 2a (i.e. k/(2N)), as
    an earlier version of this function did, silently halves the rotation
    angle in Qiskit's convention and was caught by validate_circuit()
    failing against exact diagonalization by a consistent ~4x factor in S2
    (since entanglement grows quadratically with angle for a near-product
    state, a 2x angle error showed up as a ~4x entropy error)."""
    qc = QuantumCircuit(N)
    j = N / 2.0

    for i in range(N):
        qc.ry(p, i)

    a = (k / (2.0 * j)) / 4.0
    angle = 4.0 * a  # Qiskit RZZ(angle) = exp(-i*(angle/2)*ZZ) = exp(-i*(2a)*ZZ) = k/N
    for i in range(N):
        for jq in range(i + 1, N):
            qc.rzz(angle, i, jq)
    # Global phase term from floquet_gate_sequence omitted: an overall
    # e^{i*phase} on the whole state doesn't affect reduced density
    # matrices, purity, or entropy.
    return qc


def initial_state_circuit(N):
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.ry(np.pi / 4, i)
    return qc


def renyi2_from_purity(purity):
    if purity < 1e-6 or purity > 1.0 + 1e-8:
        print(f"    WARNING: purity={purity:.6f} outside expected (0,1] "
              f"range -- numerical issue, inspect this point.")
    purity = min(max(purity, 1e-12), 1.0)
    return -np.log(purity)


def simulate_scaling_mps(N, k, p, steps, bond_dim=None,
                         method='matrix_product_state'):
    print(f"  -> Simulating N={N}, k={k}, method={method}, "
          f"bond_dim={bond_dim}...")
    sim_kwargs = dict(method=method)
    if method == 'matrix_product_state' and bond_dim is not None:
        sim_kwargs['matrix_product_state_max_bond_dimension'] = bond_dim
    simulator = AerSimulator(**sim_kwargs)

    s2_values = []
    subsystem_A = list(range(N // 2))

    qc_running = initial_state_circuit(N)
    step_circ = build_qkt_floquet_step(N, k, p)

    for t in range(1, steps + 1):
        qc_running.compose(step_circ, inplace=True)

        qc = qc_running.copy()
        qc.save_density_matrix(subsystem_A)

        result = simulator.run(qc).result()
        rho_a = np.asarray(result.data()['density_matrix'])

        purity = np.real(np.einsum('ij,ji->', rho_a, rho_a))
        s2_values.append(renyi2_from_purity(purity))

    return s2_values


def renyi2_exact(N, k, p, steps):
    """Exact statevector Renyi-2 entropy via the repo's own
    floquet_U_exact -- the same machinery every other figure is built from."""
    psi0 = coherent_product_state(N, theta=np.pi / 4)
    U_F = floquet_U_exact(N, k, p)
    nA = N // 2
    dimA, dimB = 2 ** nA, 2 ** (N - nA)

    s2_values = []
    psi = psi0.copy()
    for t in range(1, steps + 1):
        psi = U_F @ psi
        psi = psi / np.linalg.norm(psi)
        M = psi.reshape(dimA, dimB)
        rhoA = M @ M.conj().T
        purity = np.real(np.trace(rhoA @ rhoA))
        s2_values.append(renyi2_from_purity(purity))
    return s2_values


def validate_circuit(N, k, p, steps, tol=1e-6):
    """Compare the Qiskit circuit (exact 'statevector', no MPS truncation)
    against exact diagonalization. If these don't match, the circuit is
    still wrong and the MPS numbers downstream cannot be trusted."""
    s2_circuit = simulate_scaling_mps(N, k, p, steps, bond_dim=None,
                                      method='statevector')
    s2_exact = renyi2_exact(N, k, p, steps)
    max_diff = float(np.max(np.abs(np.array(s2_circuit) - np.array(s2_exact))))
    ok = max_diff < tol
    print(f"  [VALIDATE] N={N} k={k}: max|S2_circuit - S2_exact| = "
          f"{max_diff:.2e}  {'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"    circuit: {np.round(s2_circuit, 4)}")
        print(f"    exact  : {np.round(s2_exact, 4)}")
    return ok


if __name__ == "__main__":
    p_val = np.pi / 2
    val_steps = 8

    print("=" * 64)
    print("STEP 1: validating the circuit against exact diagonalization "
          "(N=6, N=8)")
    print("=" * 64)
    all_ok = True
    for N_val in (6, 8):
        for k_val in (0.5, 2.5):
            ok = validate_circuit(N_val, k_val, p_val, val_steps)
            all_ok = all_ok and ok

    if not all_ok:
        print("\nABORTING: the circuit does not reproduce exact "
              "diagonalization. Do NOT trust any MPS output below until "
              "this is fixed.")
        sys.exit(1)
    print("\nValidation passed: the circuit matches exact diagonalization "
          "to high precision. Proceeding to MPS runs.\n")

    print("=" * 64)
    print("STEP 2: bond-dimension convergence check (chi=128 vs chi=256) "
          "at N=12, chaotic (k=2.5)")
    print("=" * 64)
    conv_steps = 10
    s2_128 = simulate_scaling_mps(12, 2.5, p_val, conv_steps, bond_dim=128)
    s2_256 = simulate_scaling_mps(12, 2.5, p_val, conv_steps, bond_dim=256)
    rel_diff = np.abs(np.array(s2_128) - np.array(s2_256)) / \
               np.maximum(np.abs(s2_256), 1e-12)
    max_rel = float(np.max(rel_diff)) * 100.0
    print(f"  max relative difference (chi=128 vs chi=256): {max_rel:.2f}%")
    print("  [VERIFY] If this is NOT below 5%, chi=256 has not converged "
          "-- raise bond_dim further before trusting the manuscript's "
          "chi_MPS=256 convergence claim, and update the Fig. 14 caption "
          "with the actual number.")

    print("\n" + "=" * 64)
    print("STEP 3: full N=12, N=16 finite-size scaling at chi=256")
    print("=" * 64)
    N_list = [12, 16]
    k_vals = [0.5, 2.5]
    max_steps = 15
    BOND_DIM = 256

    results = {}
    for N in N_list:
        for k in k_vals:
            s2 = simulate_scaling_mps(N, k, p_val, max_steps, bond_dim=BOND_DIM)
            results[(N, k)] = s2
            print(f"    N={N} k={k}: S2 = {np.round(s2, 3)}")

    plt.figure(figsize=(9, 6))
    colors = {0.5: '#1f77b4', 2.5: '#ff7f0e'}
    styles = {12: '--', 16: '-'}
    markers = {12: 'o', 16: 's'}

    for (N, k), s2 in results.items():
        regime = "Regular" if k == 0.5 else "Chaotic"
        label = f"N={N}, {regime} (k={k})"
        plt.plot(range(1, max_steps + 1), s2,
                 color=colors[k], linestyle=styles[N], marker=markers[N],
                 linewidth=2, markersize=6, label=label)

    plt.xlabel('Floquet Step $t$', fontsize=12)
    plt.ylabel('Bipartite Renyi-2 Entropy $S_2$', fontsize=12)
    plt.title(f'Finite-Size Scaling of Entanglement Entropy '
              f'(MPS, $\\chi={BOND_DIM}$)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/finite_size_scaling_mps.pdf'
    plt.savefig(save_path, dpi=300)
    plt.savefig('figures/finite_size_scaling_mps.png', dpi=300)
    print(f"\nSuccess! Plot saved to {save_path} (and .png)")
    print(f"\n[VERIFY] chi=128 vs chi=256 max relative difference was "
          f"{max_rel:.2f}% -- put this number in the Fig. 14 caption "
          f"(replacing the unverified '<5%' claim) and confirm the chaotic "
          f"curves do not show a late-time DROP (a sign of residual "
          f"truncation/numerical error). If they do, raise bond_dim and "
          f"rerun before trusting this figure.")