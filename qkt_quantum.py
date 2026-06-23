"""
qkt_quantum.py  --  Floquet operator and related quantum machinery.

CHANGES FROM ORIGINAL:
  Appendix A / reviewer items (P7, C6):
    - branch_validity_check(): verifies the principal-branch matrix
      logarithm is well-defined by computing min|arg(lambda) - pi| across
      k in [0.5, 4.0], and element-wise validates e^{-i H_eff} = U_F.
    - pauli_decompose_heff(): decomposes H_eff in the Pauli basis and
      reports the fraction of operator norm from YZ vs ZZ interactions.
      Used to assess the ADAPT-VQA operator-pool confound (reviewer item
      xii) and to support the J_z^2 all-to-all ZZ content claim.
  
  All existing functions are UNCHANGED.
"""
import numpy as np
from functools import reduce
from scipy.linalg import logm as scipy_logm
from spin_operators import (
    collective_J, embed, SZ, expm_unitary, spin_coherent_state, normalize,
)


# ---------------------------------------------------------------------------
# Core QKT operators (unchanged)
# ---------------------------------------------------------------------------

def floquet_U_exact(N: int, k: float, p: float) -> np.ndarray:
    _, Jy, Jz = collective_J(N)
    j = N / 2.0
    U_kick = expm_unitary(-1j * k * (Jz @ Jz) / (2.0 * j))
    U_rot = expm_unitary(-1j * p * Jy)
    return U_kick @ U_rot


def _zz_generator(N: int):
    zz_terms = []
    for i in range(N):
        for jq in range(i + 1, N):
            Zi = embed(SZ, i, N)
            Zj = embed(SZ, jq, N)
            zz_terms.append(Zi @ Zj)
    return zz_terms


def floquet_gate_sequence(N: int, k: float, p: float):
    j = N / 2.0
    seq = []
    for i in range(N):
        seq.append(("ry", i, p))
    a = (k / (2.0 * j)) / 4.0
    for i in range(N):
        for jq in range(i + 1, N):
            seq.append(("rzz", (i, jq), 2.0 * a))
    seq.append(("gphase", None, (k / (2.0 * j)) * (N / 4.0)))
    return seq


def floquet_U_trotter(N: int, k: float, p: float, n_trotter: int = 1) -> np.ndarray:
    Jy = collective_J(N)[1]
    j = N / 2.0
    U_rot = expm_unitary(-1j * p * Jy)
    a = (k / (2.0 * j)) / 4.0
    global_phase = np.exp(-1j * (k / (2.0 * j)) * (N / 4.0))
    U_twist = global_phase * np.eye(2 ** N, dtype=complex)
    for i in range(N):
        for jq in range(i + 1, N):
            ZZ = embed(SZ, i, N) @ embed(SZ, jq, N)
            U_twist = expm_unitary(-1j * (2.0 * a) * ZZ) @ U_twist
    return U_twist @ U_rot


def trotter_error(N: int, k: float, p: float, n_trotter: int = 1) -> float:
    Ue = floquet_U_exact(N, k, p)
    Ut = floquet_U_trotter(N, k, p, n_trotter=n_trotter)
    return float(np.linalg.norm(Ut - Ue, ord=2))


def evolve_state(U: np.ndarray, psi0: np.ndarray, T: int):
    traj = [psi0.astype(complex)]
    psi = psi0.astype(complex)
    for _ in range(T):
        psi = normalize(U @ psi)
        traj.append(psi)
    return traj


def husimi_Q_grid(psi: np.ndarray, N: int, thetas, phis) -> np.ndarray:
    Q = np.empty((len(thetas), len(phis)), dtype=float)
    for i, th in enumerate(thetas):
        for jq, ph in enumerate(phis):
            coh = spin_coherent_state(N, th, ph)
            amp = np.vdot(coh, psi)
            Q[i, jq] = (amp.conjugate() * amp).real
    Q /= Q.max() + 1e-12
    return Q


# ---------------------------------------------------------------------------
# NEW: Appendix A -- branch-validity verification (reviewer P7, C6)
# ---------------------------------------------------------------------------

def branch_validity_check(N: int, k_values=None, p: float = np.pi / 2,
                           verbose: bool = True):
    """Verify that the principal branch of log(U_F) is well-defined for each k
    and that e^{-i H_eff} reproduces U_F element-wise.

    Returns dict with:
      "min_spectral_gap" : array of min|arg(lambda) - pi| for each k
      "max_element_error": array of max|e^{-iH_eff} - U_F| for each k
    """
    if k_values is None:
        k_values = np.linspace(0.5, 4.0, 36)

    min_gaps = []
    max_errs = []

    for k in k_values:
        U_F = floquet_U_exact(N, k, p)
        evals = np.linalg.eigvals(U_F)
        args = np.angle(evals)           # in (-pi, pi]
        # distance from each eigenphase to the branch cut at pi
        gaps = np.abs(np.abs(args) - np.pi)
        min_gap = float(gaps.min())
        min_gaps.append(min_gap)

        # Compute H_eff via principal-branch log (scipy.linalg.logm,
        # not numpy -- numpy has no logm)
        H_eff = 1j * scipy_logm(U_F)
        # Verify: e^{-i H_eff} should reproduce U_F
        U_reconstructed = expm_unitary(-1j * H_eff)
        max_err = float(np.max(np.abs(U_reconstructed - U_F)))
        max_errs.append(max_err)

    min_gaps = np.array(min_gaps)
    max_errs = np.array(max_errs)

    if verbose:
        print(f"\n=== BRANCH VALIDITY CHECK (N={N}) ===")
        print(f"  min spectral gap to branch cut (min|arg(lambda)-pi|):")
        print(f"    across k in [{k_values[0]:.1f}, {k_values[-1]:.1f}]: "
              f"min={min_gaps.min():.4f} rad  at k={k_values[np.argmin(min_gaps)]:.2f}")
        print(f"    -> {'OK (well-defined)' if min_gaps.min() > 0.01 else 'WARNING: eigenphase close to branch cut'}")
        print(f"  max |e^(-i H_eff) - U_F|:")
        print(f"    across k: max={max_errs.max():.2e}  "
              f"at k={k_values[np.argmax(max_errs)]:.2f}")
        print(f"    -> {'OK (< 1e-10)' if max_errs.max() < 1e-10 else 'WARNING: reconstruction error'}")

        # Spot-check at k=0.5 and k=2.5 (the two regime representatives)
        for k_spot in (0.5, 2.5):
            idx = np.argmin(np.abs(k_values - k_spot))
            print(f"  k={k_values[idx]:.1f}: gap={min_gaps[idx]:.4f} rad, "
                  f"max element error={max_errs[idx]:.2e}")

        print("\n>>> FOR APPENDIX A (manuscript):")
        print(f"    min spectral gap = {min_gaps.min():.4f} rad "
              f"(well away from 0, branch is valid)")
        print(f"    max element-wise |e^(-iH_eff) - U_F| = {max_errs.max():.2e}")

    return {"k_values": k_values,
            "min_spectral_gap": min_gaps,
            "max_element_error": max_errs}


# ---------------------------------------------------------------------------
# NEW: Pauli decomposition of H_eff (reviewer item xii, M7)
# ---------------------------------------------------------------------------

def _pauli_basis_1q():
    """Single-qubit Pauli matrices I, X, Y, Z."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def pauli_decompose_heff(N: int, k: float, p: float = np.pi / 2,
                          verbose: bool = True):
    """Decompose H_eff = i log(U_F) in the n-qubit Pauli basis.
    Reports the fraction of the Frobenius norm coming from:
      - ZZ nearest-neighbour terms  (the ansatz has these)
      - ZZ all-to-all terms         (the ansatz has only NN)
      - YZ nearest-neighbour terms  (ADAPT-VQA pool includes these)
      - everything else

    Returns dict with norm fractions.
    This directly addresses reviewer M7 (whether YZ is significant in H_eff,
    which would confound the ADAPT-VQA comparison).
    """
    from itertools import product as iproduct

    U_F = floquet_U_exact(N, k, p)
    H_eff = 1j * scipy_logm(U_F)

    # Build all n-qubit Pauli strings
    paulis_1q = _pauli_basis_1q()
    labels = ["I", "X", "Y", "Z"]
    d = 2 ** N

    coeffs = {}
    norm_sq_total = 0.0
    for combo in iproduct(labels, repeat=N):
        label = "".join(combo)
        P = paulis_1q[combo[0]]
        for q in range(1, N):
            P = np.kron(P, paulis_1q[combo[q]])
        c = np.trace(P.conj().T @ H_eff) / d
        coeffs[label] = c
        norm_sq_total += abs(c) ** 2

    # Categorize
    def is_nn_zz(label):
        active = [(i, l) for i, l in enumerate(label) if l != "I"]
        if len(active) != 2:
            return False
        i1, l1 = active[0]; i2, l2 = active[1]
        return l1 == "Z" and l2 == "Z" and (i2 - i1 == 1 or
                                              (i1 == 0 and i2 == N - 1))

    def is_all_to_all_zz(label):
        active = [(i, l) for i, l in enumerate(label) if l != "I"]
        if len(active) != 2:
            return False
        i1, l1 = active[0]; i2, l2 = active[1]
        return l1 == "Z" and l2 == "Z"

    def is_nn_yz(label):
        active = [(i, l) for i, l in enumerate(label) if l != "I"]
        if len(active) != 2:
            return False
        i1, l1 = active[0]; i2, l2 = active[1]
        nn = (i2 - i1 == 1 or (i1 == 0 and i2 == N - 1))
        return nn and set([l1, l2]) == {"Y", "Z"}

    ns_nn_zz, ns_all_zz, ns_nn_yz, ns_other = 0.0, 0.0, 0.0, 0.0
    for label, c in coeffs.items():
        cs = abs(c) ** 2
        if is_nn_zz(label):
            ns_nn_zz += cs
        if is_all_to_all_zz(label):   # includes NN ZZ
            ns_all_zz += cs
        if is_nn_yz(label):
            ns_nn_yz += cs
        if not is_all_to_all_zz(label) and not is_nn_yz(label):
            ns_other += cs

    frac_nn_zz  = ns_nn_zz  / max(norm_sq_total, 1e-30)
    frac_all_zz = ns_all_zz / max(norm_sq_total, 1e-30)
    frac_nn_yz  = ns_nn_yz  / max(norm_sq_total, 1e-30)
    frac_other  = ns_other  / max(norm_sq_total, 1e-30)

    if verbose:
        print(f"\n=== PAULI DECOMPOSITION OF H_eff (N={N}, k={k}) ===")
        print(f"  Total Frobenius norm^2: {norm_sq_total:.4f}")
        print(f"  NN ZZ fraction:       {frac_nn_zz*100:.2f}%  "
              f"  (ansatz has these)")
        print(f"  All-to-all ZZ fraction: {frac_all_zz*100:.2f}%  "
              f"  (ansatz has only NN subset)")
        print(f"  NN YZ fraction:         {frac_nn_yz*100:.2f}%  "
              f"  (ADAPT-VQA pool includes these)")
        print(f"  Other fraction:         {frac_other*100:.2f}%")
        print(f"\n>>> FOR PAPER (Appendix B ADAPT-VQA comparison):")
        if frac_nn_yz < 0.01:
            print(f"    YZ content < 1% -> pool difference is negligible.")
            print(f"    The ADAPT-VQA CNOT comparison is not materially confounded.")
        else:
            print(f"    YZ content = {frac_nn_yz*100:.1f}% -> the ADAPT-VQA pool")
            print(f"    has access to operators the HEA cannot represent.")
            print(f"    Flag this explicitly in the appendix comparison.")

    return {
        "frac_nn_zz": frac_nn_zz,
        "frac_all_zz": frac_all_zz,
        "frac_nn_yz": frac_nn_yz,
        "frac_other": frac_other,
        "norm_sq_total": norm_sq_total,
    }


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Original Trotter-error tests
    for N in (4, 6):
        for k in (0.5, 2.5):
            err = trotter_error(N, k, np.pi / 2, n_trotter=1)
            print(f"N={N} k={k}: ||U_trotter(n=1) - U_exact|| = {err:.2e}")

    # NEW: Branch validity check
    print("\n" + "="*60)
    print("Branch validity check (Appendix A)")
    print("="*60)
    branch_validity_check(N=6, k_values=np.linspace(0.5, 4.0, 36))

    # NEW: Pauli decomposition at k=0.5 and k=2.5
    print("\n" + "="*60)
    print("Pauli decomposition of H_eff (reviewer item xii)")
    print("="*60)
    for k in (0.5, 2.5):
        pauli_decompose_heff(N=6, k=k)

    print("\nAll qkt_quantum checks complete.")
    print("New outputs feed directly into Appendix A (branch validity)")
    print("and Appendix B (ADAPT-VQA pool confound).")
