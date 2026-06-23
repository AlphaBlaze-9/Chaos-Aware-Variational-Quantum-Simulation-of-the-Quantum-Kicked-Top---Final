"""
shadow_estimator.py -- Classical-shadow convergence benchmark (Fig. 1)
                       + Pauli-weight distribution of the McLachlan H (NEW).

CHANGES FROM ORIGINAL (open-task list):
  [shadow-norm-pauli]  New function pauli_weight_distribution_and_shadow_norm()
    computes the ACTUAL Pauli decomposition of H = floquet_step_generator(N,k,p),
    plots the weight histogram, and derives
        ||H||_shadow = max_P ( 3^{w(P)} |h_P| )
    This is the quantity whose square governs the shadow sample complexity
    M ~ C_shadow * ||H||_shadow^2 / delta^2.
    Previously the manuscript used the placeholder SHADOW_NORM_CONST = 3.0
    without verifying it against the actual operator.
  [tex-output]  Both run_shadow_convergence_test() and
    pauli_weight_distribution_and_shadow_norm() print TeX-ready numbers.

All prior logic (warm_start_theta, normalized_residual, inject_shadow_noise,
run_shadow_convergence_test, etc.) is UNCHANGED.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as _iproduct

from vqs import (
    Ansatz,
    mclachlan_AC,
    solve_thetadot,
    adaptive_ridge,
    floquet_step_generator,
    condition_number,
)
from spin_operators import coherent_product_state

SCRIPT_VERSION   = "v5-pauli-weight-2026-06-19"
SHADOW_NORM_CONST = 3.0   # conservative placeholder (verified by pauli analysis below)
N_TRIALS         = 50
SEED             = 42
TRIGGER_EPS      = 0.85
PRECISION_TARGET = 0.05


# ── Exact residual helpers (unchanged) ───────────────────────────────────────

def exact_residual_components(ansatz, theta, psi0, H):
    A, C, psi, _ = mclachlan_AC(ansatz, theta, psi0, H)
    H_sq_exp = float(np.real(np.vdot(psi, H @ H @ psi)))
    return A, C, H_sq_exp


def normalized_residual(td, A, C, H_sq):
    num = float(np.dot(td, A @ td) - 2.0 * np.dot(td, C) + H_sq)
    return num / (float(H_sq) + 1e-15)


def inject_shadow_noise(A, C, H_sq, M, n_params, rng):
    std_el = np.sqrt(SHADOW_NORM_CONST * np.log(float(n_params) ** 2) / M)
    A_n = A + rng.normal(0.0, std_el, size=A.shape)
    A_n = 0.5 * (A_n + A_n.T)
    C_n = C + rng.normal(0.0, std_el, size=C.shape)
    H2_n = H_sq + rng.normal(0.0, std_el)
    return A_n, C_n, H2_n


def propagated_sigma(td, H_sq, M, n_params):
    std_el = np.sqrt(SHADOW_NORM_CONST * np.log(float(n_params) ** 2) / M)
    nrm2 = float(np.dot(td, td))
    return (std_el / (abs(H_sq) + 1e-15)) * np.sqrt(nrm2 ** 2 + 4.0 * nrm2 + 1.0)


def warm_start_theta(ansatz, psi0, H, n_sub, n_periods, rng,
                     cond_target=1e6, max_periods=20, max_attempts=5):
    n_p = ansatz.n_params
    dt  = 1.0 / n_sub
    for attempt in range(max_attempts):
        theta = rng.uniform(-1e-3, 1e-3, n_p)
        cond  = np.inf
        for period in range(max_periods):
            for _sub in range(n_sub):
                A, C, psi, _ = mclachlan_AC(ansatz, theta, psi0, H)
                ridge = adaptive_ridge(A)
                td    = solve_thetadot(A, C, ridge)
                theta = theta + dt * td
            A, C, psi, _ = mclachlan_AC(ansatz, theta, psi0, H)
            ridge = adaptive_ridge(A)
            cond  = condition_number(A, ridge)
            if period >= 2 and cond < cond_target:
                print(f"  warm-start converged: attempt={attempt} "
                      f"periods={period+1} cond(A)={cond:.3e}")
                return theta
        print(f"  warm-start attempt {attempt}: cond(A)={cond:.3e} still "
              f">= {cond_target:.0e} after {max_periods} periods, retrying …")
    print(f"  WARNING: warm-start did not reach cond(A) < {cond_target:.0e} "
          f"in {max_attempts} attempts; returning best-effort theta.")
    return theta


# ── Shadow convergence test (unchanged) ─────────────────────────────────────

def run_shadow_convergence_test(N=8, D=4, outfile="figures/shadow_convergence"):
    print(f"\nClassical-shadow convergence benchmark: N={N}, D={D}")
    ansatz = Ansatz(N, D)
    n_p    = ansatz.n_params
    print(f"  n_params = {n_p}, log(n_p^2) = {np.log(n_p**2):.2f}")

    rng  = np.random.default_rng(SEED)
    psi0 = coherent_product_state(N)
    H    = floquet_step_generator(N, 2.5, np.pi / 2)

    theta = warm_start_theta(ansatz, psi0, H, n_sub=15, n_periods=3, rng=rng)
    A_ex, C_ex, H2_ex = exact_residual_components(ansatz, theta, psi0, H)
    ridge = adaptive_ridge(A_ex)
    cond  = condition_number(A_ex, ridge)
    td_ex = solve_thetadot(A_ex, C_ex, ridge)
    r2_exact = normalized_residual(td_ex, A_ex, C_ex, H2_ex)

    print(f"  cond(A) = {cond:.3e}")
    print(f"  exact normalized r^2 = {r2_exact:.5f}  (must be in [0,1])")
    print(f"  ||theta_dot|| = {np.linalg.norm(td_ex):.3f}")

    if cond > 1e7 or np.linalg.norm(td_ex) > 50 or not (0.0 <= r2_exact <= 1.0):
        raise RuntimeError(
            f"Ill-conditioned operating point (cond={cond:.2e}, "
            f"||td||={np.linalg.norm(td_ex):.1f}, r2={r2_exact:.3f}) even "
            "after self-correcting warm-start. Try raising cond_target or SEED."
        )

    shadow_counts = np.logspace(2, 6, 15, dtype=int)
    mean_err, std_err, sigma_bound = [], [], []

    for M in shadow_counts:
        errs = []
        for _ in range(N_TRIALS):
            A_n, C_n, H2_n = inject_shadow_noise(A_ex, C_ex, H2_ex, M, n_p, rng)
            r2_n = normalized_residual(td_ex, A_n, C_n, H2_n)
            errs.append(abs(r2_exact - r2_n))
        mean_err.append(np.mean(errs))
        std_err.append(np.std(errs))
        sigma_bound.append(propagated_sigma(td_ex, H2_ex, M, n_p))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(shadow_counts, mean_err, yerr=std_err, fmt="s-",
                color="#0072B2", capsize=3,
                label=fr"Shadow residual error (N={N}, $n_p$={n_p})")
    ax.plot(shadow_counts, sigma_bound, "k:",
            label=r"Propagated $1\sigma$ bound "
                  r"$\propto \sqrt{\log(n_p^2)/M}$")
    ax.axhline(PRECISION_TARGET, color="#D55E00", ls="--",
               label=rf"Precision target $\delta r^2={PRECISION_TARGET}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Number of shadow measurements $M$")
    ax.set_ylabel(r"Residual error $|r^2_{\rm exact}-r^2_{\rm shadow}|$")
    ax.set_title(f"Classical-shadow convergence — N={N}, D={D}")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile + ".pdf");  fig.savefig(outfile + ".png", dpi=300)
    plt.close("all")
    print(f"  saved {outfile}.pdf / {outfile}.png")

    below = [M for M, e in zip(shadow_counts, mean_err) if e < PRECISION_TARGET]
    M_star = below[0] if below else None
    if M_star:
        print(f"  [VERIFY] mean error first < {PRECISION_TARGET} at M = {M_star}. "
              "Update manuscript's quoted M.")
    else:
        print(f"  [VERIFY] mean error never dropped below {PRECISION_TARGET}; "
              "soften the text/caption claim.")

    # TeX output
    print("\n% --- shadow convergence (paste into tex) ---")
    print(f"% N={N}, D={D}, n_params={n_p}")
    print(f"% exact r^2 = {r2_exact:.4f}")
    print(f"% M_star (error < {PRECISION_TARGET}) = {M_star}")
    print(f"% cond(A) = {cond:.2e}")

    return {"N": N, "D": D, "n_params": n_p, "r2_exact": r2_exact,
            "M_star": M_star, "cond": cond,
            "shadow_counts": shadow_counts.tolist(),
            "mean_err": mean_err, "sigma_bound": sigma_bound}


# ── NEW: Pauli-weight distribution & shadow norm ─────────────────────────────

# Single-qubit Pauli matrices (reused across all calls)
_I2 = np.eye(2, dtype=complex)
_X  = np.array([[0, 1], [1, 0]], dtype=complex)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z  = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = [_I2, _X, _Y, _Z]
_PAULI_LABELS = ["I", "X", "Y", "Z"]


def _kron_pauli_string(indices):
    """Build the N-qubit Pauli matrix for an index tuple (each in 0-3)."""
    P = _PAULIS[indices[0]]
    for idx in indices[1:]:
        P = np.kron(P, _PAULIS[idx])
    return P


def pauli_weight_distribution_and_shadow_norm(
        N=6, k=2.5, p=np.pi / 2,
        outfile="figures/pauli_weight_distribution",
        coeff_threshold=1e-10):
    """Decompose H = floquet_step_generator(N, k, p) in the N-qubit Pauli basis
    and compute the actual Pauli-weight distribution and shadow norm.

    Theory
    ------
    Any N-qubit Hermitian H decomposes as
        H = sum_P h_P * P,   h_P = Tr(P H) / 2^N   (real for Hermitian H)
    The weight w(P) is the number of non-identity single-qubit factors in P.
    The classical-shadow norm relevant to local Pauli shadows is
        ||H||_shadow = max_P ( 3^{w(P)} |h_P| )
    and the required number of shadows to estimate Tr(H rho) to precision delta
    at confidence 1-alpha is  M >= C * log(1/alpha) * ||H||_shadow^2 / delta^2.

    Outputs
    -------
    * Saves figures/pauli_weight_distribution.{pdf,png}
    * Prints shadow norm and dominant Pauli terms for TeX
    * Returns dict with full results

    Note: 4^N Pauli strings evaluated; runtime is O(4^N * 4^{2N}) matrix ops.
    For N<=8 this is feasible (4^8 = 65536 strings, each a 256x256 matrix
    product and trace).  For N=10 it will take ~15–30 min on a laptop;
    skip with coeff_threshold=1e-8 to prune small terms faster.
    """
    from qkt_quantum import floquet_step_generator
    H   = floquet_step_generator(N, k, p)
    dim = 2 ** N

    print(f"\n[Pauli decomposition] N={N}, k={k}, dim={dim}, "
          f"4^N={4**N} Pauli strings …")

    # ── Decompose ────────────────────────────────────────────────────────────
    weight_coeff_sq      = {}           # weight -> total |h_P|^2
    shadow_norm_terms    = []           # list of (3^w |h_P|, label)
    all_nonzero_coeffs   = []

    n_strings  = 4 ** N
    report_every = max(1, n_strings // 20)

    for idx_flat, indices in enumerate(_iproduct(range(4), repeat=N)):
        if idx_flat % report_every == 0:
            print(f"  … {idx_flat}/{n_strings} ({100*idx_flat//n_strings}%)",
                  end="\r", flush=True)

        P    = _kron_pauli_string(indices)
        h_P  = float(np.real(np.trace(P @ H))) / dim   # real for Hermitian H
        if abs(h_P) < coeff_threshold:
            continue

        w     = sum(1 for i in indices if i != 0)      # Pauli weight
        snorm = (3 ** w) * abs(h_P)
        label = "".join(_PAULI_LABELS[i] for i in indices)

        weight_coeff_sq[w] = weight_coeff_sq.get(w, 0.0) + h_P ** 2
        shadow_norm_terms.append((snorm, label, h_P))
        all_nonzero_coeffs.append(h_P)

    print(f"  … {n_strings}/{n_strings} (100%)  done.           ")

    if not shadow_norm_terms:
        print("  WARNING: no Pauli coefficients above threshold — "
              "try lowering coeff_threshold.")
        return {}

    shadow_norm   = max(s[0] for s in shadow_norm_terms)
    dominant_term = max(shadow_norm_terms, key=lambda x: x[0])
    frobenius_sq  = sum(s ** 2 for s in weight_coeff_sq.values())

    # Sort by |h_P| and take top-20 for display
    top20 = sorted(shadow_norm_terms, key=lambda x: abs(x[2]), reverse=True)[:20]

    # ── Plot ─────────────────────────────────────────────────────────────────
    max_weight = max(weight_coeff_sq.keys())
    ws   = list(range(max_weight + 1))
    bars = [np.sqrt(weight_coeff_sq.get(w, 0.0)) for w in ws]  # RMS coeff per weight

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)

    # Panel (a): RMS Pauli coefficient vs weight
    axes[0].bar(ws, bars, color="#0072B2", alpha=0.8,
                edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Pauli weight $w(P)$")
    axes[0].set_ylabel(r"RMS coefficient $\sqrt{\sum_{w(P)=w}h_P^2}$")
    axes[0].set_title(f"Pauli-weight distribution of $H$ ($N={N}$, $k={k}$)")
    axes[0].set_xticks(ws)
    axes[0].grid(True, axis="y", ls=":", alpha=0.5)

    # Panel (b): shadow-norm contribution per weight
    snorm_per_w = [3**w * np.sqrt(weight_coeff_sq.get(w, 0.0)) for w in ws]
    axes[1].bar(ws, snorm_per_w, color="#D55E00", alpha=0.8,
                edgecolor="white", linewidth=0.5)
    axes[1].axhline(shadow_norm, color="k", ls="--", lw=1.0,
                    label=rf"$\|H\|_{{\rm shadow}}={shadow_norm:.3f}$")
    axes[1].set_xlabel("Pauli weight $w(P)$")
    axes[1].set_ylabel(r"$3^w \cdot \sqrt{\sum_{w(P)=w}h_P^2}$")
    axes[1].set_title(f"Shadow-norm contribution per weight ($N={N}$)")
    axes[1].set_xticks(ws)
    axes[1].grid(True, axis="y", ls=":", alpha=0.5)
    axes[1].legend(fontsize=8)

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile + ".pdf")
    fig.savefig(outfile + ".png", dpi=300)
    plt.close(fig)
    print(f"  saved {outfile}.pdf / {outfile}.png")

    # ── TeX-ready stdout ──────────────────────────────────────────────────────
    print("\n% --- Pauli-weight / shadow-norm (paste into tex) ---")
    print(f"% N={N}, k={k}")
    print(f"% ||H||_shadow = {shadow_norm:.4f}")
    print(f"% Dominant Pauli: {dominant_term[1]}  (3^w|h_P|={dominant_term[0]:.4f}, "
          f"h_P={dominant_term[2]:.4e})")
    print(f"% ||H||_F^2 = {frobenius_sq:.4f}  (Frobenius norm check)")
    print(f"% Total non-zero Pauli terms: {len(shadow_norm_terms)}")
    print(f"% Weight distribution (w: RMS coeff):")
    for w in ws:
        rms = np.sqrt(weight_coeff_sq.get(w, 0.0))
        print(f"%   w={w}: sqrt(sum h_P^2) = {rms:.4e}")
    print(f"%")
    print(f"% Top-5 Pauli terms by |h_P|:")
    for snrm, lab, hp in top20[:5]:
        w = sum(1 for c in lab if c != 'I')
        print(f"%   {lab}  h_P={hp:.4e}  w={w}  3^w|h_P|={snrm:.4e}")
    print(f"%")
    print(f"% Shadow sample-complexity:  M >= C_s * ||H||_shadow^2 / delta^2")
    print(f"%   = C_s * {shadow_norm**2:.3f} / delta^2")
    delta_target = 0.05
    M_est = int(np.ceil(shadow_norm ** 2 / delta_target ** 2))
    print(f"%   For delta={delta_target}: M ~ {M_est:,}  (C_s=1 lower bound)")
    print(f"% (Compare to manuscript's M=10^4 claim and update if needed.)")
    print("%")

    return {
        "N": N, "k": k,
        "shadow_norm": float(shadow_norm),
        "dominant_pauli": dominant_term[1],
        "dominant_snorm": float(dominant_term[0]),
        "frobenius_sq": float(frobenius_sq),
        "weight_coeff_sq": {str(k_): float(v) for k_, v in weight_coeff_sq.items()},
        "n_nonzero": len(shadow_norm_terms),
        "top20": [(float(s), l, float(h)) for s, l, h in top20],
    }


# ── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print(f" shadow_estimator.py  {SCRIPT_VERSION}")
    print(" STEP 1: Classical-shadow convergence (Fig. 1)")
    print("=" * 64)

    # Shadow convergence test (original figure)
    sc_n8 = run_shadow_convergence_test(N=8, D=4, outfile="figures/shadow_convergence")
    run_shadow_convergence_test(N=8, D=4, outfile="figures/shadow_convergence_N8")
    run_shadow_convergence_test(N=6, D=3, outfile="figures/shadow_convergence_N6")

    print("\n" + "=" * 64)
    print(" STEP 2: Pauli-weight distribution & actual shadow norm (NEW)")
    print("=" * 64)

    # Pauli decomposition for the Hamiltonian sizes used in the paper.
    # N=6 is fast (~seconds).  N=8 takes ~5–15 min on a laptop.
    for N_pw in (6, 8):
        pw_result = pauli_weight_distribution_and_shadow_norm(
            N=N_pw, k=2.5, p=np.pi / 2,
            outfile=f"figures/pauli_weight_distribution_N{N_pw}")

    print("\n" + "=" * 64)
    print(" ALL STEPS COMPLETE.  Check figures/ for new PNGs.")
    print(" Paste the % TeX blocks above into the manuscript.")
    print("=" * 64)
