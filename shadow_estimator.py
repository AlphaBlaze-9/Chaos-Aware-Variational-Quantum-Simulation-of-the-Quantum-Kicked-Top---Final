import os
import numpy as np
import matplotlib.pyplot as plt

from vqs import (
    Ansatz,
    mclachlan_AC,
    solve_thetadot,
    adaptive_ridge,
    floquet_step_generator,
    condition_number,
)
from spin_operators import coherent_product_state


SCRIPT_VERSION = "v4-selfcorrecting-warmstart-2026-06-19"

SHADOW_NORM_CONST = 3.0     # shadow-norm constant for local Pauli observables
N_TRIALS          = 50
SEED              = 42
TRIGGER_EPS       = 0.85    # eps_trig (context line only)
PRECISION_TARGET  = 0.05    # delta r^2 the figure must reach


def exact_residual_components(ansatz, theta, psi0, H):
    """Return (A, C, <H^2>) for the exact (noise-free) metric/force/denominator."""
    A, C, psi, _ = mclachlan_AC(ansatz, theta, psi0, H)
    H_sq_exp = float(np.real(np.vdot(psi, H @ H @ psi)))
    return A, C, H_sq_exp


def normalized_residual(td, A, C, H_sq):
    """Normalized McLachlan residual r^2 = (td^T A td - 2 td^T C + <H^2>)/<H^2>.

    THIS NORMALIZATION (division by <H^2> = ||iH|psi>||^2) is the bug that
    was missing before. Without it the quantity plotted was the UNNORMALIZED
    numerator, which is not bounded in [0,1] and runs to O(10)-O(100) -- the
    reason the old figure's y-axis reached ~10^3 and never approached the
    0.05 line in its own caption. With it, r^2 is bounded in [0,1] exactly as
    in Eq. (14)/(15) of the manuscript and in vqs.mclachlan_residual_sq().
    """
    num = float(np.dot(td, A @ td) - 2.0 * np.dot(td, C) + H_sq)
    return num / (float(H_sq) + 1e-15)


def inject_shadow_noise(A, C, H_sq, M, n_params, rng):
    """Per-element Gaussian shadow noise, std = sqrt(C_shadow * log(n_p^2)/M)."""
    std_el = np.sqrt(SHADOW_NORM_CONST * np.log(float(n_params) ** 2) / M)
    A_n = A + rng.normal(0.0, std_el, size=A.shape)
    A_n = 0.5 * (A_n + A_n.T)                       # keep it symmetric
    C_n = C + rng.normal(0.0, std_el, size=C.shape)
    H2_n = H_sq + rng.normal(0.0, std_el)
    return A_n, C_n, H2_n


def propagated_sigma(td, H_sq, M, n_params):
    """Analytic 1-sigma propagation of per-element shadow noise into r^2,
    holding td fixed: delta r^2 ~ (std_el/<H^2>) * sqrt(||td||^4 + 4||td||^2 + 1)."""
    std_el = np.sqrt(SHADOW_NORM_CONST * np.log(float(n_params) ** 2) / M)
    nrm2 = float(np.dot(td, td))
    return (std_el / (abs(H_sq) + 1e-15)) * np.sqrt(nrm2 ** 2 + 4.0 * nrm2 + 1.0)


def warm_start_theta(ansatz, psi0, H, n_sub, n_periods, rng,
                     cond_target=1e6, max_periods=20, max_attempts=5):
    """Reach a genuine, WELL-CONDITIONED McLachlan operating point by
    actually integrating forward Floquet sub-periods from a near-zero start,
    instead of sampling a fresh random theta at full depth D.

    This matters because a cold theta~U(0,0.1) at D=4 puts the ansatz at a
    point where the metric tensor A is numerically close to singular (the
    circuit has barely begun to explore parameter space at that depth). The
    adaptive ridge then has to inflate all the way to its kappa_max=1e8
    ceiling to keep A invertible, and that ill-conditioning is what blows up
    ||theta_dot|| (and with it, the noise-propagated error) to unphysical
    values -- NOT genuine shadow-measurement statistics.

    A FIXED period count is not robust: which point you land on after N
    periods depends on the exact floating-point path (BLAS/LAPACK backend,
    numpy/scipy version), so the same seed can converge cleanly on one
    machine and land on an ill-conditioned point on another, as happened
    when this was hardcoded to exactly 3 periods. This version instead
    integrates one period at a time and CHECKS the condition number after
    each one, continuing until it is comfortably below cond_target (default
    1e6, well under the 1e8 ridge ceiling) or max_periods is reached. If it
    still hasn't found a good point, it retries from a fresh random start
    (new sub-seed) up to max_attempts times. This makes the operating point
    robust to platform-level numerical differences instead of depending on
    a lucky fixed trajectory length.
    """
    n_p = ansatz.n_params
    dt = 1.0 / n_sub

    for attempt in range(max_attempts):
        theta = rng.uniform(-1e-3, 1e-3, n_p)
        cond = np.inf
        for period in range(max_periods):
            for _sub in range(n_sub):
                A, C, psi, _ = mclachlan_AC(ansatz, theta, psi0, H)
                ridge = adaptive_ridge(A)
                td = solve_thetadot(A, C, ridge)
                theta = theta + dt * td
            A, C, psi, _ = mclachlan_AC(ansatz, theta, psi0, H)
            ridge = adaptive_ridge(A)
            cond = condition_number(A, ridge)
            if period >= 2 and cond < cond_target:
                print(f"  warm-start converged: attempt={attempt} "
                      f"periods={period+1} cond(A)={cond:.3e}")
                return theta
        print(f"  warm-start attempt {attempt}: cond(A)={cond:.3e} still "
              f">= {cond_target:.0e} after {max_periods} periods, retrying "
              f"with a fresh start...")

    # Exhausted all attempts; return the last theta anyway -- the caller's
    # guard (cond > 1e7 check) will catch it and fail loudly rather than
    # silently plot a broken figure.
    print(f"  WARNING: warm-start did not reach cond(A) < {cond_target:.0e} "
          f"in {max_attempts} attempts; returning best-effort theta.")
    return theta


def run_shadow_convergence_test(N=8, D=4, outfile="figures/shadow_convergence"):
    print(f"\nClassical-shadow convergence benchmark: N={N}, D={D}")
    ansatz = Ansatz(N, D)
    n_p = ansatz.n_params
    print(f"  n_params = {n_p},  log(n_p^2) = {np.log(n_p**2):.2f}")

    rng = np.random.default_rng(SEED)

    psi0 = coherent_product_state(N)
    H = floquet_step_generator(N, 2.5, np.pi / 2)

    # Operating point: reached by actually evolving the ansatz forward
    # (n_sub=15 substeps per period), matching how theta would genuinely
    # arrive at this depth during the real algorithm -- not a cold random
    # draw at full depth. warm_start_theta checks the condition number after
    # every period and keeps integrating (or retries from a fresh start)
    # until it is genuinely well-conditioned, rather than trusting a fixed
    # period count -- the exact floating-point trajectory (and hence how
    # many periods are needed) can differ slightly across numpy/scipy/BLAS
    # versions and platforms.
    theta = warm_start_theta(ansatz, psi0, H, n_sub=15, n_periods=3, rng=rng)

    A_ex, C_ex, H2_ex = exact_residual_components(ansatz, theta, psi0, H)
    ridge = adaptive_ridge(A_ex)
    cond = condition_number(A_ex, ridge)

    # We hold theta-dot FIXED at its exact McLachlan value. The figure asks
    # "how precisely do M shadow snapshots resolve r^2 at a known operating
    # point" (the trigger question), which is exactly the residual evaluated
    # at the current theta-dot. Re-solving theta-dot from each noisy A is a
    # different (ill-conditioning-dominated) question.
    td_ex = solve_thetadot(A_ex, C_ex, ridge)
    r2_exact = normalized_residual(td_ex, A_ex, C_ex, H2_ex)
    print(f"  cond(A) = {cond:.3e}  (sanity: should be well below the 1e8 ceiling)")
    print(f"  exact normalized r^2 = {r2_exact:.5f}  (must be in [0,1])")
    print(f"  ||theta_dot|| = {np.linalg.norm(td_ex):.3f}")

    # Guard against the old failure mode: a cold/ill-conditioned operating
    # point pins the ridge at kappa_max and blows up ||theta_dot||, producing
    # a plot whose y-axis runs to 10^3-10^4 and never reaches 0.05. This
    # should no longer trigger now that warm_start_theta self-corrects, but
    # it is kept as a final safety net -- if it DOES fire, it means even the
    # retry logic couldn't find a well-conditioned point on this machine,
    # which would itself be worth reporting.
    if cond > 1e7 or np.linalg.norm(td_ex) > 50 or not (0.0 <= r2_exact <= 1.0):
        raise RuntimeError(
            f"Ill-conditioned operating point (cond={cond:.2e}, "
            f"||td||={np.linalg.norm(td_ex):.1f}, r2={r2_exact:.3f}) even "
            "after the self-correcting warm-start. This should be rare -- "
            "please report it. As a workaround, try raising cond_target or "
            "max_periods in warm_start_theta(), or changing SEED."
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

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of shadow measurements $M$")
    ax.set_ylabel(r"Residual error $|r^2_{\rm exact}-r^2_{\rm shadow}|$")
    ax.set_title(f"Classical-shadow convergence — N={N}, D={D}")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile + ".pdf")
    fig.savefig(outfile + ".png", dpi=300)
    plt.close("all")
    print(f"  saved {outfile}.pdf / {outfile}.png")

    # Report the M at which the mean error first dips below the target.
    below = [M for M, e in zip(shadow_counts, mean_err) if e < PRECISION_TARGET]
    if below:
        print(f"  [VERIFY] mean error first < {PRECISION_TARGET} at M = {below[0]}. "
              f"Update the manuscript's quoted M (abstract/Sec. II.E.2/conclusion) "
              f"to match this value -- do not assume it is 1e4.")
    else:
        print(f"  [VERIFY] mean error never dropped below {PRECISION_TARGET} over "
              f"M in [1e2,1e6]; SOFTEN the text/caption claim accordingly.")


if __name__ == "__main__":
    print("=" * 64)
    print(f"  shadow_estimator.py  {SCRIPT_VERSION}")
    print("  If you do NOT see this banner, you are running an OLD copy.")
    print("=" * 64)
    # N=8, D=4 (n_p=68) is the case shown as Fig. 1 -> default tex filename.
    run_shadow_convergence_test(N=8, D=4, outfile="figures/shadow_convergence")
    # keep the explicit-N variants too
    run_shadow_convergence_test(N=8, D=4, outfile="figures/shadow_convergence_N8")
    run_shadow_convergence_test(N=6, D=3, outfile="figures/shadow_convergence_N6")