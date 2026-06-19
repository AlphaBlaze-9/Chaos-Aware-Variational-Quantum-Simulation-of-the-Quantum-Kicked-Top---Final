

import os
import numpy as np
import matplotlib.pyplot as plt

from vqs import (
    Ansatz,
    mclachlan_AC,
    solve_thetadot,
    adaptive_ridge,
    floquet_step_generator,
)
from spin_operators import coherent_product_state


SHADOW_NORM_CONST = 3.0
N_TRIALS          = 50      
SEED              = 42
TRIGGER_EPS       = 0.85    




def get_exact_residual_components(ansatz, theta, psi0, H):
    
    A, C, psi, _ = mclachlan_AC(ansatz, theta, psi0, H)
    H_sq_exp = float(np.real(np.vdot(psi, H @ H @ psi)))
    return A, C, H_sq_exp


def simulate_shadow_estimation(A_exact, C_exact, H_sq_exact,
                               num_shadows, n_params, rng):
    
    var_per_element = SHADOW_NORM_CONST * np.log(float(n_params) ** 2)
    std_el = np.sqrt(var_per_element / num_shadows)

    A_noisy = A_exact + rng.normal(0.0, std_el, size=A_exact.shape)
    A_noisy = 0.5 * (A_noisy + A_noisy.T)          

    C_noisy  = C_exact  + rng.normal(0.0, std_el, size=C_exact.shape)
    H2_noisy = H_sq_exact + rng.normal(0.0, std_el)

    return A_noisy, C_noisy, H2_noisy


def theoretical_error_bound(n_params, num_shadows):
    
    sigma_el = np.sqrt(SHADOW_NORM_CONST * np.log(float(n_params) ** 2)
                       / num_shadows)
    return (float(n_params) ** 2 + float(n_params)) * sigma_el




def run_shadow_convergence_test(N=6, D=3):
    print(f"\nRunning Classical Shadow Convergence Benchmark for N={N}, D={D}...")

    ansatz = Ansatz(N, D)
    n_p    = ansatz.n_params
    print(f"  n_params = {n_p},  log(n_p^2) = {np.log(n_p**2):.2f}")

    
    rng = np.random.default_rng(SEED)

    
    theta = rng.uniform(0.0, 0.1, n_p)
    psi0  = coherent_product_state(N)
    H     = floquet_step_generator(N, 2.5, np.pi / 2)

    
    A_ex, C_ex, H2_ex = get_exact_residual_components(ansatz, theta, psi0, H)
    ridge = adaptive_ridge(A_ex)
    td_ex = solve_thetadot(A_ex, C_ex, ridge)
    r2_exact = float(np.dot(td_ex, A_ex @ td_ex)
                     - 2.0 * np.dot(td_ex, C_ex)
                     + H2_ex)
    print(f"  Exact McLachlan Residual r^2: {r2_exact:.5f}")

    
    shadow_counts = np.logspace(2, 6, 15, dtype=int)
    mean_errors, std_errors, theory_bounds = [], [], []

    for M in shadow_counts:
        trial_errs = []
        for _ in range(N_TRIALS):
            An, Cn, H2n = simulate_shadow_estimation(
                A_ex, C_ex, H2_ex, M, n_p, rng)
            td_n = solve_thetadot(An, Cn, ridge)
            r2_n = float(np.dot(td_n, An @ td_n)
                         - 2.0 * np.dot(td_n, Cn)
                         + H2n)
            trial_errs.append(abs(r2_exact - r2_n))

        mean_errors.append(np.mean(trial_errs))
        std_errors.append(np.std(trial_errs))
        theory_bounds.append(theoretical_error_bound(n_p, M))

    
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(shadow_counts, mean_errors, yerr=std_errors,
                fmt="s-", color="#0072B2", capsize=3,
                label=f"Shadow Error (N={N}, $n_p$={n_p})")

    ax.plot(shadow_counts, theory_bounds, "k:",
            label=r"Bound: $(n_p^2+n_p)\sqrt{3\ln(n_p^2)/M}$")

    ax.axhline(TRIGGER_EPS, color="#D55E00", linestyle="--",
               label=(r"Trigger $\epsilon_{trig}=0.85$"
                      r"  [naive: $\propto n_p^4\log(n_p)/\epsilon^2$;"
                      r"  shadow: $\propto\log(n_p^2)/\epsilon^2$]"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Shadow Measurements ($M$)")
    ax.set_ylabel(r"Residual Error $|r^2_{\rm exact} - r^2_{\rm shadow}|$")
    ax.set_title(f"Classical Shadow Convergence — N={N}, D={D}")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(fontsize=7.5)
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    filename = f"figures/shadow_convergence_N{N}.pdf"
    fig.savefig(filename)
    fig.savefig(filename.replace(".pdf", ".png"), dpi=300)
    plt.close("all")   
    print(f"  Saved convergence plot to {filename}")


if __name__ == "__main__":
    run_shadow_convergence_test(N=6, D=3)
    run_shadow_convergence_test(N=8, D=4)