

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
from spin_operators import coherent_product_state, normalize
from qkt_quantum import floquet_U_exact



EPSILON    = 0.05
DT         = 0.30
N_STEPS    = 10      
K_CHAOTIC  = 2.5
K_REGULAR  = 0.5
SEED       = 0
N_SETTLE   = 8       
DT_SETTLE  = 0.30




def _mclachlan(ansatz, theta, psi0, H):
    A, C, psi_th, _ = mclachlan_AC(ansatz, theta, psi0, H)
    H_sq  = float(np.real(np.vdot(psi_th, H @ H @ psi_th)))
    ridge = adaptive_ridge(A)
    return A, C, H_sq, ridge, psi_th


def _r2_and_td(A, C, H_sq, ridge):
    td = solve_thetadot(A, C, ridge)
    r2 = float(np.dot(td, A @ td) - 2.0 * np.dot(td, C) + H_sq)
    return r2, td


def _settle(ansatz, theta, psi0, H, n_steps, dt):
    for _ in range(n_steps):
        A, C, H_sq, ridge, _ = _mclachlan(ansatz, theta, psi0, H)
        td    = solve_thetadot(A, C, ridge)
        theta = theta + dt * td
    A, C, H_sq, ridge, _ = _mclachlan(ansatz, theta, psi0, H)
    r2, td = _r2_and_td(A, C, H_sq, ridge)
    return theta, r2, td


def run_adaptive_vqs(N, k, steps, dt=DT, epsilon=EPSILON,
                     n_settle=N_SETTLE, dt_settle=DT_SETTLE):
    max_depth = N + 6   

    np.random.seed(SEED)
    H_step = floquet_step_generator(N, k, np.pi / 2)
    U_F    = floquet_U_exact(N, k, np.pi / 2)
    psi0   = coherent_product_state(N)
    depth  = 1
    ansatz = Ansatz(N, depth)
    theta  = np.zeros(ansatz.n_params)
    depth_history = []

    for step in range(steps):
        A, C, H_sq, ridge, _ = _mclachlan(ansatz, theta, psi0, H_step)
        r2, td = _r2_and_td(A, C, H_sq, ridge)

        while r2 > epsilon and depth < max_depth:
            depth  += 1
            ansatz  = Ansatz(N, depth)
            theta   = np.concatenate(
                [theta, np.zeros(ansatz.n_params - len(theta))])
            print(f"   N={N}, k={k}, step {step+1}: r^2={r2:.4f} "
                  f"-> depth {depth}, settling...")
            theta, r2, td = _settle(
                ansatz, theta, psi0, H_step, n_settle, dt_settle)
            print(f"      after settling: r^2={r2:.4f}")

        depth_history.append(depth)

        A, C, H_sq, ridge, _ = _mclachlan(ansatz, theta, psi0, H_step)
        td    = solve_thetadot(A, C, ridge)
        theta = theta + dt * td
        psi0  = normalize(U_F @ psi0)

    dmax = max(depth_history)
    
    ceiling_hit = dmax >= (N + 6)
    return depth_history, dmax, ceiling_hit


def main():
    system_sizes = [4, 6, 8, 10]
    dmax_chaotic = []
    dmax_regular = []
    ceiling_cha  = []
    ceiling_reg  = []

    os.makedirs("figures", exist_ok=True)

    for N in system_sizes:
        print(f"\n{'='*50}")
        print(f"N={N}  |  chaotic (k={K_CHAOTIC})...")
        _, dmax, hit = run_adaptive_vqs(N, K_CHAOTIC, steps=N_STEPS)
        print(f"-> D_max={dmax}  ceiling_hit={hit}")
        dmax_chaotic.append(dmax)
        ceiling_cha.append(hit)

        print(f"N={N}  |  regular (k={K_REGULAR})...")
        _, dmax, hit = run_adaptive_vqs(N, K_REGULAR, steps=N_STEPS)
        print(f"-> D_max={dmax}  ceiling_hit={hit}")
        dmax_regular.append(dmax)
        ceiling_reg.append(hit)

    
    fig, ax = plt.subplots(figsize=(6, 4))

    
    ax.plot(system_sizes, dmax_regular, "o-", color="#0072B2",
            label="Regular (k=0.5)")
    for i, hit in enumerate(ceiling_reg):
        if hit:
            ax.plot(system_sizes[i], dmax_regular[i], "o",
                    color="#0072B2", mfc="none", ms=10)

    
    ax.plot(system_sizes, dmax_chaotic, "s-", color="#D55E00",
            label="Chaotic (k=2.5)")
    for i, hit in enumerate(ceiling_cha):
        if hit:
            ax.plot(system_sizes[i], dmax_chaotic[i], "s",
                    color="#D55E00", mfc="none", ms=10)

    ax.set_xlabel("System Size $N$ (qubits)")
    ax.set_ylabel(r"Maximum Adaptive Depth $D_{\max}$")
    ax.set_title(r"$D_{\max}$ vs. System Size (depth ceiling $= N+6$)")
    ax.set_xticks(system_sizes)
    ax.text(0.55, 0.12, "open markers: depth-limited (lower bound)",
            transform=ax.transAxes, fontsize=7.5, color="gray")
    ax.grid(True, ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/exact_dmax_scaling.pdf")
    fig.savefig("figures/exact_dmax_scaling.png", dpi=300)
    plt.close(fig)

    print("\nSaved: figures/exact_dmax_scaling.pdf")
    print("Regular D_max :", dict(zip(system_sizes, dmax_regular)))
    print("Chaotic D_max :", dict(zip(system_sizes, dmax_chaotic)))


if __name__ == "__main__":
    main()