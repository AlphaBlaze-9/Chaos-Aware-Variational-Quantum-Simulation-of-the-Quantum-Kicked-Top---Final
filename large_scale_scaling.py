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



EPSILON      = 0.05
# NOTE on convention: this script expands depth WHILE r2 > EPSILON, i.e. it
# keeps adding layers until the normalized residual drops below 0.05 -- a
# "converge well" criterion. This is the opposite sense from the
# manuscript's adaptive trigger (Sec. II.E.1), which expands depth only
# once r2 EXCEEDS a high failure threshold (varepsilon_trig=0.85) and
# otherwise tolerates a fairly large residual. The two are not the same
# definition of "D_max" -- this script's D_max answers "how deep must the
# circuit be for the residual to nearly vanish," while the manuscript's
# adaptive integrator answers "how deep before the residual stops being
# clearly too large." Keep this distinction in mind when comparing numbers
# from this script against Fig. 4/Fig. 8 of the manuscript.
DT           = 0.30
N_STEPS      = 10
K_CHAOTIC    = 2.5
K_REGULAR    = 0.5
SEED         = 0
N_SETTLE     = 8
DT_SETTLE    = 0.30

# Depth ceiling, expressed as N + CEILING_OFFSET. The original ceiling of
# N+6 caused the chaotic-regime points at N=8 and N=10 to saturate before
# the McLachlan integrator could converge, producing lower bounds rather
# than true D_max values (see referee point #3). Raising the offset gives
# the chaotic trajectories more room to actually converge. This is still a
# finite ceiling, not an unbounded one, both because circuit depth must
# stay finite for the comparison to mean anything and because the
# Tikhonov-regularized linear solve becomes increasingly expensive and
# numerically delicate at very large depth.
CEILING_OFFSET = 15




def _mclachlan(ansatz, theta, psi0, H):
    A, C, psi_th, _ = mclachlan_AC(ansatz, theta, psi0, H)
    H_sq  = float(np.real(np.vdot(psi_th, H @ H @ psi_th)))
    ridge = adaptive_ridge(A)
    return A, C, H_sq, ridge, psi_th


def _r2_and_td(A, C, H_sq, ridge):
    td = solve_thetadot(A, C, ridge)
    raw_residual = float(np.dot(td, A @ td) - 2.0 * np.dot(td, C) + H_sq)
    # BUG FIX: this previously returned raw_residual directly, without
    # dividing by H_sq = <psi|H_eff^2|psi>. The manuscript's normalized
    # residual r^2 (Eq. 11) and vqs.py's own mclachlan_residual_sq() both
    # divide by this quantity so that r^2 is bounded in [0,1]; without it,
    # this function was returning an unnormalized number that generically
    # sits well above EPSILON=0.05 regardless of k, which is why the
    # regular (k=0.5) and chaotic (k=2.5) regimes were behaving
    # identically and both saturating the depth ceiling in the D_max-vs-N
    # plot -- the trigger was firing almost every step in both regimes,
    # not just the chaotic one. Dividing by H_sq restores the actual
    # bounded [0,1] residual that the rest of the threshold logic
    # (EPSILON=0.05) was designed around.
    r2 = raw_residual / (float(H_sq) + 1e-15)
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
                     n_settle=N_SETTLE, dt_settle=DT_SETTLE,
                     ceiling_offset=CEILING_OFFSET):
    max_depth = N + ceiling_offset

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

    # Fixed: this used to hardcode "N + 6" independently of max_depth, so
    # if the ceiling formula above ever changed, this check would silently
    # go stale and misreport whether the depth ceiling was actually hit.
    # It now always reflects the ceiling that was actually used this run.
    ceiling_hit = dmax >= max_depth
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
    ax.set_title(
        rf"$D_{{\max}}$ vs. System Size (depth ceiling $= N+{CEILING_OFFSET}$)")
    ax.set_xticks(system_sizes)
    ax.text(0.55, 0.12,
            "open markers: depth ceiling reached (lower bound, not\n"
            "a converged $D_{\\max}$)",
            transform=ax.transAxes, fontsize=7.5, color="gray")
    ax.grid(True, ls=":")
    ax.legend()
    fig.tight_layout()
    # The manuscript's Fig. 12 includes figures/depth_scaling.png, so write
    # that name (this is the figure the .tex actually shows). The old
    # exact_dmax_scaling.* name is kept as an alias for backward compat.
    for base in ("figures/depth_scaling", "figures/exact_dmax_scaling"):
        fig.savefig(base + ".pdf")
        fig.savefig(base + ".png", dpi=300)
    plt.close(fig)

    print("\nSaved: figures/depth_scaling.{pdf,png} (and exact_dmax_scaling.*)")
    print("Regular D_max :", dict(zip(system_sizes, dmax_regular)))
    print("Chaotic D_max :", dict(zip(system_sizes, dmax_chaotic)))
    print("Regular ceiling hit:", dict(zip(system_sizes, ceiling_reg)))
    print("Chaotic ceiling hit:", dict(zip(system_sizes, ceiling_cha)))
    if any(ceiling_reg) or any(ceiling_cha):
        print(
            "\nWARNING: at least one point still hit the depth ceiling "
            f"(N + {CEILING_OFFSET}) even after raising it. Any such point "
            "is a lower bound on D_max, not a converged value, and must be "
            "reported/plotted as such -- do not present it as a "
            "demonstrated D_max in the manuscript."
        )


if __name__ == "__main__":
    main()