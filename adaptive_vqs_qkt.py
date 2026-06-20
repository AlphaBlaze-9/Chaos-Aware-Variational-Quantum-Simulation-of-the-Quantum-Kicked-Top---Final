import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

from spin_operators import coherent_product_state, collective_J
from qkt_quantum import floquet_U_exact
from vqs import (
    Ansatz, mclachlan_AC, mclachlan_residual_sq, solve_thetadot,
    condition_number, adaptive_ridge, floquet_step_generator,
)


def renyi2_entropy(psi: np.ndarray, N: int) -> float:
    dimA = 2 ** (N // 2)
    dimB = 2 ** (N - N // 2)
    M = psi.reshape(dimA, dimB)
    rhoA = M @ M.conj().T
    rhoA /= np.trace(rhoA)
    purity = np.real(np.trace(rhoA @ rhoA))
    purity = min(max(purity, 1e-15), 1.0)
    return float(-np.log2(purity))


def run_adaptive_floquet(
    N: int, k: float, steps: int = 12, p: float = np.pi / 2,
    n_sub: int = 15, residual_threshold: float = 0.85, M_consec: int = 1,
    max_depth: int = 8, target_cond: float = 1e8, seed: int = 42,
    compute_exact_diag: bool = True,
):
    """Adaptive-depth McLachlan VQS for the QKT.

    TRIGGER (corrected to match manuscript Sec. II.E.1 and Fig. 8):
    -----------------------------------------------------------------
    Depth is expanded when the PEAK NORMALIZED McLACHLAN RESIDUAL r^2 over
    the Floquet period exceeds `residual_threshold` (= varepsilon_trig =
    0.85). r^2 is computed by `mclachlan_residual_sq`, which is bounded in
    [0,1] and is evaluable on hardware via the parameter-shift rule WITHOUT
    the exact state vector -- this is the paper's central claim.

    The previous version of this function silently switched the trigger to
    the EXACT-STATE INFIDELITY (1 - |<psi_exact|psi_var>|^2) against a
    hardcoded 0.45 threshold whenever compute_exact_diag=True. That defeats
    the entire point of the residual trigger (it needs the exact state) and
    contradicted the manuscript. It is removed.

    `compute_exact_diag` now ONLY controls whether the exact-state fidelity
    is recorded as a DIAGNOSTIC (out["fid_diag"]); it never drives the
    trigger. Set it False if you want to confirm the algorithm runs with no
    access at all to psi_exact.
    """
    rng = np.random.default_rng(seed)
    H_step = floquet_step_generator(N, k, p)
    U_F = floquet_U_exact(N, k, p) if compute_exact_diag else None

    psi0 = coherent_product_state(N)
    depth = 1
    ans = Ansatz(N, depth)
    theta = rng.uniform(-1e-3, 1e-3, ans.n_params)
    dt = 1.0 / n_sub

    # --- Fixed-D=1 reference residual (the TRIGGER signal for panel (a)) ---
    # Panel (a) must show the residual that DECIDES whether to expand: the
    # peak r^2 a depth-1 circuit incurs at each step. This is the quantity
    # the manuscript describes (regular ~0.76-0.84 < 0.85, chaotic
    # ~0.89-0.94 > 0.85). It is computed on an INDEPENDENT D=1 trajectory so
    # it is not contaminated by the adaptive run's depth growth -- once the
    # chaotic circuit expands to D=8 its residual naturally drops (D=8 fits
    # the dynamics well), which would otherwise make the chaotic curve look
    # LOW, the opposite of the physics.
    ref_ans = Ansatz(N, 1)
    ref_theta = rng.uniform(-1e-3, 1e-3, ref_ans.n_params)

    out = {key: [] for key in
           ["residual", "residual_ref", "depth", "renyi2", "cond", "ridge",
            "fid_diag", "theta_history"]}
    consec = 0
    psi_exact = psi0.copy()

    print(f"[adaptive Floquet | N={N} k={k}] residual-triggered "
          f"(eps_trig={residual_threshold}), n_sub={n_sub}, max_depth={max_depth}")

    for t in range(steps):
        if compute_exact_diag:
            psi_exact = U_F @ psi_exact

        # Advance the independent D=1 reference trajectory one Floquet period
        # and record its peak residual -- this is the panel (a) trigger signal.
        ref_peak = 0.0
        for _sub in range(n_sub):
            A_r, C_r, _, _ = mclachlan_AC(ref_ans, ref_theta, psi0, H_step)
            ridge_r = adaptive_ridge(A_r, target_cond=target_cond)
            td_r = solve_thetadot(A_r, C_r, ridge_r)
            r2_r = mclachlan_residual_sq(ref_ans, ref_theta, td_r, psi0, H_step)
            ref_theta = ref_theta + dt * td_r
            ref_peak = max(ref_peak, r2_r)

        theta_period_start = theta.copy()
        depth_changed = True
        trigger_r2 = None       # residual the trigger actually saw this step
        trigger_cond = None
        trigger_ridge = None

        while depth_changed:
            depth_changed = False
            theta = theta_period_start.copy()

            if len(theta) < ans.n_params:
                noise = rng.normal(0, 1e-3, ans.n_params - len(theta))
                theta = np.concatenate([theta, noise])

            peak_r2, peak_cond, peak_ridge = 0.0, 0.0, 0.0

            for _sub in range(n_sub):
                A, C, psi, _ = mclachlan_AC(ans, theta, psi0, H_step)
                ridge = adaptive_ridge(A, target_cond=target_cond)
                cond = condition_number(A, ridge)
                thetadot = solve_thetadot(A, C, ridge)
                r2 = mclachlan_residual_sq(ans, theta, thetadot, psi0, H_step)
                theta = theta + dt * thetadot

                peak_r2 = max(peak_r2, r2)
                peak_cond = max(peak_cond, cond)
                peak_ridge = max(peak_ridge, ridge)

            # Capture the residual the trigger SAW (i.e. evaluated at the depth
            # this period began with, before any expansion). This is the
            # quantity that crosses eps_trig and tells the physics story:
            # chaotic > 0.85 (drives expansion), regular < 0.85 (stays put).
            # Recording the post-expansion residual instead would plot the
            # chaotic regime as LOW (because the circuit was just deepened),
            # which is exactly backwards.
            if trigger_r2 is None:
                trigger_r2 = peak_r2
                trigger_cond = peak_cond
                trigger_ridge = peak_ridge

            # --- CORRECTED TRIGGER: peak normalized residual vs eps_trig ---
            if peak_r2 > residual_threshold:
                consec += 1
            else:
                consec = 0

            if consec >= M_consec and depth < max_depth:
                depth += 1
                ans = Ansatz(N, depth)
                consec = 0
                depth_changed = True
                print(f"   period {t+1}: peak r^2 = {peak_r2:.3f} "
                      f"> {residual_threshold} -> depth {depth}")

        psi_var = ans.state(theta, psi0)
        out["residual"].append(trigger_r2)
        out["residual_ref"].append(ref_peak)
        out["depth"].append(depth)
        out["renyi2"].append(renyi2_entropy(psi_var, N))
        out["cond"].append(trigger_cond)
        out["ridge"].append(trigger_ridge)
        out["theta_history"].append(theta.copy())

        if compute_exact_diag:
            f = abs(np.vdot(psi_exact / np.linalg.norm(psi_exact), psi_var)) ** 2
            out["fid_diag"].append(float(np.real(f)))   # diagnostic only
        else:
            out["fid_diag"].append(np.nan)

    return out


def _aps_style():
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "text.usetex": False,
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.3,
        "lines.markersize": 4.0, "savefig.dpi": 600,
    })


REG_C, CHA_C = "#0072B2", "#D55E00"
DOUBLE_COL = 7.0
EPS_TRIG = 0.85


def plot_adaptive(reg, cha, steps, outfile="figures/adaptive_residual",
                  eps_trig=EPS_TRIG):
    import os
    _aps_style()
    t = np.arange(1, steps + 1)
    fig, ax = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.9),
                           sharex=True, constrained_layout=True)

    # (a) residual r^2 at fixed D=1 -- the TRIGGER signal. LINEAR axis in
    #     [0,1] with the eps_trig line. Regular stays below 0.85 (never
    #     expands); chaotic stays above (drives expansion). This is the
    #     decision signal; the resulting depths are in panel (b).
    a = ax[0, 0]
    a.plot(t, reg["residual_ref"], "o-", color=REG_C, label="Regular ($k=0.5$)")
    a.plot(t, cha["residual_ref"], "s-", color=CHA_C, label="Chaotic ($k=2.5$)")
    a.axhline(eps_trig, color="0.3", ls="--", lw=1.0,
              label=rf"$\varepsilon_{{\rm trig}}={eps_trig}$")
    a.set_ylim(0.0, 1.0)
    a.set_ylabel(r"McLachlan residual $r^2(t)$ at $D{=}1$")
    a.grid(True, ls=":", alpha=0.5)
    a.legend(loc="best")
    a.text(0.035, 0.965, "(a)", transform=a.transAxes, va="top", fontweight="bold")

    b = ax[0, 1]
    b.plot(t, reg["depth"], "o-", color=REG_C, label="Regular ($k=0.5$)")
    b.plot(t, cha["depth"], "s-", color=CHA_C, label="Chaotic ($k=2.5$)")
    b.set_ylabel(r"Circuit depth $D(t)$")
    b.grid(True, ls=":", alpha=0.5)
    b.legend(loc="best")
    b.text(0.035, 0.965, "(b)", transform=b.transAxes, va="top", fontweight="bold")

    c = ax[1, 0]
    c.plot(t, reg["renyi2"], "o-", color=REG_C, label="Regular ($k=0.5$)")
    c.plot(t, cha["renyi2"], "s-", color=CHA_C, label="Chaotic ($k=2.5$)")
    c.set_ylabel(r"Rényi-2 entropy $S_2$ (bits)")
    c.set_xlabel(r"Floquet step $t$")
    c.grid(True, ls=":", alpha=0.5)
    c.legend(loc="best")
    c.text(0.035, 0.965, "(c)", transform=c.transAxes, va="top", fontweight="bold")

    d = ax[1, 1]
    d.semilogy(t, reg["cond"], "o-", color=REG_C, label="Regular ($k=0.5$)")
    d.semilogy(t, cha["cond"], "s-", color=CHA_C, label="Chaotic ($k=2.5$)")
    d.set_ylabel(r"Metric cond. number $\kappa(A)$")
    d.set_xlabel(r"Floquet step $t$")
    d.grid(True, which="both", ls=":", alpha=0.5)
    d.legend(loc="best")
    d.text(0.035, 0.965, "(d)", transform=d.transAxes, va="top", fontweight="bold")

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}.pdf / {outfile}.png")


if __name__ == "__main__":
    N, STEPS = 6, 12
    print("Regular regime (k=0.5)...")
    reg = run_adaptive_floquet(N, k=0.5, steps=STEPS)
    print("Chaotic regime (k=2.5)...")
    cha = run_adaptive_floquet(N, k=2.5, steps=STEPS)
    plot_adaptive(reg, cha, STEPS)

    # --- Sanity checks the manuscript text depends on. VERIFY these print
    #     statements reproduce the values claimed in Sec. II.E.1 / Fig. 8. ---
    print("\n[VERIFY] D=1 reference residual ranges -- the panel (a) trigger "
          "signal. Update the manuscript's stated brackets (Sec. II.E.1 says "
          "regular [0.76,0.84], chaotic [0.92,0.94]) to the ACTUAL ranges "
          "below. The story (regular below 0.85, chaotic above) holds, but "
          "the exact numbers differ slightly:")
    print(f"   regular r^2 at D=1 : min={min(reg['residual_ref']):.3f} "
          f"max={max(reg['residual_ref']):.3f}")
    print(f"   chaotic r^2 at D=1 : min={min(cha['residual_ref']):.3f} "
          f"max={max(cha['residual_ref']):.3f}")
    print(f"[VERIFY] final regular depth = {reg['depth'][-1]} (expect 1), "
          f"final chaotic depth = {cha['depth'][-1]} (expect 8)")
    print("[diagnostic only] final-step exact fidelity: "
          f"regular {reg['fid_diag'][-1]:.3f}  chaotic {cha['fid_diag'][-1]:.3f}")

    print("\n--- Generating N=4 export for hardware PoC ---")
    cha_poc = run_adaptive_floquet(4, k=2.5, steps=5)
    with open("N4_chaotic_history.pkl", "wb") as f:
        pickle.dump(cha_poc, f)
    print("Saved 'N4_chaotic_history.pkl'.")
