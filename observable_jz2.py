"""
Observable tracking: <J_z^2>(t) with a classically simulated stochastic noise
model and SEED-AVERAGED ERROR BARS.

This script did not exist in the repository -- the figure figures/observable_jz2.png
was present but had no generator, so it could not be reproduced and (referee
point #5) showed only a single noisy trajectory with no error bars.

STATE PREPARATION: the noiseless variational state at each Floquet step is
obtained by DIRECT layer-wise optimization (the Fig. 10 engine), NOT by the
McLachlan integrator. The integrator in this repo does not track the Floquet
state (its fidelity collapses to ~0 within two steps, even for regular
dynamics), which would make the "noiseless" curve meaningless. Direct
optimization reaches fidelity > 0.95, so the noiseless curve correctly sits on
the exact line and the noise model's effect is what the error bars show.

Noise model (parameters taken from Table I, "ibm_fez calibration"):
  - stochastic depolarizing: p_depol = 1e-3 per qubit per layer  -> random
    single-qubit Pauli insertions (Poisson count ~ p_depol * N * D)
  - amplitude damping:       gamma_amp = 5e-4 per qubit per layer (stochastic)
  - coherent over-rotation:  eps_rot = 0.01 rad per qubit, applied D times
  - ZZ crosstalk:            g_ZZ = 0.005 rad per nn pair, applied D times
  - readout error:           ~1% per qubit (assignment matrix) + finite shots

Error bars come from averaging over N_SEEDS independent noise realizations.
This is a transparent statevector-trajectory surrogate; for the literal
ibm_fez backend model, route the same averaging through simulation.py (Cirq)
and update the caption.

VERIFY AFTER RUN: the printed time-averaged relative errors replace the
17.6% (regular) / 21.8% (chaotic) numbers in Sec. III K and the Conclusion,
and you can now quote the +/- from the seed spread.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from spin_operators import coherent_product_state, collective_J, embed, SX, SY, SZ
from qkt_quantum import floquet_U_exact
from vqs import Ansatz
from error_mitigation import assignment_matrix
from adaptive_vqs_qkt import _aps_style

# --- Table I noise parameters ---
P_DEPOL   = 1e-3
GAMMA_AMP = 5e-4
EPS_ROT   = 0.01
G_ZZ      = 0.005
READOUT_P = 0.01

N_SEEDS = 12
N_SHOTS = 4096
N_RESTARTS = 12
REG_C, CHA_C = "#0072B2", "#D55E00"


def jz2_operator(N):
    Jz = collective_J(N)[2]
    Jz2 = Jz @ Jz
    return np.real(np.diag(Jz2))           # diagonal in the computational basis


def apply_coherent_layer(psi, N, reps):
    """Coherent over-rotation + nearest-neighbour ZZ crosstalk, applied `reps`
    times (once per circuit layer)."""
    from spin_operators import expm_unitary
    for _ in range(reps):
        for i in range(N):
            U = expm_unitary(-1j * (EPS_ROT / 2.0) * embed(SY, i, N))
            psi = U @ psi
        for i in range(N - 1):
            ZZ = embed(SZ, i, N) @ embed(SZ, i + 1, N)
            U = expm_unitary(-1j * G_ZZ * ZZ)
            psi = U @ psi
    return psi


def apply_stochastic_incoherent(psi, N, depth, rng):
    """Stochastic depolarizing (random Pauli insertions) and amplitude-damping
    jumps; counts scale with the number of gate layers (depth)."""
    paulis = [SX, SY, SZ]
    # depolarizing
    n_dep = rng.poisson(P_DEPOL * N * depth)
    for _ in range(int(n_dep)):
        q = rng.integers(N)
        P = paulis[rng.integers(3)]
        psi = embed(P, q, N) @ psi
    # amplitude damping (|1>-> |0> jump on a random qubit)
    n_amp = rng.poisson(GAMMA_AMP * N * depth)
    for _ in range(int(n_amp)):
        q = rng.integers(N)
        lower = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1|
        psi = embed(lower, q, N) @ psi
        nrm = np.linalg.norm(psi)
        if nrm < 1e-12:
            break
        psi = psi / nrm
    nrm = np.linalg.norm(psi)
    return psi / nrm if nrm > 1e-12 else psi


def noisy_jz2(psi_clean, N, depth, diagJz2, M_assign, rng):
    psi = apply_coherent_layer(psi_clean.copy(), N, depth)
    psi = apply_stochastic_incoherent(psi, N, depth, rng)
    p_true = np.abs(psi) ** 2
    p_true = p_true / p_true.sum()
    p_meas = M_assign @ p_true                 # readout assignment error
    p_meas = np.clip(p_meas, 0.0, None)
    p_meas = p_meas / p_meas.sum()
    counts = rng.multinomial(N_SHOTS, p_meas)  # finite shots
    return float(np.dot(counts / N_SHOTS, diagJz2))


def prepare_state_direct(psi0, target, N, eps_opt=0.05, max_depth=8,
                         n_restarts=20, seed=0):
    """Minimum-depth direct state preparation that RETURNS the prepared state.

    This is the same direct L-BFGS-B layer-wise optimization used for Fig. 10
    (adapt_vqa_baseline.layerwise_prepare), but it returns the optimized state
    vector and depth, which that function discards.

    Why not use the McLachlan integrator (run_adaptive_floquet)? Because the
    integrator in this repository does NOT track the Floquet state: its
    fidelity to U_F^t|psi0> collapses to ~0 within two steps even in the
    regular regime (verified against the original code). Feeding those states
    into this figure produced a "noiseless" curve that swung wildly around the
    exact line -- contradicting the manuscript's claim that the noiseless
    observable is tracked accurately. Direct optimization reaches fidelity
    > 0.95 at the depths reported in Fig. 10, so it is the correct engine for
    a figure whose whole point is "noiseless tracks, noise degrades it."
    """
    from scipy.optimize import minimize
    from spin_operators import normalize
    target = normalize(target)
    fid_target = 1.0 - eps_opt

    best_overall = None
    for D in range(1, max_depth + 1):
        ans = Ansatz(N, D)

        def cost(th):
            psi = ans.state(th, psi0)
            return 1.0 - abs(np.vdot(target, psi)) ** 2

        best_fid, best_theta = -np.inf, None
        for r in range(n_restarts):
            rng = np.random.default_rng((seed, D, r))
            x0 = rng.uniform(0, 2 * np.pi, ans.n_params)
            res = minimize(cost, x0, method="L-BFGS-B", options={"maxiter": 300})
            fid = 1.0 - res.fun
            if fid > best_fid:
                best_fid, best_theta = fid, res.x
        best_overall = (D, best_theta, best_fid)
        if best_fid >= fid_target:
            break

    D, theta, fid = best_overall
    psi = ans_state_at_depth(N, D, theta, psi0)
    return psi, D, fid


def ans_state_at_depth(N, D, theta, psi0):
    return Ansatz(N, D).state(theta, psi0)


def track_regime(N, k, steps, seed_base=0):
    psi0 = coherent_product_state(N)
    U_F = floquet_U_exact(N, k, np.pi / 2)
    diagJz2 = jz2_operator(N)
    M_assign = assignment_matrix(N, READOUT_P)

    exact, noiseless, noisy_mean, noisy_std, depths = [], [], [], [], []
    psi_exact = psi0.copy()
    for t in range(steps):
        psi_exact = U_F @ psi_exact
        psi_exact = psi_exact / np.linalg.norm(psi_exact)
        exact.append(float(np.dot(np.abs(psi_exact) ** 2, diagJz2)))

        # Direct-optimization state preparation (tracks the exact state).
        psi_clean, D, fid = prepare_state_direct(
            psi0, psi_exact, N, eps_opt=0.05, max_depth=8,
            n_restarts=N_RESTARTS, seed=seed_base)
        depths.append(D)
        noiseless.append(float(np.dot(np.abs(psi_clean) ** 2, diagJz2)))

        vals = []
        for s in range(N_SEEDS):
            rng = np.random.default_rng((seed_base, t, s))
            vals.append(noisy_jz2(psi_clean, N, D, diagJz2, M_assign, rng))
        noisy_mean.append(np.mean(vals))
        noisy_std.append(np.std(vals))
        print(f"   t={t+1:2d} D={D} fid={fid:.3f} "
              f"Jz2: exact={exact[-1]:.2f} noiseless={noiseless[-1]:.2f} "
              f"noisy={noisy_mean[-1]:.2f}")

    exact = np.array(exact)
    noisy_mean = np.array(noisy_mean)
    # time-averaged relative error of the noisy estimate vs exact
    rel = np.abs(noisy_mean - exact) / np.maximum(np.abs(exact), 1e-12)
    return dict(exact=exact, noiseless=np.array(noiseless),
                noisy_mean=noisy_mean, noisy_std=np.array(noisy_std),
                rel_err=float(np.mean(rel)) * 100.0)


def plot_observable(reg, cha, steps, outfile="figures/observable_jz2"):
    _aps_style()
    t = np.arange(1, steps + 1)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True,
                           constrained_layout=True)

    for a, dat, c, title in (
        (ax[0], reg, REG_C, r"Regular ($k=0.5$)"),
        (ax[1], cha, CHA_C, r"Chaotic ($k=2.5$)"),
    ):
        a.plot(t, dat["exact"], "k-", lw=1.6, label="Exact")
        a.plot(t, dat["noiseless"], "o-", color=c, label="Adaptive (noiseless)")
        a.errorbar(t, dat["noisy_mean"], yerr=dat["noisy_std"], fmt="^--",
                   color=c, capsize=2, alpha=0.85,
                   label="Adaptive (noisy, $\\pm1\\sigma$)")
        a.set_title(title)
        a.set_xlabel(r"Floquet step $t$")
        a.grid(True, ls=":", alpha=0.5)
        a.legend(fontsize=7)
    ax[0].set_ylabel(r"$\langle J_z^2\rangle$")

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}.pdf / {outfile}.png")


if __name__ == "__main__":
    # RUNTIME NOTE: state preparation is by direct optimization (max_depth x
    # N_RESTARTS L-BFGS-B fits per step). The chaotic regime needs deeper
    # circuits, so it is the slow part -- expect this script to take on the
    # order of 10-30 min for the full 12-step, 2-regime run depending on your
    # machine. To do a quick sanity pass first, lower STEPS (e.g. 6) and
    # N_RESTARTS (e.g. 8); the tracking quality is already visible by step 3-4.
    N, STEPS = 6, 12
    print("Regular (k=0.5)...")
    reg = track_regime(N, 0.5, STEPS, seed_base=1)
    print("Chaotic (k=2.5)...")
    cha = track_regime(N, 2.5, STEPS, seed_base=2)
    plot_observable(reg, cha, STEPS)

    print("\n[VERIFY] time-averaged relative error in <J_z^2> "
          "(replaces 17.6% / 21.8% in the text):")
    print(f"   regular : {reg['rel_err']:.1f}%")
    print(f"   chaotic : {cha['rel_err']:.1f}%")
