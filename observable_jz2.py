"""
Observable tracking: <J_z^2>(t) for the adaptive VQS, with a classically
simulated stochastic noise model and SEED-AVERAGED ERROR BARS.

This script did not exist in the repository -- the figure figures/observable_jz2.png
was present but had no generator, so it could not be reproduced and (referee
point #5) showed only a single noisy trajectory with no error bars.

Noise model (parameters taken from Table I, "ibm_fez calibration"):
  - stochastic depolarizing: p_depol = 1e-3 per qubit per layer  -> random
    single-qubit Pauli insertions (Poisson count ~ p_depol * N * D)
  - amplitude damping:       gamma_amp = 5e-4 per qubit per layer (stochastic)
  - coherent over-rotation:  eps_rot = 0.01 rad per qubit, applied D times
  - ZZ crosstalk:            g_ZZ = 0.005 rad per nn pair, applied D times
  - readout error:           ~1% per qubit (assignment matrix) + finite shots

Error bars come from averaging over N_SEEDS independent noise realizations
(different depolarizing/amp-damping draws and different shot samples). This is
a transparent statevector-trajectory surrogate; if you want the figure to
carry the *literal* ibm_fez backend model, run the same averaging through the
Cirq sandbox in simulation.py instead and update the caption accordingly.

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
from adaptive_vqs_qkt import run_adaptive_floquet, _aps_style

# --- Table I noise parameters ---
P_DEPOL   = 1e-3
GAMMA_AMP = 5e-4
EPS_ROT   = 0.01
G_ZZ      = 0.005
READOUT_P = 0.01

N_SEEDS = 20
N_SHOTS = 4096
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


def track_regime(N, k, steps, seed_base=0):
    out = run_adaptive_floquet(N, k=k, steps=steps)
    depths = out["depth"]
    thetas = out["theta_history"]

    psi0 = coherent_product_state(N)
    U_F = floquet_U_exact(N, k, np.pi / 2)
    diagJz2 = jz2_operator(N)
    M_assign = assignment_matrix(N, READOUT_P)

    exact, noiseless, noisy_mean, noisy_std = [], [], [], []
    psi_exact = psi0.copy()
    for t in range(steps):
        psi_exact = U_F @ psi_exact
        psi_exact = psi_exact / np.linalg.norm(psi_exact)
        exact.append(float(np.dot(np.abs(psi_exact) ** 2, diagJz2)))

        ans = Ansatz(N, depths[t])
        psi_clean = ans.state(thetas[t], psi0)
        noiseless.append(float(np.dot(np.abs(psi_clean) ** 2, diagJz2)))

        vals = []
        for s in range(N_SEEDS):
            rng = np.random.default_rng((seed_base, t, s))
            vals.append(noisy_jz2(psi_clean, N, depths[t], diagJz2, M_assign, rng))
        noisy_mean.append(np.mean(vals))
        noisy_std.append(np.std(vals))

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
    N, STEPS = 6, 14
    print("Regular (k=0.5)...")
    reg = track_regime(N, 0.5, STEPS, seed_base=1)
    print("Chaotic (k=2.5)...")
    cha = track_regime(N, 2.5, STEPS, seed_base=2)
    plot_observable(reg, cha, STEPS)

    print("\n[VERIFY] time-averaged relative error in <J_z^2> "
          "(replaces 17.6% / 21.8% in the text):")
    print(f"   regular : {reg['rel_err']:.1f}%")
    print(f"   chaotic : {cha['rel_err']:.1f}%")
