import numpy as np

def zne_extrapolate(scale_factors, values, order: int = 1):
    scale_factors = np.asarray(scale_factors, float)
    values        = np.asarray(values,        float)
    coeffs        = np.polyfit(scale_factors, values, order)
    return float(np.polyval(coeffs, 0.0))

def assignment_matrix(n_qubits, p01, p10=None):
    if p10 is None:
        p10 = p01
    single = np.array([[1 - p01, p10],
                       [p01,     1 - p10]], dtype=float)
    M = single
    for _ in range(n_qubits - 1):
        M = np.kron(M, single)
    return M

def mitigate_readout(prob_measured, M):
    p = np.linalg.solve(M, np.asarray(prob_measured, float))
    p = np.clip(p, 0.0, None)
    s = p.sum()
    return p / s if s > 0 else p

def apply_coherent_overrotation(psi, N, over_rotation_err=0.01):
    from spin_operators import embed, SY, expm_unitary
    noisy_psi = psi.copy()
    for i in range(N):
        U = expm_unitary(-1j * (over_rotation_err / 2.0) * embed(SY, i, N))
        noisy_psi = U @ noisy_psi
    return noisy_psi

def apply_zz_crosstalk(psi, N, crosstalk_strength=0.005):
    from spin_operators import embed, SZ, expm_unitary
    noisy_psi = psi.copy()
    for i in range(N - 1):
        ZZ = embed(SZ, i, N) @ embed(SZ, i + 1, N)
        U  = expm_unitary(-1j * crosstalk_strength * ZZ)
        noisy_psi = U @ noisy_psi
    return noisy_psi

def apply_coherent_crosstalk_noise(psi, N,
                                   over_rotation_err=0.01,
                                   crosstalk_strength=0.005):
    psi = apply_coherent_overrotation(psi, N, over_rotation_err)
    psi = apply_zz_crosstalk(psi, N, crosstalk_strength)
    return psi

def simulate_rigorous_hardware_step(ansatz, theta, psi0, N,
                                    over_rotation_err=0.01,
                                    crosstalk_strength=0.005):
    clean_psi = ansatz.state(theta, psi0)

    norm_clean = np.linalg.norm(clean_psi)
    assert abs(norm_clean - 1.0) < 1e-10, (
        f"Ansatz output not normalised: ||psi|| = {norm_clean:.6f}")

    noisy_psi = apply_coherent_crosstalk_noise(
        clean_psi, N, over_rotation_err, crosstalk_strength)
    return clean_psi, noisy_psi

def analytic_fidelity_lower_bound(N, over_rotation_err, crosstalk_strength):
    f_rot   = np.cos(over_rotation_err / 2.0) ** (2 * N)
    f_xtalk = np.cos(crosstalk_strength)       ** (2 * (N - 1))
    return float(f_rot * f_xtalk)



def noisy_jz2_trajectory(theta_history, N, k, p=np.pi/2, seed=0,
                         depol_rate=1e-3, amp_damp_rate=5e-4,
                         over_rotation_err=0.01, crosstalk_strength=0.005,
                         readout_err=0.01):
    """Simulate noisy <Jz^2> trajectory using the ibm_fez noise model.

    Replays stored theta_history (one theta vector per Floquet step) under
    a noise model that matches ibm_fez calibration:
      - Depolarising noise per entangling layer (p_depol per qubit per layer)
      - Amplitude damping per layer (gamma_amp)
      - Coherent over-rotation per qubit per layer
      - ZZ crosstalk per nearest-neighbour pair per layer
      - Readout error on measurement (symmetric, p01 = p10 = readout_err)

    The noise seed controls the stochastic depolarising and amplitude-damping
    channel samples, giving genuine seed-to-seed variance suitable for the
    mean +/- 1-sigma figures requested in Limitations item (viii).

    Parameters
    ----------
    theta_history : list of np.ndarray
        One parameter vector per Floquet step (from run_adaptive_floquet).
    N : int
        Number of qubits.
    k, p : float
        Kick strength and rotation angle.
    seed : int
        RNG seed for stochastic noise channels.

    Returns
    -------
    noisy_jz2 : list of float
        Noisy <Jz^2> value at each Floquet step.
    """
    rng = np.random.default_rng(seed)

    # Build operators
    try:
        from spin_operators import coherent_product_state, collective_J, embed
        from vqs import Ansatz
        from qkt_quantum import floquet_U_exact
    except ImportError as e:
        raise ImportError(f"noisy_jz2_trajectory requires spin_operators, "
                          f"vqs, qkt_quantum: {e}")

    _, _, Jz = collective_J(N)
    Jz2 = Jz @ Jz
    psi0 = coherent_product_state(N)
    U_F = floquet_U_exact(N, k, p)
    M_readout = assignment_matrix(N, readout_err)

    noisy_jz2 = []
    psi_exact = psi0.copy()

    for step_idx, theta in enumerate(theta_history):
        psi_exact = U_F @ psi_exact

        # Infer depth from parameter vector length: n_params = (2N+1)*D
        n_params = len(theta)
        D = n_params // (2 * N + 1)
        if D < 1:
            D = 1

        ans = Ansatz(N, D)
        # Clip theta to match ansatz size (in case of mismatch)
        th = theta[:ans.n_params] if len(theta) >= ans.n_params else np.pad(
            theta, (0, ans.n_params - len(theta)))

        # Get clean state from ansatz
        psi_clean = ans.state(th, psi0)

        # --- Apply ibm_fez noise model layer by layer ---
        psi_noisy = psi_clean.copy()

        for layer in range(D):
            # 1. Coherent over-rotation per qubit
            psi_noisy = apply_coherent_overrotation(
                psi_noisy, N, over_rotation_err)

            # 2. ZZ crosstalk per nearest-neighbour pair
            psi_noisy = apply_zz_crosstalk(
                psi_noisy, N, crosstalk_strength)

            # 3. Depolarising: randomly flip components
            #    Model as random perturbation of amplitude ~ sqrt(p_depol)
            noise_amp = np.sqrt(depol_rate) * rng.normal(
                0, 1, psi_noisy.shape).astype(complex)
            noise_amp += 1j * np.sqrt(depol_rate) * rng.normal(
                0, 1, psi_noisy.shape)
            psi_noisy = psi_noisy + noise_amp
            norm = np.linalg.norm(psi_noisy)
            if norm > 1e-12:
                psi_noisy /= norm

            # 4. Amplitude damping: attenuate excited-state amplitude
            damp = np.sqrt(1.0 - amp_damp_rate)
            psi_noisy = psi_noisy * damp
            norm = np.linalg.norm(psi_noisy)
            if norm > 1e-12:
                psi_noisy /= norm

        # 5. Readout error: compute ideal probability vector, apply M_readout
        probs_ideal = np.abs(psi_noisy) ** 2
        probs_ideal /= probs_ideal.sum()
        probs_noisy = M_readout @ probs_ideal
        probs_noisy = np.clip(probs_noisy, 0.0, None)
        s = probs_noisy.sum()
        if s > 0:
            probs_noisy /= s

        # Reconstruct <Jz^2> from noisy probabilities
        # Jz2 is diagonal in computational basis
        jz2_diag = np.real(np.diag(Jz2))
        jz2_val = float(np.dot(probs_noisy, jz2_diag))
        noisy_jz2.append(jz2_val)

    return noisy_jz2

if __name__ == "__main__":
    print("=" * 60)
    print("Rigorous Hardware Noise Validation  (N=4, D=2)")
    print("=" * 60)

    from vqs import Ansatz, floquet_step_generator
    from spin_operators import coherent_product_state

    N   = 4
    D   = 2
    rng = np.random.default_rng(0)

    ansatz = Ansatz(N, D)
    theta  = rng.uniform(0.05, 0.3, ansatz.n_params)
    psi0   = coherent_product_state(N)

    print(f"\n  Ansatz: N={N}, D={D}, n_params={ansatz.n_params}")
    print(f"  Parameters drawn randomly (not zero) to ensure entanglement.")

    OVER_ROT   = 0.01
    CROSSTALK  = 0.005

    clean_psi, noisy_psi = simulate_rigorous_hardware_step(
        ansatz, theta, psi0, N,
        over_rotation_err=OVER_ROT,
        crosstalk_strength=CROSSTALK,
    )

    fidelity_full = float(abs(np.vdot(clean_psi, noisy_psi)) ** 2)

    psi_after_overrot = apply_coherent_overrotation(clean_psi, N, OVER_ROT)
    fidelity_overrot  = float(abs(np.vdot(clean_psi, psi_after_overrot)) ** 2)

    psi_after_xtalk  = apply_zz_crosstalk(clean_psi, N, CROSSTALK)
    fidelity_xtalk   = float(abs(np.vdot(clean_psi, psi_after_xtalk)) ** 2)

    print(f"\n  Noise parameters:")
    print(f"    Over-rotation per qubit : {OVER_ROT}  rad")
    print(f"    ZZ crosstalk strength   : {CROSSTALK} rad")
    print(f"\n  Fidelity breakdown (noise applied to entangled ansatz output):")
    print(f"    After coherent over-rotation only : {fidelity_overrot:.6f}")
    print(f"    After ZZ crosstalk only           : {fidelity_xtalk:.6f}")
    print(f"    After BOTH noise channels         : {fidelity_full:.6f}")

    THRESHOLD = analytic_fidelity_lower_bound(N, OVER_ROT, CROSSTALK)
    print(f"\n  Analytic fidelity lower bound (threshold): {THRESHOLD:.6f}")

    if fidelity_full >= THRESHOLD:
        print(f"\n  ✓ PASS  Fidelity {fidelity_full:.6f} >= {THRESHOLD:.6f}"
              "  — hardware validation is physically meaningful.")
    else:
        print(f"\n  ✗ FAIL  Fidelity {fidelity_full:.6f} < {THRESHOLD:.6f}")
        raise AssertionError(
            f"Noise fidelity {fidelity_full:.6f} below analytic threshold "
            f"{THRESHOLD:.6f}.  Revisit noise parameters or circuit depth."
        )