

import numpy as np
from functools import reduce
from spin_operators import (
    collective_J, embed, SZ, expm_unitary, spin_coherent_state, normalize,
)





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


if __name__ == "__main__":
    
    for N in (4, 6):
        for k in (0.5, 2.5):
            err = trotter_error(N, k, np.pi / 2, n_trotter=1)
            print(f"N={N} k={k}: ||U_trotter(n=1) - U_exact|| = {err:.2e}")
