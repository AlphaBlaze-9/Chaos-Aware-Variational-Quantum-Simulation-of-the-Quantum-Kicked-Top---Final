import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from spin_operators import collective_J, embed, SZ, SY, expm_unitary, normalize


def _ry(angle, Yi, I):
    return np.cos(angle / 2.0) * I - 1j * np.sin(angle / 2.0) * Yi


def _rz(angle, Zi, I):
    return np.cos(angle / 2.0) * I - 1j * np.sin(angle / 2.0) * Zi


@dataclass
class Ansatz:
    N: int
    depth: int
    ops: dict = field(default_factory=dict)

    def __post_init__(self):
        self.I = np.eye(2 ** self.N, dtype=complex)
        self.Y = [embed(SY, i, self.N) for i in range(self.N)]
        self.Z = [embed(SZ, i, self.N) for i in range(self.N)]
        self.Zdiag = [np.real(np.diag(self.Z[i])) for i in range(self.N)]
        self.ZZ = []
        self.ZZdiag = []
        for i in range(self.N):
            jq = (i + 1) % self.N
            ZZg = embed(SZ, i, self.N) @ embed(SZ, jq, self.N)
            self.ZZ.append(ZZg)
            self.ZZdiag.append(np.real(np.diag(ZZg)))

    @property
    def n_params(self):
        return self.depth * (2 * self.N + 1)

    def unitary(self, theta: np.ndarray) -> np.ndarray:
        U = self.I.copy()
        idx = 0
        for _ in range(self.depth):
            for i in range(self.N):
                U = _ry(theta[idx], self.Y[i], self.I) @ U
                idx += 1
            for i in range(self.N):
                U = _rz(theta[idx], self.Z[i], self.I) @ U
                idx += 1
            chi = theta[idx]
            idx += 1
            for ZZg in self.ZZ:
                U = expm_unitary(-1j * (chi / 2.0) * ZZg) @ U
        return U

    def state(self, theta: np.ndarray, psi0: np.ndarray) -> np.ndarray:
        psi = psi0.astype(complex).copy()
        idx = 0
        for _ in range(self.depth):
            for i in range(self.N):
                a = theta[idx]; idx += 1
                psi = (np.cos(a / 2.0) * psi) - 1j * np.sin(a / 2.0) * (self.Y[i] @ psi)
            for i in range(self.N):
                a = theta[idx]; idx += 1
                psi = np.exp(-1j * (a / 2.0) * self.Zdiag[i]) * psi
            chi = theta[idx]; idx += 1
            for zzd in self.ZZdiag:
                psi = np.exp(-1j * (chi / 2.0) * zzd) * psi
        return normalize(psi)


def _param_derivatives(ansatz: Ansatz, theta: np.ndarray, psi0: np.ndarray,
                        eps: float = 1e-6) -> List[np.ndarray]:
    derivs = []
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        dpsi = (ansatz.state(tp, psi0) - ansatz.state(tm, psi0)) / (2 * eps)
        derivs.append(dpsi)
    return derivs


def mclachlan_AC(ansatz: Ansatz, theta: np.ndarray, psi0: np.ndarray,
                  H: np.ndarray):
    psi = ansatz.state(theta, psi0)
    d = _param_derivatives(ansatz, theta, psi0)
    n = len(theta)
    A = np.zeros((n, n))
    C = np.zeros(n)
    Hpsi = H @ psi
    for i in range(n):
        for jj in range(n):
            A[i, jj] = np.real(np.vdot(d[i], d[jj]))
        C[i] = np.real(np.vdot(d[i], -1j * Hpsi))
    return A, C, psi, d


def mclachlan_residual_sq(ansatz: Ansatz, theta: np.ndarray, thetadot: np.ndarray,
                           psi0: np.ndarray, H: np.ndarray) -> float:
    psi = ansatz.state(theta, psi0)
    d = _param_derivatives(ansatz, theta, psi0)
    H_psi = H @ psi
    v = 1j * H_psi
    h_norm_sq = np.real(np.vdot(v, v))
    for i in range(len(theta)):
        v = v + thetadot[i] * d[i]
    raw_residual = float(np.real(np.vdot(v, v)))
    return raw_residual / (float(h_norm_sq) + 1e-15)


def solve_thetadot(A: np.ndarray, C: np.ndarray, ridge: float = 0.0):
    n = A.shape[0]
    return np.linalg.solve(A + ridge * np.eye(n), C)


def condition_number(A: np.ndarray, ridge: float = 0.0) -> float:
    M = A + ridge * np.eye(A.shape[0])
    return float(np.linalg.cond(M))


def adaptive_ridge(A: np.ndarray, target_cond: float = 1e8,
                    ridge_min: float = 1e-10, ridge_max: float = 1e-1) -> float:
    if condition_number(A, 0.0) <= target_cond:
        return 0.0
    lo, hi = np.log10(ridge_min), np.log10(ridge_max)
    best = ridge_max
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        r = 10 ** mid
        if condition_number(A, r) <= target_cond:
            best = r
            hi = mid
        else:
            lo = mid
    return best


def floquet_step_generator(N: int, k: float, p: float) -> np.ndarray:
    from qkt_quantum import floquet_U_exact
    U = floquet_U_exact(N, k, p)
    w, V = np.linalg.eig(U)
    angles = -np.angle(w)
    Hstep = (V * angles) @ np.linalg.inv(V)
    Hstep = 0.5 * (Hstep + Hstep.conj().T)
    return Hstep


# ---------------------------------------------------------------------------
# [PATCH] Stub functions so main.py can import successfully.
#
# These two functions were called by main.py's --vqs / --vqs-floquet flags
# but were never actually implemented anywhere in this file (or any other
# file in the repo). They are NOT real implementations -- they exist only
# so `from vqs import vqs_compare_depths, vqs_floquet_two_k` doesn't crash
# the whole script at import time.
#
# Do NOT pass --vqs or --vqs-floquet to main.py -- you'll just hit the
# NotImplementedError below, by design. All real VQS results (Figs. 7, 8,
# 12) come from adaptive_vqs_qkt.py, which is fully implemented and uses
# the real McLachlan residual trigger (mclachlan_residual_sq above).
# ---------------------------------------------------------------------------

def vqs_compare_depths(*args, **kwargs):
    raise NotImplementedError(
        "vqs_compare_depths was never implemented in this repo. "
        "Use adaptive_vqs_qkt.py for real VQS depth-comparison results "
        "(Figs. 7, 8, 12 in the paper)."
    )


def vqs_floquet_two_k(*args, **kwargs):
    raise NotImplementedError(
        "vqs_floquet_two_k was never implemented in this repo. "
        "Use adaptive_vqs_qkt.py for real VQS Floquet results "
        "(Figs. 7, 8, 12 in the paper)."
    )


if __name__ == "__main__":
    from spin_operators import coherent_product_state

    N, D = 4, 2
    ans = Ansatz(N, D)
    theta = np.full(ans.n_params, 0.1)
    psi0 = coherent_product_state(N)
    H = floquet_step_generator(N, 2.5, np.pi / 2)

    A, C, psi, d = mclachlan_AC(ans, theta, psi0, H)
    print("n_params:", ans.n_params, " A shape:", A.shape)
    print("cond(A):", f"{condition_number(A):.3e}")

    ridge = adaptive_ridge(A)
    print("chosen ridge:", f"{ridge:.2e}", " cond after:", f"{condition_number(A, ridge):.3e}")

    td = solve_thetadot(A, C, ridge)
    print("residual^2 at solved thetadot:",
          f"{mclachlan_residual_sq(ans, theta, td, psi0, H):.4e}")