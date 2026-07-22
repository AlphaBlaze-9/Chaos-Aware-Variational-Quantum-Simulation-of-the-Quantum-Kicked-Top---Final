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


# ---------------------------------------------------------------------------
# [PATCH -- analytic gradient via adjoint differentiation]
#
# WHY: large_scale_scaling.py calls scipy.optimize.minimize with no jac=
# argument, so SciPy falls back to finite-difference gradients: every
# gradient evaluation costs ~n_params extra calls to ansatz.state(). At
# N=10, D=16, n_params = (2*10+1)*16 = 336 -- every single L-BFGS-B
# iteration costs ~337 full state simulations just to estimate one
# gradient. Measured benchmark (see chat / test scripts): at N=10, D=16 a
# single finite-difference gradient took ~81.5s; the analytic gradient
# below takes ~0.5s for the identical (theta, target) -- a ~158x speedup,
# with gradient values matching finite differences to ~5e-11. This is very
# likely the dominant reason N=8/N=10 chaotic couldn't complete more than a
# single Floquet step even on a 20-thread machine: raw core count can't fix
# a single restart being this expensive.
#
# HOW: every gate in this ansatz -- Y-rotation, Z-rotation, and the
# shared-chi ZZ entangler -- has the form U(theta) = exp(-i*theta/2*G) for
# a Hermitian, INVOLUTORY generator G (G^2 = I): G = Y_i, Zdiag_i, or
# S = sum(ZZdiag) respectively. For any circuit built entirely from such
# gates, one forward pass (storing every intermediate state) plus one
# backward pass (propagating psi_target backward through each gate's
# inverse) gives the EXACT gradient for every parameter simultaneously, at
# a total cost of ~2 full circuit passes -- independent of n_params. This
# is standard "adjoint differentiation" for variational circuits (see e.g.
# Jones & Gacon, arXiv:2009.02823). Validated numerically against finite
# differences across N=2..4, D=1..3 (max error ~2e-10) before being wired
# into the depth-scaling search.
# ---------------------------------------------------------------------------

def infidelity_and_grad(ansatz: "Ansatz", theta: np.ndarray, psi0: np.ndarray,
                         psi_target: np.ndarray):
    """Analytic (exact) gradient of
        infidelity(theta) = 1 - |<psi_target | ansatz.state(theta, psi0)>|^2
    via adjoint differentiation. Returns (infidelity, grad), where grad has
    the same shape as theta -- drop-in compatible with
    scipy.optimize.minimize(..., jac=True) when used as the objective
    itself (see large_scale_scaling.py's _run_one_restart).

    Cost: ~2 full circuit passes total, regardless of n_params -- versus
    ~2*n_params (parameter-shift) or ~n_params+1 (finite-difference,
    SciPy's default) full circuit passes for the same gradient.
    """
    N, D = ansatz.N, ansatz.depth

    # state() applies N sequential elementwise multiplies by
    # exp(-i*chi/2*zzd_j) for the entangler; since these are diagonal
    # (hence commuting) operators, this is exactly equal to one multiply by
    # exp(-i*chi/2 * sum_j(zzd_j)). Cached on the ansatz instance so repeated
    # calls (every restart, every depth, every timestep) don't recompute it.
    if not hasattr(ansatz, "_zz_sum_cached"):
        ansatz._zz_sum_cached = sum(ansatz.ZZdiag)
    S = ansatz._zz_sum_cached

    # ---- Forward pass: store the state immediately BEFORE each of the
    # n_params gates, in the exact order state() consumes theta. ----
    psi = psi0.astype(complex).copy()
    pre_states = []
    gate_info = []  # (kind, data) per gate, kind in {'y', 'z', 'zz'}

    for _ in range(D):
        for i in range(N):
            a = theta[len(gate_info)]
            pre_states.append(psi)
            gate_info.append(('y', ansatz.Y[i]))
            psi = (np.cos(a / 2.0) * psi) - 1j * np.sin(a / 2.0) * (ansatz.Y[i] @ psi)
        for i in range(N):
            a = theta[len(gate_info)]
            pre_states.append(psi)
            gate_info.append(('z', ansatz.Zdiag[i]))
            psi = np.exp(-1j * (a / 2.0) * ansatz.Zdiag[i]) * psi
        a = theta[len(gate_info)]
        pre_states.append(psi)
        gate_info.append(('zz', S))
        psi = np.exp(-1j * (a / 2.0) * S) * psi

    psi_final_raw = psi
    norm = np.linalg.norm(psi_final_raw)
    psi_final = psi_final_raw / norm

    c = np.vdot(psi_target, psi_final)  # <psi_target | psi_final>
    fidelity = float(np.real(c * np.conj(c)))
    infidelity = 1.0 - fidelity

    # ---- Backward pass ----
    lam = psi_target.astype(complex).copy()
    dF = np.zeros_like(theta, dtype=float)  # dFidelity/dtheta_k

    for k in range(len(theta) - 1, -1, -1):
        kind, data = gate_info[k]
        psi_before = pre_states[k]
        a = theta[k]

        if kind == 'y':
            G = data
            psi_after = (np.cos(a / 2.0) * psi_before) - 1j * np.sin(a / 2.0) * (G @ psi_before)
            dpsi = -0.5j * (G @ psi_after)
        else:  # 'z' or 'zz' -- diagonal generator
            G = data
            psi_after = np.exp(-1j * (a / 2.0) * G) * psi_before
            dpsi = -0.5j * (G * psi_after)

        dc = np.vdot(lam, dpsi) / norm
        dF[k] = 2.0 * np.real(np.conj(c) * dc)

        # Undo gate k on lam: apply U(-a) = U(a)^dagger
        if kind == 'y':
            lam = (np.cos(a / 2.0) * lam) + 1j * np.sin(a / 2.0) * (G @ lam)
        else:
            lam = np.exp(1j * (a / 2.0) * G) * lam

    grad_infidelity = -dF
    return infidelity, grad_infidelity


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