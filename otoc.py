



from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Iterable, Dict, List

from qkt_quantum import floquet_U
from spin_operators import hadamard_state  




def _pauli(axis: str) -> np.ndarray:
    axis = axis.lower()
    if axis == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if axis == "y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if axis == "z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if axis == "i":
        return np.eye(2, dtype=complex)
    raise ValueError(f"unknown axis '{axis}' (use X/Y/Z)")


def local_pauli(N: int, site: int, axis: str) -> np.ndarray:
    
    if not (0 <= site < N):
        raise ValueError(f"site {site} out of range for N={N}")
    mats = [_pauli("i")] * N
    mats[site] = _pauli(axis)
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def parse_local(spec: str, N: int) -> np.ndarray:
    
    spec = spec.strip().upper()
    if len(spec) < 2:
        raise ValueError("Use e.g. 'X1' or 'Z2'")
    axis = spec[0]
    idx = int(spec[1:])
    site = idx - 1 if 1 <= idx <= N else idx
    if axis not in ("X", "Y", "Z"):
        raise ValueError("Axis must be X, Y or Z.")
    return local_pauli(N, site, axis)




def heisenberg_evolve_operator(U: np.ndarray, W: np.ndarray, t: int) -> np.ndarray:
    
    Wt = W.copy()
    Udag = U.conj().T
    for _ in range(t):
        Wt = Udag @ Wt @ U
    return Wt




@dataclass
class OTOCResult:
    t: np.ndarray        
    F: np.ndarray        
    C: np.ndarray        
    meta: Dict[str, str] 


def otoc_trace(U: np.ndarray, W: np.ndarray, V: np.ndarray, T: int) -> OTOCResult:
    
    d = U.shape[0]
    Udag = U.conj().T
    Vdag = V.conj().T

    Wt = W.copy()
    F = np.zeros(T + 1, dtype=complex)
    C = np.zeros(T + 1, dtype=float)

    for t in range(T + 1):
        F[t] = np.trace(Wt.conj().T @ Vdag @ Wt @ V) / d
        C[t] = 2.0 * (1.0 - float(np.real(F[t])))
        Wt = Udag @ Wt @ U  

    return OTOCResult(
        t=np.arange(T + 1),
        F=F,
        C=C,
        meta={"average": "trace", "note": "Pauli assumption for C(t) identity"},
    )


def otoc_state_heisenberg(
    U: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    psi0: np.ndarray,
    T: int,
) -> OTOCResult:
    
    Udag = U.conj().T
    Vdag = V.conj().T

    Wt = W.copy()
    F = np.zeros(T + 1, dtype=complex)
    C = np.zeros(T + 1, dtype=float)

    for t in range(T + 1):
        F[t] = np.vdot(psi0, (Wt.conj().T @ Vdag @ Wt @ V) @ psi0)
        C[t] = 2.0 * (1.0 - float(np.real(F[t])))
        Wt = Udag @ Wt @ U

    return OTOCResult(
        t=np.arange(T + 1),
        F=F,
        C=C,
        meta={"average": "state", "note": "Heisenberg evaluation on pure state"},
    )


def otoc_forward_backward(
    U: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    psi0: np.ndarray,
    T: int,
) -> OTOCResult:
    
    Udag = U.conj().T
    F = np.zeros(T + 1, dtype=complex)
    C = np.zeros(T + 1, dtype=float)
    Wdag, Vdag = W.conj().T, V.conj().T

    for t in range(T + 1):
        psi = V @ psi0
        for _ in range(t):
            psi = U @ psi
        psi = W @ psi
        for _ in range(t):
            psi = Udag @ psi
        psi = Vdag @ psi
        for _ in range(t):
            psi = U @ psi
        psi = Wdag @ psi
        for _ in range(t):
            psi = Udag @ psi

        F[t] = np.vdot(psi0, psi)
        C[t] = 2.0 * (1.0 - float(np.real(F[t])))

    return OTOCResult(
        t=np.arange(T + 1),
        F=F,
        C=C,
        meta={"average": "state", "note": "forward-backward echo"},
    )




def compute_otoc_QKT(
    N: int,
    k: float,
    p: float,
    T: int,
    W_spec: str = "X1",
    V_spec: str = "Z2",
    mode: str = "trace",  
) -> OTOCResult:
    
    U = floquet_U(N, k=k, p=p)
    W = parse_local(W_spec, N)
    V = parse_local(V_spec, N)

    if mode == "trace":
        res = otoc_trace(U, W, V, T)
    elif mode == "state":
        psi0 = hadamard_state(N)
        res = otoc_state_heisenberg(U, W, V, psi0, T)
    elif mode == "echo":
        psi0 = hadamard_state(N)
        res = otoc_forward_backward(U, W, V, psi0, T)
    else:
        raise ValueError("mode must be 'trace', 'state', or 'echo'")

    res.meta.update({"N": str(N), "k": f"{k}", "p": f"{p}", "W": W_spec, "V": V_spec, "mode": mode})
    return res


def save_otoc_plot(
    res_list: List[OTOCResult],
    title: str,
    out_png: str,
):
    
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.0, 4.5))
    ax = plt.gca()
    eps = 1e-12
    for res in res_list:
        label = f"k={res.meta['k']}, {res.meta['mode']}"
        ax.semilogy(res.t, res.C + eps, label=label, linewidth=2.0)
    ax.set_xlabel("time step $t$", fontsize=16)
    ax.set_ylabel(r"$C(t) = -\langle [W(t), V]^2 \rangle$", fontsize=16)
    
    ax.set_ylim(1e-8, 3.0)
    ax.legend(fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
