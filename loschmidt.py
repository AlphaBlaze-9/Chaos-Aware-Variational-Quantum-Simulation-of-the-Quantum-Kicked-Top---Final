
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from qkt_quantum import floquet_U
from spin_operators import hadamard_state


@dataclass
class EchoResult:
    t: np.ndarray          
    L: np.ndarray          
    meta: Dict[str, str]


def compute_echo_QKT(
    N: int,
    k: float,
    p: float,
    T: int,
    rel_delta_k: float = 0.01,   
    use_plus_state: bool = True,
) -> EchoResult:
    
    U  = floquet_U(N, k=k, p=p)
    Up = floquet_U(N, k=k * (1.0 + rel_delta_k), p=p)

    psi  = hadamard_state(N) if use_plus_state else np.eye(2**N)[:, 0]
    psip = psi.copy()

    T = int(T)
    L = np.zeros(T + 1, dtype=float)
    for t in range(T + 1):
        L[t] = abs(np.vdot(psi, psip))**2
        if t < T:
            psi  = U  @ psi
            psip = Up @ psip

    return EchoResult(
        t=np.arange(T + 1),
        L=L,
        meta={"N": str(N), "k": f"{k}", "p": f"{p}",
              "rel_delta_k": f"{rel_delta_k}"}
    )


def save_echo_plot(results: List[EchoResult], out_png: str, title: str = "Loschmidt Echo"):
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.0, 4.5))
    ax = plt.gca()
    for r in results:
        label = f"k={r.meta['k']}, $\\Delta k/k$={r.meta['rel_delta_k']}"
        ax.plot(r.t, r.L, label=label, linewidth=2.0)
    ax.set_xlabel("time step $t$", fontsize=16)
    ax.set_ylabel(r"$L(t)=|\langle \psi(t) | \psi'(t) \rangle|^2$", fontsize=16)
    
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()