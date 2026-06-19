

from functools import reduce
import numpy as np


SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def embed(op2: np.ndarray, i: int, N: int) -> np.ndarray:
    
    return reduce(np.kron, [op2 if j == i else I2 for j in range(N)])


def pauli_lists(N: int):
    
    ops = {"X": [], "Y": [], "Z": []}
    for i in range(N):
        ops["X"].append(embed(SX, i, N))
        ops["Y"].append(embed(SY, i, N))
        ops["Z"].append(embed(SZ, i, N))
    return ops


def collective_J(N: int):
    
    dim = 2 ** N
    Jx = np.zeros((dim, dim), dtype=complex)
    Jy = np.zeros((dim, dim), dtype=complex)
    Jz = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        Jx += embed(SX, i, N) / 2.0
        Jy += embed(SY, i, N) / 2.0
        Jz += embed(SZ, i, N) / 2.0
    return Jx, Jy, Jz


def hadamard_state(N: int) -> np.ndarray:
    
    h = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi = h
    for _ in range(N - 1):
        psi = np.kron(psi, h)
    return psi


def coherent_product_state(N: int, theta: float = np.pi / 4) -> np.ndarray:
    
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    single = np.array([c, s], dtype=complex)  
    psi = single
    for _ in range(N - 1):
        psi = np.kron(psi, single)
    return psi / np.linalg.norm(psi)


def spin_coherent_state(N: int, theta: float, phi: float) -> np.ndarray:
    
    _, Jy, Jz = collective_J(N)
    top = np.zeros(2 ** N, dtype=complex)
    top[0] = 1.0  
    state = expm_unitary(-1j * theta * Jy) @ top
    state = expm_unitary(-1j * phi * Jz) @ state
    return state / np.linalg.norm(state)


def expm_unitary(A: np.ndarray) -> np.ndarray:
    
    
    H = 1j * A
    
    H = 0.5 * (H + H.conj().T)
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * w)) @ V.conj().T


def normalize(psi: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(psi)
    return psi if n < 1e-15 else psi / n
