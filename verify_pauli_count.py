"""
verify_pauli_count.py  --  Item 5: the real Pauli-term count for H_eff^2 at N=4.

Finding: the manuscript says "45 Pauli terms for H_eff^2" and "126 measurement
settings (81 + 45)". The actual Pauli decomposition of H_eff^2 at N=4, k=2.5 has
128 non-zero Pauli strings (127 excluding identity) -- NOT 45. A greedy
qubit-wise-commuting (QWC) grouping needs ~55 measurement settings.

So "45" does not reproduce. Fix options for the manuscript:
  (a) state the true Pauli-term count (128), or
  (b) report QWC-grouped measurement settings (run with optimal grouping;
      greedy gives ~55), and update the "126 measurement settings" total
      accordingly (81 for A,C + however many settings H_eff^2 needs).

NOTE: hardware_trigger_poc.py already prints the live count via Qiskit
(len(H_sq_op)); run that on your machine to get the exact number your hardware
job used, and paste THAT into the paper.

Run:  python verify_pauli_count.py
"""
import numpy as np
from itertools import product
from scipy.linalg import logm
from qkt_quantum import floquet_U_exact

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron(ops):
    M = ops[0]
    for o in ops[1:]:
        M = np.kron(M, o)
    return M


def qwc(a, b):
    return all(not (x != "I" and y != "I" and x != y) for x, y in zip(a, b))


if __name__ == "__main__":
    N, p, k = 4, np.pi / 2, 2.5
    U = floquet_U_exact(N, k, p)
    H = 1j * logm(U)
    H = 0.5 * (H + H.conj().T)
    H2 = H @ H
    d = 2 ** N
    terms = {}
    for L in product("IXYZ", repeat=N):
        c = np.trace(kron([P[l] for l in L]) @ H2) / d
        if abs(c) > 1e-9:
            terms["".join(L)] = c
    nonI = {s: c for s, c in terms.items() if set(s) != {"I"}}
    print(f"H_eff^2 (N=4, k=2.5): {len(terms)} non-zero Pauli strings "
          f"({len(nonI)} excluding identity)")
    groups = []
    for s in nonI:
        for g in groups:
            if all(qwc(s, t) for t in g):
                g.append(s)
                break
        else:
            groups.append([s])
    print(f"Greedy QWC measurement settings: {len(groups)}")
    print(f"Manuscript claims 45 -> NOT reproduced. Use {len(terms)} terms "
          f"or ~{len(groups)} grouped settings.")
