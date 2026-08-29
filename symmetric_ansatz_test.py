"""
symmetric_ansatz_test.py

Reproduces the symmetric-subspace (collective-generator) ansatz control test
reported in the Discussion of the manuscript.

Ansatz: D layers, each  U_l = exp(-i c_l Jz^2) exp(-i a_l Jz) exp(-i b_l Jy),
3 parameters per layer. Every generator is permutation-symmetric, so the state
cannot leave the (N+1)-dimensional totally symmetric subspace; the computation
is therefore carried out directly in the spin-j basis (j=N/2), which is exact
and much cheaper than the full 2^N space.

Protocol matches the main N-scaling result: eps_opt=0.05, 50 random restarts
per depth, Floquet steps t=1..10, k in {0.5,1.5,2.5}, N in {4,6,8}.
"""
from __future__ import annotations
import json, sys
import numpy as np
from scipy.optimize import minimize

EPS_OPT, N_RESTARTS, N_STEPS, MAXITER = 0.05, 50, 10, 400
KS = {"0.5": 0.5, "1.5": 1.5, "2.5": 2.5}
P = np.pi / 2


def spin_ops(j):
    """Jy (dense), Jz (diagonal) in the |j,m> basis, m = j..-j."""
    m = np.arange(j, -j - 1, -1.0)
    d = len(m)
    Jz = m.copy()
    off = np.sqrt(j * (j + 1) - m[1:] * (m[1:] + 1))  # <m|J+|m+1>
    Jp = np.zeros((d, d))
    for i in range(d - 1):
        Jp[i, i + 1] = off[i]
    Jm = Jp.T
    Jy = (Jp - Jm) / (2j)
    return Jy, Jz


def cached(j):
    Jy, Jz = spin_ops(j)
    wy, Vy = np.linalg.eigh(Jy)
    return {"wy": wy, "Vy": Vy, "Jz": Jz, "Jz2": Jz ** 2, "d": len(Jz)}


def expJy(b, c):
    return (c["Vy"] * np.exp(-1j * b * c["wy"])) @ c["Vy"].conj().T


def coherent_j(j, theta):
    """[R_y(theta)|0>]^{oxN} expressed in |j,m>: binomial amplitudes."""
    from math import comb
    N = int(2 * j)
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    amp = np.array([np.sqrt(comb(N, i)) * ct ** (N - i) * st ** i
                    for i in range(N + 1)], dtype=complex)
    return amp / np.linalg.norm(amp)


def floquet(j, k, p, c):
    return np.diag(np.exp(-1j * (k / (2 * j)) * c["Jz2"])) @ expJy(p, c)


def state(x, D, c, psi0):
    psi = psi0
    for l in range(D):
        b, a, cc = x[3 * l:3 * l + 3]
        psi = expJy(b, c) @ psi
        psi = np.exp(-1j * a * c["Jz"]) * psi
        psi = np.exp(-1j * cc * c["Jz2"]) * psi
    return psi


def infid(x, D, c, psi0, tgt):
    return float(1.0 - abs(np.vdot(tgt, state(x, D, c, psi0))) ** 2)


def min_depth(c, psi0, tgt, max_depth, rng):
    for D in range(1, max_depth + 1):
        for _ in range(N_RESTARTS):
            r = minimize(infid, rng.uniform(0, 2 * np.pi, 3 * D),
                         args=(D, c, psi0, tgt), method="L-BFGS-B",
                         options={"maxiter": MAXITER})
            if r.fun < EPS_OPT:
                return D
    return None


def main():
    out = {"eps_opt": EPS_OPT, "n_restarts": N_RESTARTS,
           "n_steps": N_STEPS, "results": {}}
    for N in (4, 6, 8):
        j = N / 2
        c = cached(j)
        psi0 = coherent_j(j, np.pi / 4)
        rng = np.random.default_rng(12345 + N)
        for ks, k in KS.items():
            U = floquet(j, k, P, c)
            depths, psi_t = [], psi0.copy()
            for t in range(1, N_STEPS + 1):
                psi_t = U @ psi_t
                depths.append(min_depth(c, psi0, psi_t, N + 12, rng))
            a = np.array(depths, float)
            out["results"][f"{N},{ks}"] = {
                "mean_depth": float(a.mean()), "std_depth": float(a.std(ddof=1)),
                "max_depth": int(a.max()), "depths": depths}
            print(f"N={N} k={k}: <D>_t={a.mean():.2f}+-{a.std(ddof=1):.2f} "
                  f"depths={depths}", flush=True)

    json.dump(out, open("symmetric_ansatz_results.json", "w"), indent=1)
    print("\n=== SYMMETRIC-SUBSPACE ANSATZ <D>_t ===")
    print(f"{'N':>3} {'k=0.5':>7} {'k=1.5':>7} {'k=2.5':>7} {'gap':>7}")
    for N in (4, 6, 8):
        r, al, ch = (out["results"][f"{N},{s}"]["mean_depth"] for s in ("0.5", "1.5", "2.5"))
        print(f"{N:>3} {r:>7.2f} {al:>7.2f} {ch:>7.2f} {ch-r:>7.2f}")


if __name__ == "__main__":
    main()
