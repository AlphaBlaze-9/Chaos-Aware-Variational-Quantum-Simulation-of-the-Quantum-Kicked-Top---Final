"""
residual_at_optimum.py

Control test for the depth-adequacy interpretation of the normalized McLachlan
residual r^2.

In the layer-wise loop, r^2 is evaluated at the *integrator's* parameters, which
in the chaotic regime have drifted after fidelity collapse. That conflates two
distinct failures: (i) the ansatz tangent space is genuinely too small at this
depth, and (ii) the trajectory is in the wrong place. This script separates them
by evaluating r^2 at the parameters found by direct optimization -- i.e. at a
point that is known to represent the target state to within eps_opt.

Prediction if r^2 is a true depth-adequacy diagnostic:
    r^2 < eps_trig  exactly when  D >= D_min(k,t),
    r^2 > eps_trig  when          D <  D_min(k,t).

Outputs residual_at_optimum.json
"""
from __future__ import annotations

import json
import time
import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize

from vqs import (Ansatz, mclachlan_AC, mclachlan_residual_sq,
                 solve_thetadot, floquet_step_generator)
from spin_operators import collective_J

N = 6
P = np.pi / 2
EPS_OPT = 0.05
EPS_TRIG = 0.85
N_RESTARTS = 10
MAXITER = 300
MAX_DEPTH = 8
N_STEPS = 12
RIDGE = 1e-10


def initial_state(N):
    ct, st = np.cos(np.pi / 8), np.sin(np.pi / 8)
    one = np.array([ct, st], dtype=complex)
    psi = one
    for _ in range(N - 1):
        psi = np.kron(psi, one)
    return psi / np.linalg.norm(psi)


def floquet_op(N, k, p):
    _, Jy, Jz = collective_J(N)
    j = N / 2
    return sla.expm(-1j * (k / (2 * j)) * (Jz @ Jz)) @ sla.expm(-1j * p * Jy)


def run(k, out):
    H = floquet_step_generator(N, k, P)
    U = floquet_op(N, k, P)
    psi0 = initial_state(N)
    rng = np.random.default_rng(7)
    psi_t = psi0.copy()

    for t in range(1, N_STEPS + 1):
        psi_t = U @ psi_t
        rec = []
        for D in range(1, MAX_DEPTH + 1):
            ans = Ansatz(N=N, depth=D)

            def infid(x):
                return float(1 - abs(np.vdot(psi_t, ans.state(x, psi0))) ** 2)

            best = None
            for _ in range(N_RESTARTS):
                r = minimize(infid, rng.uniform(0, 2 * np.pi, (2 * N + 1) * D),
                             method="L-BFGS-B", options={"maxiter": MAXITER})
                if best is None or r.fun < best.fun:
                    best = r
            A, C, psi, d = mclachlan_AC(ans, best.x, psi0, H)
            td = solve_thetadot(A, C, ridge=RIDGE)
            r2 = mclachlan_residual_sq(ans, best.x, td, psi0, H)
            suff = best.fun < EPS_OPT
            rec.append({"D": D, "infid": float(best.fun),
                        "r2_at_opt": float(r2), "sufficient": bool(suff)})
            print(f"  k={k} t={t:2d} D={D}  infid={best.fun:.4f} "
                  f"r2={r2:.4f} {'SUFF' if suff else 'insuff'}", flush=True)
            if suff:
                break
        out[f"{k},{t}"] = rec


if __name__ == "__main__":
    out = {"N": N, "eps_opt": EPS_OPT, "eps_trig": EPS_TRIG,
           "n_restarts": N_RESTARTS, "results": {}}
    t0 = time.time()
    import sys
    ks = [float(x) for x in sys.argv[1:]] or [0.5, 2.5]
    for k in ks:
        run(k, out["results"])
    out["wallclock_s"] = time.time() - t0
    tag = "_".join(str(k) for k in ks)
    json.dump(out, open(f"residual_at_optimum_{tag}.json", "w"), indent=1)
    print(f"\nDone in {out['wallclock_s']:.0f}s")
