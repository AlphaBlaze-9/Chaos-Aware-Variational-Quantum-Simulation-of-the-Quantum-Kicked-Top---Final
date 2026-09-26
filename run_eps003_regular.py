"""Table II regular-plateau check: D_max at eps_opt=0.03 for k in {0.5, 1.0, 1.5}.

Usage:  python run_eps003_regular.py [k ...]     (default: 0.5 1.5 1.0)

Reproduces plot_dmax_vs_k.dmax_direct() EXACTLY (N=6, 12 Floquet steps, depths 1..8,
50 L-BFGS-B restarts per depth seeded with default_rng((int(1000k), t, D, r)),
success iff infidelity <= eps_opt), but checkpoints after every (t, D) to
eps003_checkpoint_k{k}.json so an interrupted run resumes where it stopped.
Final summary is written to eps003_regular_result.json.
"""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize
from qkt_quantum import floquet_U_exact
from spin_operators import coherent_product_state, normalize
from plot_dmax_vs_k import _make_ansatz

N, STEPS, EPS, MAXD, NR = 6, 12, 0.03, 8, 50

def run_k(k):
    ck = f"eps003_checkpoint_k{k}.json"
    st = json.load(open(ck)) if os.path.exists(ck) else {"k": k, "done": {}, "dmax_per_t": {}}
    psi0 = coherent_product_state(N); U = floquet_U_exact(N, k, np.pi/2)
    t0 = time.time()
    for t in range(1, STEPS+1):
        if str(t) in st["dmax_per_t"]:
            continue
        psi_t = psi0.copy()
        for _ in range(t): psi_t = normalize(U @ psi_t)
        for D in range(1, MAXD+1):
            key = f"{t},{D}"
            if key in st["done"]:
                n_success = st["done"][key]
            else:
                ans = _make_ansatz(N, D, "nn"); n_success = 0
                for r in range(NR):
                    rng = np.random.default_rng((int(k*1000), t, D, r))
                    x0 = rng.uniform(0, 2*np.pi, ans.n_params)
                    res = minimize(lambda th: 1.0 - abs(np.vdot(psi_t, ans.state(th, psi0)))**2,
                                   x0, method="L-BFGS-B", options={"maxiter": 300})
                    if res.fun <= EPS: n_success += 1
                st["done"][key] = n_success
                json.dump(st, open(ck, "w"))
            print(f"  k={k} t={t:2d} D={D}: {n_success}/{NR} successes  [{time.time()-t0:.0f}s]", flush=True)
            if n_success >= 1:
                st["dmax_per_t"][str(t)] = D; json.dump(st, open(ck, "w")); break
        else:
            st["dmax_per_t"][str(t)] = None; json.dump(st, open(ck, "w"))  # ceiling hit
    dm = [v for v in st["dmax_per_t"].values() if v is not None]
    result = {"dmax_eps003": max(dm) if dm else None, "ceiling_hit": any(v is None for v in st["dmax_per_t"].values()),
              "dmax_per_t": st["dmax_per_t"], "success_counts": st["done"], "n_restarts": NR, "eps_opt": EPS, "N": N, "steps": STEPS}
    print(f"=== k={k}: D_max(eps_opt=0.03) = {result['dmax_eps003']} ; per-t = {st['dmax_per_t']} ===", flush=True)
    return result

if __name__ == "__main__":
    ks = [float(a) for a in sys.argv[1:]] or [0.5, 1.5, 1.0]
    out = json.load(open("eps003_regular_result.json")) if os.path.exists("eps003_regular_result.json") else {}
    for k in ks:
        out[str(k)] = run_k(k); json.dump(out, open("eps003_regular_result.json", "w"), indent=1)
    print("ALL DONE", flush=True)
