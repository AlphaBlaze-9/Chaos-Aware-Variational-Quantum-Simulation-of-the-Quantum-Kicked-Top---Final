import numpy as np
from scipy.optimize import minimize
from functools import reduce
from spin_operators import embed, SX, SY, SZ, I2, normalize





def build_pool(N):
    pool = []
    for i in range(N):
        pool.append((f"Y{i}", embed(SY, i, N), 0))
        pool.append((f"Z{i}", embed(SZ, i, N), 0))
    for i in range(N):
        j = (i + 1) % N
        ZiZj = embed(SZ, i, N) @ embed(SZ, j, N)
        YiZj = embed(SY, i, N) @ embed(SZ, j, N)
        pool.append((f"Z{i}Z{j}", ZiZj, 2))
        pool.append((f"Y{i}Z{j}", YiZj, 2))
    return pool


def _apply_op(theta, G, psi):
    return np.cos(theta) * psi - 1j * np.sin(theta) * (G @ psi)


def _state_from_ops(thetas, gens, psi0):
    psi = psi0.astype(complex).copy()
    for th, G in zip(thetas, gens):
        psi = _apply_op(th, G, psi)
    return normalize(psi)


def _infidelity(thetas, gens, psi0, target):
    psi = _state_from_ops(thetas, gens, psi0)
    return 1.0 - abs(np.vdot(target, psi)) ** 2


def adapt_vqa_prepare(psi0, target, N, fid_target=0.99, grad_tol=1e-3,
                      max_ops=40, seed=0):
    rng = np.random.default_rng(seed)
    pool = build_pool(N)
    gens, thetas, labels = [], [], []
    cnot_count = 0
    opt_cycles = 0

    psi0 = normalize(psi0)
    target = normalize(target)

    for _ in range(max_ops):
        psi = _state_from_ops(thetas, gens, psi0)
        fid = abs(np.vdot(target, psi)) ** 2
        if fid >= fid_target:
            break

        ov = np.vdot(target, psi)                     
        best_g, best_idx = -1.0, None
        for idx, (lab, G, cc) in enumerate(pool):
            d_amp = np.vdot(target, -1j * (G @ psi)) 
            grad = -2.0 * np.real(np.conjugate(ov) * d_amp)
            if abs(grad) > best_g:
                best_g, best_idx = abs(grad), idx

        if best_g < grad_tol:
            break

        lab, G, cc = pool[best_idx]
        gens.append(G); thetas.append(0.0); labels.append(lab)
        cnot_count += cc


        res = minimize(_infidelity, np.array(thetas),
                       args=(gens, psi0, target), method="BFGS",
                       options={"maxiter": 200})
        thetas = list(res.x)
        opt_cycles += int(res.nfev)

    psi = _state_from_ops(thetas, gens, psi0)
    fid = float(abs(np.vdot(target, psi)) ** 2)
    return dict(fidelity=fid, n_ops=len(gens), cnot_count=cnot_count,
                opt_cycles=opt_cycles, ops=labels)





def layerwise_prepare(psi0, target, N, fid_target=0.99, max_depth=8, seed=0):

    from vqs import Ansatz
    target = normalize(target)
    cnot_count = 0
    opt_cycles = 0
    for D in range(1, max_depth + 1):
        ans = Ansatz(N, D)

        def cost(th):
            psi = ans.state(th, psi0)
            return 1.0 - abs(np.vdot(target, psi)) ** 2

        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0, 0.1, ans.n_params)
        res = minimize(cost, x0, method="BFGS", options={"maxiter": 300})
        opt_cycles += int(res.nfev)
        fid = 1.0 - res.fun
        if fid >= fid_target or D == max_depth:
            cnot_count = D * 2 * N  

            return dict(fidelity=float(fid), depth=D, cnot_count=cnot_count,
                        opt_cycles=opt_cycles)
    return dict(fidelity=float(fid), depth=max_depth,
                cnot_count=max_depth * 2 * N, opt_cycles=opt_cycles)


if __name__ == "__main__":
    from spin_operators import coherent_product_state
    from qkt_quantum import floquet_U_exact

    N = 4
    psi0 = coherent_product_state(N)
    U = floquet_U_exact(N, 2.5, np.pi / 2)
    target = np.linalg.matrix_power(U, 4) @ psi0 

    adapt = adapt_vqa_prepare(psi0, target, N)
    layer = layerwise_prepare(psi0, target, N)
    print("ADAPT-VQA :", adapt)
    print("Layer-wise:", layer)
