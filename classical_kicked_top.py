from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Dict, Optional

K_REGULAR: float = 0.5
K_CHAOTIC: float = 2.5

def sph_to_cart(theta: float, phi: float) -> np.ndarray:
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], dtype=float)

def cart_to_sph(v: np.ndarray) -> Tuple[float, float]:
    x, y, z = v
    r = np.linalg.norm(v)
    if r == 0:
        return 0.0, 0.0
    zc = np.clip(z / r, -1.0, 1.0)
    theta = np.arccos(zc)
    phi = (np.arctan2(y, x)) % (2.0 * np.pi)
    return theta, phi

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0.0 else v / n

@dataclass
class CKTopParams:
    k: float = K_CHAOTIC
    p: float = np.pi / 2

def twist_z(v: np.ndarray, k: float) -> np.ndarray:
    x, y, z = v
    a = k * z
    ca, sa = np.cos(a), np.sin(a)
    return np.array([x * ca - y * sa, x * sa + y * ca, z], dtype=float)

def rot_y(v: np.ndarray, p: float) -> np.ndarray:
    x, y, z = v
    cp, sp = np.cos(p), np.sin(p)
    return np.array([cp * x + sp * z, y, -sp * x + cp * z], dtype=float)

def kicked_top_step(v: np.ndarray, params: CKTopParams) -> np.ndarray:
    return normalize(rot_y(twist_z(v, params.k), params.p))

def angle_distance(u: np.ndarray, v: np.ndarray) -> float:
    dot = float(np.dot(u, v))
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.arccos(dot))

def project_to_tangent(base: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w - np.dot(w, base) * base

def rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    a = normalize(axis)
    c, s = np.cos(angle), np.sin(angle)
    return (v * c + np.cross(a, v) * s + a * np.dot(a, v) * (1.0 - c))

def renormalize_neighbor(base: np.ndarray, neighbor: np.ndarray, target_angle: float) -> np.ndarray:
    d = neighbor - base
    t = project_to_tangent(base, d)
    nt = np.linalg.norm(t)

    if nt < 1e-15:
        e = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(e, base)) > 0.9:
            e = np.array([0.0, 1.0, 0.0])
        t = project_to_tangent(base, e)
        t = normalize(t)
    else:
        t = t / nt

    axis = np.cross(t, base)
    if np.linalg.norm(axis) < 1e-15:
        axis = normalize(np.cross(np.array([1.0, 0.0, 0.0]), base))
        if np.linalg.norm(axis) < 1e-15:
            axis = normalize(np.cross(np.array([0.0, 1.0, 0.0]), base))

    new_neighbor = rotate_about_axis(base, axis, target_angle)
    return normalize(new_neighbor)

@dataclass
class FTLEConfig:
    steps: int = 2000
    burn_in: int = 200
    delta0: float = 1e-6
    rescale_each: int = 1

def finite_time_lyapunov(
    v0: np.ndarray,
    params: CKTopParams,
    cfg: Optional[FTLEConfig] = None
) -> float:
    if cfg is None:
        cfg = FTLEConfig()

    rng = np.random.default_rng(12345)

    base = normalize(v0)
    rand = rng.normal(size=3)
    t0 = project_to_tangent(base, rand)
    if np.linalg.norm(t0) < 1e-12:
        t0 = project_to_tangent(base, np.array([1.0, 0.0, 0.0]))
    t0 = normalize(t0)
    axis0 = normalize(np.cross(t0, base))
    w = rotate_about_axis(base, axis0, cfg.delta0)

    v = base.copy()

    for _ in range(cfg.burn_in):
        v = kicked_top_step(v, params)
        w = kicked_top_step(w, params)
        w = renormalize_neighbor(v, w, cfg.delta0)

    sum_log = 0.0
    count = 0

    for _ in range(cfg.steps):
        v = kicked_top_step(v, params)
        w = kicked_top_step(w, params)
        delta = angle_distance(v, w)
        if delta < 1e-300:
            delta = 1e-300
        sum_log += np.log(delta / cfg.delta0)
        count += 1
        w = renormalize_neighbor(v, w, cfg.delta0)

    return 0.0 if count == 0 else float(sum_log / count)

def random_on_sphere(num: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=num)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=num)
    theta = np.arccos(u)
    pts = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), u],
        axis=1
    )
    return pts

def grid_initial_conditions(n_theta: int, n_phi: int) -> np.ndarray:
    eps = 1e-3
    thetas = np.linspace(eps, np.pi - eps, n_theta)
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    pts = []
    for th in thetas:
        st = np.sin(th)
        for ph in phis:
            pts.append([st * np.cos(ph), st * np.sin(ph), np.cos(th)])
    return np.array(pts, dtype=float)

def poincare_points_for_k(
    k: float,
    p: float,
    n_ic: int = 64,
    steps: int = 1200,
    discard: int = 200,
    seed: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    params = CKTopParams(k=k, p=p)
    v0s = random_on_sphere(n_ic, seed)
    phis_all: List[float] = []
    z_all: List[float] = []

    for v0 in v0s:
        v = normalize(v0)
        for i in range(steps):
            v = kicked_top_step(v, params)
            if i >= discard:
                _, ph = cart_to_sph(v)
                phis_all.append(ph)
                z_all.append(v[2])

    return np.array(phis_all, dtype=float), np.array(z_all, dtype=float)

def poincare_compare_points(
    p: float,
    k_regular: float = K_REGULAR,
    k_chaotic: float = K_CHAOTIC,
    n_ic: int = 64,
    steps: int = 1200,
    discard: int = 200,
    seed_regular: int = 2,
    seed_chaotic: int = 3
) -> Dict[str, Dict[str, np.ndarray]]:
    phi_r, z_r = poincare_points_for_k(
        k=k_regular, p=p, n_ic=n_ic, steps=steps, discard=discard, seed=seed_regular
    )
    phi_c, z_c = poincare_points_for_k(
        k=k_chaotic, p=p, n_ic=n_ic, steps=steps, discard=discard, seed=seed_chaotic
    )

    return {
        "regular": {
            "k": np.array([k_regular], dtype=float),
            "phi": phi_r,
            "z": z_r,
            "title": f"Regular: k = {k_regular}",
        },
        "chaotic": {
            "k": np.array([k_chaotic], dtype=float),
            "phi": phi_c,
            "z": z_c,
            "title": f"Chaotic: k = {k_chaotic}",
        },
    }

def ftle_scan_over_k(
    k_values: Iterable[float],
    p: float,
    n_seeds: int = 8,
    ftle_cfg: Optional[FTLEConfig] = None
) -> Dict[str, np.ndarray]:
    if ftle_cfg is None:
        ftle_cfg = FTLEConfig(steps=1500, burn_in=150, delta0=1e-6)

    ks = list(k_values)
    means, stds = [], []

    for k in ks:
        params = CKTopParams(k=k, p=p)
        vals = []
        v0s = random_on_sphere(n_seeds, seed=int(1000 * k) % (2**32 - 1))
        for v0 in v0s:
            lam = finite_time_lyapunov(v0, params, cfg=ftle_cfg)
            vals.append(lam)

        arr = np.array(vals, dtype=float)
        means.append(np.mean(arr))
        stds.append(np.std(arr))

    return {"k": np.array(ks), "mean_ftle": np.array(means), "std_ftle": np.array(stds)}

def first_positive_threshold(k: np.ndarray, lam: np.ndarray, smooth: int = 3):
    if smooth > 1:
        L = np.convolve(lam, np.ones(smooth) / smooth, mode="same")
    else:
        L = lam
    for i in range(1, len(L)):
        if L[i - 1] <= 0.0 and L[i] > 0.0:
            return float(k[i])
    return None