"""
ftle_convergence.py -- T-dependence of the finite-time Lyapunov exponent.

Purpose: distinguish a genuine positive Lyapunov exponent from the residual
lambda ~ ln(T)/T that any finite-time estimator returns on regular (algebraically
separating) trajectories.

Reproduces the numbers quoted in Sec. III C of the manuscript:

    T      ln(T)/T   k=0.5    k=1.5    k=2.5
    150     0.0334   0.0158   0.0236   0.1026
    300     0.0190   0.0100   0.0137   0.1023
    600     0.0107   0.0061   0.0078   0.0997
   1200     0.0059   0.0036   0.0044   0.1027
   2400     0.0032   0.0020   0.0024   0.1034

k=0.5 and k=1.5 decay in proportion to ln(T)/T and extrapolate to zero;
k=2.5 is T-independent at ~0.10 nats/kick.

Also regenerates figures/ftle_vs_k.csv at the parameters stated in the paper
(T=300, burn_in=0, N_IC=80) -- note that these are NOT the library defaults.

Run:  python ftle_convergence.py
"""
from __future__ import annotations

import csv
import numpy as np

from classical_kicked_top import ftle_scan_over_k, FTLEConfig

P = np.pi / 2
N_IC = 80
T_MAIN = 300
BURN_IN = 0


def convergence_table(ks=(0.5, 1.5, 2.5), Ts=(150, 300, 600, 1200, 2400)):
    print("T-dependence of <FTLE>  (N_IC=%d, burn_in=%d)" % (N_IC, BURN_IN))
    header = f"{'T':>6} {'ln(T)/T':>9} | " + " ".join(f"k={k:<6}" for k in ks)
    print(header)
    for T in Ts:
        cfg = FTLEConfig(steps=T, burn_in=BURN_IN, delta0=1e-6, rescale_each=1)
        res = ftle_scan_over_k(list(ks), p=P, n_seeds=N_IC, ftle_cfg=cfg)
        row = " ".join(f"{m:.4f}   " for m in res["mean_ftle"])
        print(f"{T:>6} {np.log(T)/T:>9.4f} | {row}")


def regenerate_csv(path="figures/ftle_vs_k.csv", n_k=36):
    ks = list(np.linspace(0.5, 4.0, n_k))
    cfg = FTLEConfig(steps=T_MAIN, burn_in=BURN_IN, delta0=1e-6, rescale_each=1)
    res = ftle_scan_over_k(ks, p=P, n_seeds=N_IC, ftle_cfg=cfg)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "mean_ftle", "std_ftle", "sem_ftle",
                    "p", "steps", "burn_in", "n_seeds"])
        for k, m, s in zip(res["k"], res["mean_ftle"], res["std_ftle"]):
            w.writerow([k, m, s, s / np.sqrt(N_IC), P, T_MAIN, BURN_IN, N_IC])
    print(f"\nWrote {path} (T={T_MAIN}, burn_in={BURN_IN}, N_IC={N_IC})")


if __name__ == "__main__":
    convergence_table()
    regenerate_csv()
