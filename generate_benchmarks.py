import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from spin_operators import coherent_product_state
from qkt_quantum import floquet_U_exact
from adapt_vqa_baseline import adapt_vqa_prepare, layerwise_prepare


def run_benchmark(N=4, k=2.5, target_steps=(1, 2, 3, 4, 5),
                  fid_target=0.99, seed=0):
    psi0 = coherent_product_state(N)
    U = floquet_U_exact(N, k, np.pi / 2)

    rows = []
    for t in target_steps:
        target = np.linalg.matrix_power(U, t) @ psi0
        adapt = adapt_vqa_prepare(psi0, target, N, fid_target=fid_target, seed=seed)
        layer = layerwise_prepare(psi0, target, N, fid_target=fid_target, seed=seed)
        rows.append(dict(t=t,
                         adapt_cnot=adapt["cnot_count"], adapt_cyc=adapt["opt_cycles"],
                         adapt_fid=adapt["fidelity"],
                         layer_cnot=layer["cnot_count"], layer_cyc=layer["opt_cycles"],
                         layer_fid=layer["fidelity"]))
        print(f"t={t}: ADAPT cnot={adapt['cnot_count']} cyc={adapt['opt_cycles']} "
              f"F={adapt['fidelity']:.3f} | LAYER cnot={layer['cnot_count']} "
              f"cyc={layer['opt_cycles']} F={layer['fidelity']:.3f}")
    return rows


def _aps_style():
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 600,
    })


def plot_benchmark(rows, outfile="figures/benchmark_adapt_vs_layerwise"):
    import os
    _aps_style()
    t = [r["t"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)

    ax[0].plot(t, [r["adapt_cnot"] for r in rows], "o-", color="#D55E00",
               label="ADAPT-VQA")
    ax[0].plot(t, [r["layer_cnot"] for r in rows], "s-", color="#0072B2",
               label="Layer-wise (this work)")
    ax[0].set_xlabel(r"Target complexity (Floquet step $t$)")
    ax[0].set_ylabel("CNOT count to reach $F\\geq0.99$")
    ax[0].grid(True, ls=":", alpha=0.5); ax[0].legend(loc="best")
    ax[0].text(0.035, 0.965, "(a)", transform=ax[0].transAxes, va="top",
               fontweight="bold")

    ax[1].plot(t, [r["adapt_cyc"] for r in rows], "o-", color="#D55E00",
               label="ADAPT-VQA")
    ax[1].plot(t, [r["layer_cyc"] for r in rows], "s-", color="#0072B2",
               label="Layer-wise (this work)")
    ax[1].set_xlabel(r"Target complexity (Floquet step $t$)")
    ax[1].set_ylabel("Classical optimization cycles")
    ax[1].grid(True, ls=":", alpha=0.5); ax[1].legend(loc="best")
    ax[1].text(0.035, 0.965, "(b)", transform=ax[1].transAxes, va="top",
               fontweight="bold")

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile + ".pdf", bbox_inches="tight")
    fig.savefig(outfile + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}.pdf / {outfile}.png")


if __name__ == "__main__":
    rows = run_benchmark(N=4, k=2.5, target_steps=(1, 2, 3, 4, 5))
    plot_benchmark(rows)