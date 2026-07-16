# Chaos-Aware Variational Quantum Simulation of the Quantum Kicked Top

Code for the paper by Samarth Muralidhara. Core numerics are pure NumPy/SciPy. Cirq and Qiskit are only needed for the noise sandbox and hardware interface.

## Install

```bash
pip install -r requirements.txt
# optional:
pip install cirq
pip install qiskit qiskit-ibm-runtime
```

## Reproducing the figures

| Figure | Script |
|---|---|
| Fig. 1 — shadow convergence | `python shadow_estimator.py` |
| Fig. 2 — Husimi Q function | `python main.py` |
| Fig. 3, 4 — Poincaré sections, FTLE | `python run_classical_analysis.py` |
| Fig. 5 — OTOCs | `python otoc.py` |
| Fig. 6 — Loschmidt echo | `python loschmidt.py` |
| Fig. 7, 8, 12 — VQS fidelity, adaptive-loop residual/depth/entropy, J²_z | `python adaptive_vqs_qkt.py` |
| Fig. 9 — condition number vs depth | `python tensor_analysis.py` |
| Fig. 10 — D_max vs k | `python plot_dmax_vs_k.py` |
| Fig. 11 — D_max vs N (central result), incl. k=1.5 confound check | `python large_scale_scaling.py` |
| Fig. 13 — MPS finite-size scaling | `python finite_size_scaling_mps.py` |
| Fig. 14 — architecture test, NN vs all-to-all (Appendix C) | `python plot_dmax_vs_k.py` (Step 3b; also writes `figures/architecture_test.png`, the disclaimed NNN check from Step 3) |
| Fig. 15 — ADAPT-VQA benchmark (Appendix D) | `python generate_benchmarks.py` |

All figures save to `figures/`.

## What each file does

| File | Description |
|---|---|
| `spin_operators.py` | Collective spin operators and spin-coherent states |
| `qkt_quantum.py` | Exact Floquet operator U_F and native-gate decomposition |
| `vqs.py` | McLachlan A, C, residual r², Tikhonov regularization (bisection search to a target condition number, `adaptive_ridge()` — see Eq. 13 in the paper) |
| `adaptive_vqs_qkt.py` | Adaptive-depth VQS simulation, main results (Figs. 7, 8, 12) |
| `classical_kicked_top.py` | Classical map, FTLE, Poincaré sections |
| `shadow_estimator.py` | Classical shadow estimation of the McLachlan residual |
| `adapt_vqa_baseline.py` | ADAPT-VQA and layer-wise state prep with CNOT accounting; also houses the eps_opt=0.03/0.02 sensitivity sweep, basin-hopping CI, and barren-plateau variance tooling (exploratory — not cited as numbers in the paper, which defers this analysis to future work) |
| `error_mitigation.py` | ZNE and readout error inversion |
| `simulation.py` | Cirq noise sandbox (needs Cirq) |
| `hardware_manager.py` | IBM hardware interface and telemetry dump (needs Qiskit) |
| `hardware_telemetry.json` | Real `ibm_fez` calibration data retrieved via `dump_backend_telemetry()`; see the `_provenance` field for the retrieval timestamp and backend |

## Known gaps / open items

- The `eps_opt=0.03` sensitivity check (`plot_dmax_vs_k.py` Step 2, and Sec. Limitations item vi in the paper) has only ever been run on the chaotic-side kick strengths (k=2.5, 3.0, 3.5). The regular plateau (k≤1.5) has not been rerun at this tighter threshold, so the paper does not claim a sharper regular/chaotic separation from this check alone — see the paper text for the exact scoping. Rerunning `plot_dmax_vs_k.py`'s Step 2 with `k_border` extended to include 0.5, 1.0, 1.5 would close this gap.
- `figures/architecture_test.png` (from `plot_dmax_vs_k.py` Step 3, the NNN entangler) is a disconnected-graph artifact at N=6 and is explicitly disclaimed in Appendix C — it is not cited as a result. `figures/architecture_test_all_to_all.png` (Step 3b) is the actual Fig. 14 comparison.

## Hardware (optional)

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")

import hardware_manager as h
h.dump_backend_telemetry("ibm_brisbane")  # writes real calibration to hardware_telemetry.json
```

## Citation

```
@article{muralidhara2025chaos,
  title  = {Chaos-Aware Variational Quantum Simulation of the Quantum Kicked Top},
  author = {Muralidhara, Samarth},
  year   = {2025},
  note   = {arXiv:[ID]}
}
```
