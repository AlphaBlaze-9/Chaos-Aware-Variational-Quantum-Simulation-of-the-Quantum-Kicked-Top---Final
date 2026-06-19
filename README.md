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
| Fig. 7, 8, 13 — adaptive VQS, fidelity, J²_z | `python adaptive_vqs_qkt.py` |
| Fig. 9 — condition number vs depth | `python tensor_analysis.py` |
| Fig. 10 — D_max vs k | `python plot_dmax_vs_k.py` |
| Fig. 11 — ADAPT-VQA benchmark | `python generate_benchmarks.py` |
| Fig. 12 — D_max vs N | `python large_scale_scaling.py` |
| Fig. 14 — MPS finite-size scaling | `python finite_size_scaling_mps.py` |

All figures save to `figures/`.

## What each file does

| File | Description |
|---|---|
| `spin_operators.py` | Collective spin operators and spin-coherent states |
| `qkt_quantum.py` | Exact Floquet operator U_F and native-gate decomposition |
| `vqs.py` | McLachlan A, C, residual r², Tikhonov regularization |
| `adaptive_vqs_qkt.py` | Adaptive-depth VQS simulation, main results |
| `classical_kicked_top.py` | Classical map, FTLE, Poincaré sections |
| `shadow_estimator.py` | Classical shadow estimation of the McLachlan residual |
| `adapt_vqa_baseline.py` | ADAPT-VQA and layer-wise state prep with CNOT accounting |
| `error_mitigation.py` | ZNE and readout error inversion |
| `simulation.py` | Cirq noise sandbox (needs Cirq) |
| `hardware_manager.py` | IBM hardware interface and telemetry dump (needs Qiskit) |
| `hardware_telemetry.json` | Placeholder — populate with real device data, do not cite defaults |

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
