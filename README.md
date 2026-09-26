# Variational Quantum Simulation of the Quantum Kicked Top: Circuit-Depth Resource Scaling Across the Regular-to-Chaotic Transition

Code and data for the paper by Samarth Muralidhara (submitted to *AVS Quantum Science*, 2026). Core numerics are pure NumPy/SciPy. Cirq and Qiskit are only needed for the noise sandbox and the hardware interface.

## Install

```bash
pip install -r requirements.txt
# optional:
pip install cirq
pip install qiskit qiskit-ibm-runtime
```

## Reproducing the figures (main text)

| Figure | Script |
|---|---|
| Fig. 1 — classical-shadow convergence (N=8, D=4) | `python shadow_estimator.py` |
| Fig. 2 — Husimi Q function | `python main.py` |
| Figs. 3, 4 — Poincaré sections, FTLE vs k | `python run_classical_analysis.py` |
| Fig. 5 — OTOCs | `python otoc.py` |
| Fig. 6 — Loschmidt echo | `python loschmidt.py` |
| Fig. 7 — fixed-depth fidelity (direct optimization) | `python fig7_fidelity_compare.py` |
| Fig. 8 — adaptive-loop residual / depth / entropy / κ(A) | `python adaptive_vqs_qkt.py` |
| Fig. 9 — condition number vs depth | `python tensor_analysis.py` |
| Fig. 10 — D_max vs k (Step 1); **Table II** (Step 2); Supp. Note 3 architecture test (Step 3b) | `python plot_dmax_vs_k.py` |
| Fig. 11, Table I (HEA columns) — ⟨D⟩_t vs N, incl. k=1.5 confound check | `python large_scale_scaling.py` → `figures/depth_scaling_results.json` |
| Table I (symmetric-subspace columns) | `python symmetric_ansatz_test.py` → `symmetric_ansatz_results.json` |
| Fig. 12 — ⟨J_z²⟩ tracking with surrogate noise model | `python observable_jz2.py` |
| Fig. 13 — MPS finite-size scaling | `python finite_size_scaling_mps.py` |
| Sec. III G 1 control (r² at optimized parameters) | `python residual_at_optimum.py 2.5` → `residual_at_optimum_2.5.json` |
| Eq. (19) hardware values | archived in `hw_trial_results.json` (script: `hardware_trigger_poc.py`) |

## Reproducing the Supplementary Material

| Item | Source |
|---|---|
| Supp. Note 1 — branch validity of H_eff = i log(U_F), Fig. S1 | `python supplementary/branch_validity.py` |
| Supp. Note 2 — integrator convergence, Table 1, Fig. S2; κ_max sweep, Fig. S3 | `python check_convergence.py`; `python adaptive_vqs_qkt.py` |
| Supp. Table 2 — all fixed simulation parameters | every entry names the script that sets it |
| Supp. Note 3 — NN vs all-to-all D_max, Fig. S4, Table 3 | `python plot_dmax_vs_k.py` (Step 3b) |
| Supp. Note 4 — ADAPT-VQA comparison, Fig. S5 | `python generate_benchmarks.py`; Pauli-pool fractions: `python verify_pauli_count.py` |
| Supp. Table 4 — noise-model rates and ibm_fez calibration medians | `observable_jz2.py` (rates); `hardware_telemetry.json` (calibration, retrieved 2026-06-13) |
| Fig. S6 — shadow convergence at N=6, D=8 | `python -c "from shadow_estimator import run_shadow_convergence_test as r; r(N=6, D=8, outfile='figures/shadow_convergence_N6_D8')"` |

All figures save to `figures/`.

## Reproducing Table II (ε_opt = 0.03 sensitivity)

Table II reports the worst-case sufficient depth D_max at ε_opt = 0.03 for six kick strengths. The three chaotic values (k = 2.5, 3.0, 3.5) are cached in `plot_dmax_vs_k.py` Step 2 from an earlier run. The three regular-plateau values (k = 0.5, 1.0, 1.5) are computed by the same `dmax_direct()` routine (N=6, 12 Floquet steps, 50 seeded L-BFGS-B restarts per depth) and can be run either as part of `plot_dmax_vs_k.py` Step 2 or, on a laptop, one k at a time in parallel:

```bash
python run_eps003_regular.py 0.5    # ~1-2 h on one core
python run_eps003_regular.py 1.5
python run_eps003_regular.py 1.0
```

Each run writes `eps003_regular_result.json` incrementally (D_max, ceiling flag, and the per-(t,D) restart-success fractions). Because `dmax_direct()` seeds every restart with `default_rng((int(1000k), t, D, r))`, the result is deterministic for a fixed NumPy/SciPy version.

## What each file does

| File | Description |
|---|---|
| `spin_operators.py` | Collective spin operators and spin-coherent states |
| `qkt_quantum.py` | Exact Floquet operator U_F and native-gate decomposition |
| `vqs.py` | McLachlan A, C, residual r², Tikhonov regularization (bisection to a target condition number, `adaptive_ridge()` — Eq. 13) |
| `adaptive_vqs_qkt.py` | Adaptive-depth VQS simulation (Fig. 8, κ_max sweep) |
| `classical_kicked_top.py` | Classical map, FTLE, Poincaré sections |
| `shadow_estimator.py` | Classical-shadow estimation of the McLachlan residual (Fig. 1, Fig. S6) and Pauli-weight / shadow-norm analysis |
| `plot_dmax_vs_k.py` | D_max via direct L-BFGS-B optimization: k-sweep (Fig. 10), ε_opt=0.03 sensitivity (Table II), architecture test (Supp. Note 3) |
| `run_eps003_regular.py` | Standalone per-k runner for the Table II regular-plateau points |
| `large_scale_scaling.py` | ⟨D⟩_t vs N with adjoint gradients (Fig. 11, Table I) |
| `symmetric_ansatz_test.py` | Symmetric-subspace control ansatz (Table I) |
| `residual_at_optimum.py` | r² evaluated at directly optimized parameters (Sec. III G 1) |
| `observable_jz2.py` | ⟨J_z²⟩ tracking with the per-layer surrogate noise model (Fig. 12) |
| `adapt_vqa_baseline.py` | ADAPT-VQA and layer-wise state prep with CNOT accounting (Supp. Note 4); also houses exploratory basin-hopping CI and barren-plateau variance tooling not cited as numbers in the paper |
| `error_mitigation.py` | ZNE and readout error inversion |
| `simulation.py` | Cirq noise sandbox (needs Cirq) |
| `hardware_manager.py` | IBM hardware interface and telemetry dump (needs Qiskit) |
| `hardware_telemetry.json` | Real `ibm_fez` calibration retrieved via `dump_backend_telemetry()`; see `_provenance` for the timestamp |
| `hw_trial_results.json` | The 20 per-trial hardware expectation values behind Eq. (19) |

## Reproducibility notes

- **Bit-reproducible on a fixed NumPy/SciPy version:** every stochastic quantity is seeded (restart seeds `(int(1000k), t, D, r)` in `dmax_direct`; seed 7 in `residual_at_optimum.py`; seed 42 in `shadow_estimator.py`).
- **Optimizer sensitivity across library versions:** best-of-n L-BFGS-B restarts can settle in a different local minimum when the underlying linear algebra differs between NumPy/SciPy builds. Two quantities in the paper are affected at the second decimal only: the Sec. III G 1 control values (0.61 / 0.20 / 0.12 at t=1,2,3; an earlier archived environment gave 0.55 / 0.20 / 0.11 at the same accepting depths D=3,4,5) and the exact crossing M* in Fig. 1 (517,947). Accepting depths and all conclusions are unchanged. Rerunning `python residual_at_optimum.py 2.5` under NumPy 2.4.4 / SciPy 1.17.1 reproduces the paper's 0.61 / 0.20 / 0.12; the archived `residual_at_optimum_2.5.json` predates that environment and should be regenerated with the same command so the released data matches the manuscript.
- **Numbers that reproduce exactly from archived data files:** Table I (both halves), the Welch t-tests / CIs of Sec. III I (`figures/depth_scaling_results.json`), the FTLE values of Sec. III C (`figures/ftle_vs_k.csv`), and Eq. (19) (`hw_trial_results.json`, sample standard deviation).
- The all-to-all entangler test in `plot_dmax_vs_k.py` Step 3b is run with a 600 s per-k time budget; `figures/architecture_test.png` (Step 3, the NNN entangler) is a disconnected-graph artifact at N=6 and is not cited in the paper — `figures/architecture_test_all_to_all.png` is the result reported in Supplementary Note 3.

## Hardware (optional)

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")

import hardware_manager as h
h.dump_backend_telemetry("ibm_fez")  # writes real calibration to hardware_telemetry.json
```

## Citation

```
@article{muralidhara2026vqs,
  title  = {Variational Quantum Simulation of the Quantum Kicked Top: Circuit-Depth Resource Scaling Across the Regular-to-Chaotic Transition},
  author = {Muralidhara, Samarth},
  year   = {2026},
  note   = {Code and data: Zenodo DOI 10.5281/zenodo.21542719}
}
```
