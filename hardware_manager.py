"""
hardware_manager.py  --  IBM hardware interface.

CHANGES FROM ORIGINAL:
  (x) evaluate_observables_on_hardware() now accepts a resilience_level
      parameter so that hardware_trigger_poc.py can call it twice --
      once with resilience_level=1 (TREX, mitigated) and once with
      resilience_level=0 (raw) -- to record both values as required by
      reviewer item x / M8.

  All other functions are unchanged.
"""
import json
from datetime import datetime, timezone

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import (QiskitRuntimeService, Session,
                                     EstimatorV2 as Estimator)
    _HAS_QISKIT = True
except Exception:
    _HAS_QISKIT = False


def _require_qiskit():
    if not _HAS_QISKIT:
        raise ImportError(
            "Qiskit not installed. Run `pip install qiskit qiskit-ibm-runtime`."
        )


def floquet_qiskit_circuit(N, k, p):
    _require_qiskit()
    from qkt_quantum import floquet_gate_sequence
    qc = QuantumCircuit(N)
    for gate in floquet_gate_sequence(N, k, p):
        kind, where, angle = gate
        if kind == "ry":
            qc.ry(angle, where)
        elif kind == "rzz":
            i, j = where
            qc.rzz(angle, i, j)
        elif kind == "gphase":
            qc.global_phase += -angle
    return qc


def get_service():
    _require_qiskit()
    # Let Qiskit automatically use your saved local credentials
    return QiskitRuntimeService(channel="ibm_quantum_platform")
def transpile_to_backend(circuit, backend_name, optimization_level=3, seed=42):
    _require_qiskit()
    service = get_service()
    backend = service.backend(backend_name)
    tqc = transpile(circuit, backend=backend,
                    optimization_level=optimization_level, seed_transpiler=seed)
    two_qubit = sum(1 for inst in tqc.data if inst.operation.num_qubits == 2)
    return tqc, two_qubit


def evaluate_observables_on_hardware(pubs, backend_name="ibm_fez",
                                      resilience_level=1):
    """Submit PUBs to IBM hardware and return results.

    Parameters
    ----------
    resilience_level : int
        0 = raw (no mitigation), 1 = TREX readout mitigation (default).
        Pass 0 to get unmitigated values for comparison (reviewer item x/M8).
    """
    _require_qiskit()
    service = get_service()
    backend = service.backend(backend_name)
    print(f"Submitting to '{backend_name}' "
          f"(resilience_level={resilience_level})...")
    estimator = Estimator(mode=backend)
    estimator.options.resilience_level = resilience_level
    job = estimator.run(pubs)
    print(f"Job ID: {job.job_id()}. Awaiting result...")
    result = job.result()
    print("Complete.")
    return result


def dump_backend_telemetry(backend_name, outfile="hardware_telemetry.json"):
    _require_qiskit()
    service = get_service()
    backend = service.backend(backend_name)
    props = backend.properties()
    config = backend.configuration()

    qubits = []
    for q in range(config.n_qubits):
        try: freq = props.frequency(q) / 1e9
        except Exception: freq = None
        try: t1_val = props.t1(q) * 1e6
        except Exception: t1_val = None
        try: t2_val = props.t2(q) * 1e6
        except Exception: t2_val = None
        try: ro_err = props.readout_error(q)
        except Exception: ro_err = None
        qubits.append({
            "qubit": q, "T1_us": t1_val, "T2_us": t2_val,
            "readout_error": ro_err, "frequency_GHz": freq,
        })

    two_q_errors = []
    for gate in props.gates:
        if len(gate.qubits) == 2:
            err = next((p.value for p in gate.parameters
                        if p.name == "gate_error"), None)
            two_q_errors.append({"gate": gate.gate,
                                  "qubits": list(gate.qubits),
                                  "gate_error": err})

    telemetry = {
        "_provenance": {
            "source": "REAL backend.properties()",
            "backend": backend_name,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        },
        "coupling_map": config.coupling_map,
        "basis_gates": config.basis_gates,
        "qubits": qubits,
        "two_qubit_gate_errors": two_q_errors,
    }
    with open(outfile, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"Wrote telemetry for '{backend_name}' to {outfile}")
    return telemetry
