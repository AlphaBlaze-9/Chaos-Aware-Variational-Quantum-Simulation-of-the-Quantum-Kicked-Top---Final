import pickle
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Operator
from hardware_manager import evaluate_observables_on_hardware, get_service
from vqs import floquet_step_generator

def ansatz_qiskit_circuit(N, depth, theta):
    qc = QuantumCircuit(N)
    idx = 0
    
    for _ in range(depth):
        for i in range(N):
            qc.ry(theta[idx], i)
            idx += 1
            
        for i in range(N):
            qc.rz(theta[idx], i)
            idx += 1
            
        chi = theta[idx]
        idx += 1
        for i in range(N):
            jq = (i + 1) % N
            qc.rzz(chi, i, jq)
            
    return qc

def main():
    N = 4
    k = 2.5 
    p = np.pi / 2
    epsilon = 0.05
    target_backend = "ibm_fez" 

    print("--- Initiating Physical Hardware Proof of Concept ---")
    
    try:
        with open("N4_chaotic_history.pkl", "rb") as f:
            cha_poc = pickle.load(f)
    except FileNotFoundError:
        print("Error: N4_chaotic_history.pkl not found. Run adaptive_vqs_qkt.py first.")
        return

    step_idx = 0 
    theta_target = cha_poc["theta_history"][step_idx]
    current_depth = cha_poc["depth"][step_idx]
    
    print(f"Successfully loaded parameters: N={N}, Depth={current_depth}, "
          f"Total Parameters={len(theta_target)}")

    qc = ansatz_qiskit_circuit(N, current_depth, theta_target)

    H_step_matrix = floquet_step_generator(N, k, p)
    
    H_op = SparsePauliOp.from_operator(Operator(H_step_matrix))
    H_sq_op = SparsePauliOp.from_operator(Operator(H_step_matrix @ H_step_matrix))
    
    print(f"Constructed SparsePauliOp for H_step with {len(H_op)} Pauli terms.")

    try:
        service = get_service()
        backend = service.backend(target_backend)
        
        transpiled_qc = transpile(qc, backend=backend, optimization_level=3)
        
        initial_layout = transpiled_qc.layout.initial_layout
        physical_qubits = [initial_layout[q] for q in qc.qubits]
        
        H_op_mapped = H_op.apply_layout(physical_qubits, num_qubits=backend.num_qubits)
        H_sq_op_mapped = H_sq_op.apply_layout(physical_qubits, num_qubits=backend.num_qubits)
        
        pubs = [(transpiled_qc, [H_op_mapped, H_sq_op_mapped])]
        
        num_trials = 5 
        hw_H_vals = []
        hw_Hsq_vals = []
        
        print(f"\nSubmitting {num_trials} independent jobs to {target_backend} to calculate mean +/- std...")
        
        for trial in range(num_trials):
            print(f"  -> Trial {trial + 1}/{num_trials}")
            result = evaluate_observables_on_hardware(pubs, backend_name=target_backend)
            
            pub_result = result[0]
            hw_H_vals.append(pub_result.data.evs[0])
            hw_Hsq_vals.append(pub_result.data.evs[1])
            
        hw_H_mean = np.mean(hw_H_vals)
        hw_H_std = np.std(hw_H_vals)
        hw_Hsq_mean = np.mean(hw_Hsq_vals)
        hw_Hsq_std = np.std(hw_Hsq_vals)
        
        print("\n--- Hardware Execution Complete ---")
        print(f"Hardware measured <H_step>   = {hw_H_mean:.3f} +/- {hw_H_std:.3f}")
        print(f"Hardware measured <H_step^2> = {hw_Hsq_mean:.3f} +/- {hw_Hsq_std:.3f}")
        
        print(f"\nAnalysis for Peer Review:")
        print(f"The classical trigger threshold is set at epsilon = {epsilon}.")
        print("The successful retrieval of these mitigated expectation values directly "
              "from the heavy-hex processor proves that the requisite components for calculating "
              "the McLachlan residual r^2 are measurable within the constraints of NISQ hardware.")
              
    except Exception as e:
        print(f"\nHardware Execution Failed. Ensure your IBM Quantum token is active and "
              f"the target backend '{target_backend}' is available.")
        print(f"Error Details: {e}")

if __name__ == "__main__":
    main()