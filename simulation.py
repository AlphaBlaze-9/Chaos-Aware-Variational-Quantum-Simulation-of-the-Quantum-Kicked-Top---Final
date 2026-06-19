

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

try:
    import cirq
    _HAS_CIRQ = True
except Exception:  
    _HAS_CIRQ = False


def _require_cirq():
    if not _HAS_CIRQ:
        raise ImportError("Cirq not installed. Run `pip install cirq` to use simulation.py.")





def create_ansatz(qubits, depth, params):
    
    circuit = cirq.Circuit()
    n_qubits = len(qubits)
    param_idx = 0
    
    
    for q in qubits:
        if param_idx < len(params):
            circuit.append(cirq.ry(params[param_idx])(q))
            param_idx += 1
            
    
    for d in range(depth):
        
        for i in range(0, n_qubits - 1, 2):
            circuit.append(cirq.CZ(qubits[i], qubits[i+1]))
        for i in range(1, n_qubits - 1, 2):
            circuit.append(cirq.CZ(qubits[i], qubits[i+1]))
            
        
        for q in qubits:
            if param_idx < len(params):
                circuit.append(cirq.ry(params[param_idx])(q))
                circuit.append(cirq.rz(params[param_idx+1])(q))
                param_idx += 2
                
    return circuit

def count_required_params(n_qubits, depth):
    return n_qubits + depth * (2 * n_qubits)





def add_noise_to_circuit(circuit, p_depol, gamma_amp_damp=0.0):
    
    noisy_ops = []
    
    for moment in circuit:
        for op in moment:
            noisy_ops.append(op)
            for q in op.qubits:
                if p_depol > 0:
                    noisy_ops.append(cirq.depolarize(p_depol).on(q))
                if gamma_amp_damp > 0:
                    noisy_ops.append(cirq.amplitude_damp(gamma_amp_damp).on(q))
                    
    return cirq.Circuit(noisy_ops)

def get_density_matrix(circuit):
    
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.simulate(circuit)
    return result.final_density_matrix

def compute_fidelity(rho_noisy, rho_clean):
    
    
    sqrt_noisy = sqrtm(rho_noisy)
    
    
    product = sqrt_noisy @ rho_clean @ sqrt_noisy
    
    
    sqrt_product = sqrtm(product)
    
    
    trace = np.trace(sqrt_product)
    fidelity = (trace.real)**2 
    return fidelity

def compute_purity(rho):
    
    rho_sq = rho @ rho
    return np.trace(rho_sq).real


def compute_renyi2(rho):
    
    purity = compute_purity(rho)
    purity = min(max(float(purity), 1e-15), 1.0)
    return float(-np.log2(purity))





def run_noise_sweep_comparison():
    print("\n--- 1. Noise Effects: Baseline vs Adaptive VQS ---")
    
    n_qubits = 4
    qubits = cirq.LineQubit.range(n_qubits)
    noise_levels = [0, 0.001, 0.005, 0.01, 0.02, 0.05]
    
    depth_baseline = 2
    depth_adaptive = 4
    
    np.random.seed(42)
    params_base = np.random.uniform(0, 2*np.pi, count_required_params(n_qubits, depth_baseline))
    params_adapt = np.random.uniform(0, 2*np.pi, count_required_params(n_qubits, depth_adaptive))
    
    
    rho_base_clean = get_density_matrix(create_ansatz(qubits, depth_baseline, params_base))
    rho_adapt_clean = get_density_matrix(create_ansatz(qubits, depth_adaptive, params_adapt))
    
    fidelities_base = []
    fidelities_adapt = []
    
    for p in noise_levels:
        gamma = p 
        
        
        noisy_c_base = add_noise_to_circuit(create_ansatz(qubits, depth_baseline, params_base), p, gamma)
        rho_base = get_density_matrix(noisy_c_base)
        fidelities_base.append(compute_fidelity(rho_base, rho_base_clean))
        
        
        noisy_c_adapt = add_noise_to_circuit(create_ansatz(qubits, depth_adaptive, params_adapt), p, gamma)
        rho_adapt = get_density_matrix(noisy_c_adapt)
        fidelities_adapt.append(compute_fidelity(rho_adapt, rho_adapt_clean))
        
        print(f"Noise p={p:<6}: Fid_Base={fidelities_base[-1]:.4f}, Fid_Adapt={fidelities_adapt[-1]:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(noise_levels, fidelities_base, 'o--', label=f'Baseline (Depth {depth_baseline})')
    plt.plot(noise_levels, fidelities_adapt, 's-', label=f'Adaptive (Depth {depth_adaptive})')
    plt.xlabel('Noise Probability (p)')
    plt.ylabel('Fidelity relative to Noiseless')
    plt.title('VQS Reliability: Baseline vs Deep Circuits')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def compute_noisy_otoc_value(qubits, depth, params, p_noise):
    
    _require_cirq()
    circuit_fwd = create_ansatz(qubits, depth, params)
    circuit_rev = circuit_fwd**-1
    
    full_ops = []
    full_ops.append(add_noise_to_circuit(circuit_fwd, p_noise))
    full_ops.append(cirq.X(qubits[0])) 
    full_ops.append(add_noise_to_circuit(circuit_rev, p_noise))
    
    otoc_circuit = cirq.Circuit(full_ops)
    rho_final = get_density_matrix(otoc_circuit)
    
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    V_matrix = Z
    for _ in range(len(qubits) - 1):
        V_matrix = np.kron(V_matrix, I)
        
    expectation = np.trace(rho_final @ V_matrix).real
    return expectation

def run_chaos_diagnostics():
    print("\n--- 2. Chaos Signatures: OTOC & Purity Decay ---")
    
    n_qubits = 4
    qubits = cirq.LineQubit.range(n_qubits)
    depth = 4 
    
    np.random.seed(999)
    params = np.random.uniform(0, 2*np.pi, count_required_params(n_qubits, depth))
    
    noise_levels = [0, 0.002, 0.005, 0.01, 0.02, 0.05]
    otoc_values = []
    purities = []
    
    for p in noise_levels:
        val = compute_noisy_otoc_value(qubits, depth, params, p)
        otoc_values.append(val)
        
        circ_fwd = add_noise_to_circuit(create_ansatz(qubits, depth, params), p)
        rho_fwd = get_density_matrix(circ_fwd)
        purities.append(compute_purity(rho_fwd))

        print(f"Noise p={p:<6}: OTOC_proxy={val:.4f}, Purity={purities[-1]:.4f}, "
              f"S2={compute_renyi2(rho_fwd):.4f}")
        
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('Noise Probability (p)')
    ax1.set_ylabel('OTOC Value (Correlation)', color=color)
    ax1.plot(noise_levels, otoc_values, 'o-', color=color, label='OTOC (Z0)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Purity Tr(rho^2)', color=color)  
    ax2.plot(noise_levels, purities, 's--', color=color, label='Purity')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Classicalization: OTOC Suppression & Entropy Decay')
    plt.show()





def zero_noise_extrapolation_demo():
    print("\n--- 3. Error Mitigation: Zero-Noise Extrapolation (ZNE) ---")
    
    n_qubits = 2
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.rx(0.5)(qubits[0])
    )
    
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    Z0_matrix = np.kron(Z, I)
    
    def measure_z0(rho):
        return np.trace(rho @ Z0_matrix).real

    rho_clean = get_density_matrix(circuit)
    true_val = measure_z0(rho_clean)
    print(f"True Expectation Value: {true_val:.5f}")
    
    base_p = 0.02
    scale_factors = [1.0, 2.0, 3.0]
    measured_vals = []
    
    for r in scale_factors:
        p_scaled = base_p * r
        rho_noisy = get_density_matrix(add_noise_to_circuit(circuit, p_scaled))
        val = measure_z0(rho_noisy)
        measured_vals.append(val)
        print(f"Scale r={r}, p={p_scaled:.3f}, Measured={val:.5f}")
        
    params = np.polyfit(scale_factors, measured_vals, 1)
    mitigated_val = params[1]
    
    print(f"Unmitigated (r=1): {measured_vals[0]:.5f}")
    print(f"ZNE Mitigated (r=0): {mitigated_val:.5f}")
    print(f"Error Improvement: {abs(true_val - measured_vals[0]) - abs(true_val - mitigated_val):.5f}")

    plt.figure(figsize=(8, 5))
    plt.plot([0] + scale_factors, [true_val] + measured_vals, 'o', label='Simulated Data')
    x_range = np.linspace(0, 3.5, 10)
    plt.plot(x_range, params[0]*x_range + params[1], '--', alpha=0.7, label='Linear Fit')
    plt.scatter([0], [true_val], c='green', marker='*', s=200, label='True Value', zorder=10)
    plt.scatter([0], [mitigated_val], c='red', marker='x', s=100, label='Extrapolated', zorder=10)
    plt.xlabel('Noise Scale Factor (r)')
    plt.ylabel('<Z0>')
    plt.title('Zero-Noise Extrapolation (ZNE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    
    
    zero_noise_extrapolation_demo()