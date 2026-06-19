import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import os

def build_qkt_floquet_step(N, k, p):
    qc = QuantumCircuit(N)
    
    theta = k / (N / 2)
    for i in range(N):
        for j in range(i + 1, N):
            qc.rzz(theta, i, j)
            
    for i in range(N):
        qc.ry(p, i)
        
    return qc

def simulate_scaling_mps(N, k, p, steps):
    print(f"  -> Simulating N={N}, k={k}...")
    simulator = AerSimulator(method='matrix_product_state')
    s2_values = []
    
    subsystem_A = list(range(N // 2))
    
    for t in range(1, steps + 1):
        qc = QuantumCircuit(N)
        
        for i in range(N):
            qc.ry(np.pi / 4, i)
        
        step_circ = build_qkt_floquet_step(N, k, p)
        for _ in range(t):
            qc.compose(step_circ, inplace=True)
            
        qc.save_density_matrix(subsystem_A)
        
        result = simulator.run(qc).result()
        rho_a = result.data()['density_matrix']
        
        purity = np.trace(np.dot(rho_a, rho_a)).real
        s2 = -np.log(purity)
        s2_values.append(s2)
        
    return s2_values

if __name__ == "__main__":
    N_list = [12, 16]
    k_vals = [0.5, 2.5] 
    p_val = np.pi / 2
    max_steps = 15

    results = {}
    for N in N_list:
        for k in k_vals:
            s2 = simulate_scaling_mps(N, k, p_val, max_steps)
            results[(N, k)] = s2

    plt.figure(figsize=(9, 6))
    colors = {0.5: '#1f77b4', 2.5: '#ff7f0e'} 
    styles = {12: '--', 16: '-'}
    markers = {12: 'o', 16: 's'}

    for (N, k), s2 in results.items():
        regime = "Regular" if k == 0.5 else "Chaotic"
        label = f"N={N}, {regime} (k={k})"
        plt.plot(range(1, max_steps + 1), s2, 
                 color=colors[k], linestyle=styles[N], marker=markers[N], 
                 linewidth=2, markersize=6, label=label)

    plt.xlabel('Floquet Step $t$', fontsize=12)
    plt.ylabel('Bipartite Rényi-2 Entropy $S_2$', fontsize=12)
    plt.title('Finite-Size Scaling of Entanglement Entropy (MPS)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/finite_size_scaling_mps.pdf'
    plt.savefig(save_path, dpi=300)
    # The .tex includes the .png form of this figure, which the original
    # script never wrote -- so Fig. 14 was stale/missing on recompile. Save
    # the PNG too.
    plt.savefig('figures/finite_size_scaling_mps.png', dpi=300)
    print(f"\nSuccess! Plot saved to {save_path} (and .png)")