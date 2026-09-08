from adaptive_vqs_qkt import run_adaptive_floquet
import numpy as np

reg15 = run_adaptive_floquet(6, k=0.5, steps=12, use_rk4=True, dt=1/15)
reg30 = run_adaptive_floquet(6, k=0.5, steps=12, use_rk4=True, dt=1/30)
cha15 = run_adaptive_floquet(6, k=2.5, steps=12, use_rk4=True, dt=1/15)
cha30 = run_adaptive_floquet(6, k=2.5, steps=12, use_rk4=True, dt=1/30)

print("reg maxΔF:", np.max(np.abs(np.array(reg15['residual_ref'])-np.array(reg30['residual_ref']))))
print("cha maxΔF:", np.max(np.abs(np.array(cha15['residual_ref'])-np.array(cha30['residual_ref']))))