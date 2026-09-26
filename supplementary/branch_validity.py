"""Supplementary Note 1: distance of eig(U_F) from the branch cut of the principal log (Fig. S1).
Run from the repository root:  python supplementary/branch_validity.py
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qkt_quantum import floquet_U_exact
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ks = [0.5,1.0,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,4.0]
def dist(N,k): 
    lam = np.linalg.eigvals(floquet_U_exact(N,k,np.pi/2)); return float(np.min(np.pi-np.abs(np.angle(lam))))
out = {"k": ks, "min_branch_distance_rad": {str(N): [dist(N,k) for k in ks] for N in [4,6,8,10]}}
fine_k = np.arange(0.5,4.0001,0.05); fine = [dist(6,k) for k in fine_k]
os.makedirs("figures", exist_ok=True)
json.dump(out, open("figures/supp_note1_branch.json","w"), indent=1)
fig,ax=plt.subplots(figsize=(6,3.4))
for N,m in zip([4,6,8,10],['o','s','^','D']):
    ax.plot(ks,out["min_branch_distance_rad"][str(N)],m+'-',ms=4,lw=1,label=f'$N={N}$')
ax.plot(fine_k,fine,'-',color='0.6',lw=0.8,label='$N=6$ (fine, $\\Delta k=0.05$)',zorder=0)
ax.set_yscale('log'); ax.set_xlabel('Kick strength $k$'); ax.set_ylabel(r'$\min_j |\pi-|\arg\lambda_j||$  (rad)')
ax.axhline(1e-3,color='r',ls=':',lw=0.8); ax.grid(True,ls=':',alpha=0.5); ax.legend(fontsize=7,ncol=3)
fig.tight_layout(); fig.savefig("figures/figS1_branch_distance.png",dpi=300); fig.savefig("figures/figS1_branch_distance.pdf")
i=int(np.argmin(fine)); print(f"closest approach: N=6, k={fine_k[i]:.2f}, {fine[i]:.2e} rad"); print("saved figures/figS1_branch_distance.*")
