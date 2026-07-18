"""Milestone: fit the structural QB force-field model on one week (dropback-only) and validate
the mechanism BEFORE building attribution / counterfactual. Reports:
  - incremental R^2  (force field vs a constant-drift baseline = fraction of QB-acceleration
    variance explained by the engagement-gated rusher forces),
  - P(eta1 > 0)   : do blockers SHIELD the QB from a rusher's push?  + gate ratio g(0)/g(2),
  - P(beta_v > 0) : do rushers BEARING DOWN push harder than standing rushers?
A go/no-go for the later attribution + counterfactual stages.
"""
import sys, pickle
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import feature_engineering as fe
from models import QBForceFieldModel

print("building QB force design (week 0, dropback-only) ...", flush=True)
data, enc = fe.build_qb_force_design_from_files(weeks=range(1))
N = len(data["a_qb"])
print(f"  QB-frames N={N:,}  Rmax={data['Rmax']}  dose coverage={enc['dose_coverage']:.3f}", flush=True)
print(f"  |a_qb| mean={np.linalg.norm(data['a_qb'],axis=1).mean():.3f} yd/s^2  "
      f"dist median={np.median(data['dist'][data['rmask']>0]):.2f} yd  "
      f"closing-speed median={np.median(data['sclose'][data['rmask']>0]):.2f} yd/s", flush=True)
pickle.dump((data, enc), open("qb_force_design_wk0.pkl", "wb"))

ETA0 = 0.0  # fixed free-rusher gate level (see QBForceFieldModel)
m = QBForceFieldModel(drift_mode="const", kernel="exp", eta0=ETA0)
samples = m.run_svi_inference(data, num_steps=8000, lr=5e-3, n_draws=1000)
pickle.dump(samples, open("qb_force_samples_wk0.pkl", "wb"))

print("\n=== posterior summaries ===", flush=True)
for k in ["eta1", "beta_v", "beta_a", "ell", "sigma"]:
    s = np.asarray(samples[k]).ravel()
    print(f"  {k:7s} mean={s.mean():+.4f}  P(>0)={np.mean(s>0):.3f}  "
          f"95%=[{np.percentile(s,2.5):+.4f}, {np.percentile(s,97.5):+.4f}]")

# ---- incremental R^2 from posterior-mean params (const drift baseline => R2(drift)=0) ----
pm = {k: float(np.asarray(samples[k]).mean()) for k in ["eta1", "beta_v", "beta_a", "ell"]}
drift = np.asarray(samples["drift"]).mean(0)
a, uhat, dist = data["a_qb"], data["uhat"], data["dist"]
sclose, aclose, dose, rmask = data["sclose"], data["aclose"], data["dose"], data["rmask"]
gate = 1.0 / (1.0 + np.exp(-(ETA0 - pm["eta1"] * dose)))
dclip = np.maximum(dist, 0.5)
Phi = np.exp(-dclip / pm["ell"])
Psi = pm["beta_v"] * np.maximum(sclose, 0) + pm["beta_a"] * np.maximum(aclose, 0)
fmag = gate * Phi * Psi * rmask
mean = np.einsum("nr,nrd->nd", fmag, uhat) + drift


def r2(y, yh):
    return 1.0 - ((y - yh) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


R2, R2x, R2y = r2(a, mean), r2(a[:, [0]], mean[:, [0]]), r2(a[:, [1]], mean[:, [1]])
g0, g2 = 1 / (1 + np.exp(-ETA0)), 1 / (1 + np.exp(-(ETA0 - pm["eta1"] * 2)))
p_eta1 = float(np.mean(np.asarray(samples["eta1"]) > 0))
p_bv = float(np.mean(np.asarray(samples["beta_v"]) > 0))
p_ba = float(np.mean(np.asarray(samples["beta_a"]) > 0))

print("\n=== MILESTONE VERDICT ===", flush=True)
print(f"incremental R^2 (force field vs const drift): {R2:.3f}   (x {R2x:.3f}, y {R2y:.3f})")
print(f"SHIELDING  P(eta1>0) = {p_eta1:.3f}   gate ratio g(0)/g(2) = {g0/g2:.2f}  (free vs doubled)")
print(f"KINEMATICS P(beta_v>0) = {p_bv:.3f}   P(beta_a>0) = {p_ba:.3f}  (bearing-down vs standing)")
GO = (p_eta1 > 0.9) and (p_bv > 0.9) and (R2 > 0.02)
print(f"\nGO for attribution + counterfactual: {GO}")
print("  (criteria: P(eta1>0)>0.9 AND P(beta_v>0)>0.9 AND incremental R^2>0.02)")
