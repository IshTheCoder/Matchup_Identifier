"""(a) Archetype split: does the mis-signed engagement gate eta1<0 just proxy interior-vs-edge?
   Refit the point force model with an interior charge term; if eta1 collapses toward 0, yes.
(b) Spatial-field model: model the QB push as the gradient of a SUM of per-rusher anisotropic
   Gaussian pressure potentials (geometry replaces the gate). Compare incremental R^2.
"""
import sys, pickle
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import feature_engineering as fe
from models import QBForceFieldModel, QBSpatialFieldModel

print("building QB force design (week 0, dropback-only) ...", flush=True)
data, enc = fe.build_qb_force_design_from_files(weeks=range(1))
a = data["a_qb"]
print(f"  N={len(a):,}  Rmax={data['Rmax']}  dose coverage={enc['dose_coverage']:.3f}", flush=True)


def r2(mean):
    return 1.0 - ((a - mean) ** 2).sum() / ((a - a.mean(0)) ** 2).sum()


def softplus(x):
    return np.logaddexp(0.0, x)


def point_mean(s, eta0=0.0, archetype=False):
    p = {k: float(np.asarray(s[k]).mean()) for k in ["eta1", "beta_v", "beta_a", "ell"]}
    gate = 1 / (1 + np.exp(-(eta0 - p["eta1"] * data["dose"])))
    Phi = np.exp(-np.maximum(data["dist"], 0.5) / p["ell"])
    Psi = p["beta_v"] * np.maximum(data["sclose"], 0) + p["beta_a"] * np.maximum(data["aclose"], 0)
    fmag = gate * Phi * Psi * data["rmask"]
    if archetype:
        fmag = fmag * np.exp(float(np.asarray(s["gamma_int"]).mean()) * data["is_interior"])
    return np.einsum("nr,nrd->nd", fmag, data["uhat"]) + np.asarray(s["drift"]).mean(0)


def spatial_mean(s):
    p = {k: float(np.asarray(s[k]).mean()) for k in ["l_perp", "l_par", "c0", "c_a"]}
    d = data["dist"][..., None] * data["uhat"]
    v = data["rvel"]; vmag = np.linalg.norm(v, axis=-1, keepdims=True)
    vhat = v / (vmag + 0.5)
    ip2, ia2 = 1 / p["l_perp"] ** 2, 1 / p["l_par"] ** 2
    proj = (d * vhat).sum(-1)
    Sinv_d = ip2 * d + (ia2 - ip2) * proj[..., None] * vhat
    quad = ip2 * (d * d).sum(-1) + (ia2 - ip2) * proj ** 2
    psi = np.exp(-0.5 * quad)
    q = softplus(p["c0"] + p["c_a"] * np.maximum(data["aclose"], 0))
    F = (q * psi * data["rmask"])[..., None] * Sinv_d
    return F.sum(1) + np.asarray(s["drift"]).mean(0)


# ============================================================ (a) archetype split
print("\n=== (a) ARCHETYPE SPLIT: is eta1<0 just interior-vs-edge? ===", flush=True)
s_base = QBForceFieldModel(kernel="exp", eta0=0.0).run_svi_inference(data, num_steps=6000, lr=5e-3)
s_arch = QBForceFieldModel(kernel="exp", eta0=0.0, use_archetype=True).run_svi_inference(data, num_steps=6000, lr=5e-3)
e_base = float(np.asarray(s_base["eta1"]).mean())
e_arch = float(np.asarray(s_arch["eta1"]).mean())
g_int = float(np.asarray(s_arch["gamma_int"]).mean())
print(f"  eta1 WITHOUT archetype control: {e_base:+.3f}  (R^2 {r2(point_mean(s_base)):.3f})")
print(f"  eta1 WITH    archetype control: {e_arch:+.3f}  (R^2 {r2(point_mean(s_arch, archetype=True)):.3f})")
print(f"  interior charge gamma_int = {g_int:+.3f}  (exp = {np.exp(g_int):.2f}x force for interior DL)")
print(f"  -> eta1 moved {e_base:+.3f} -> {e_arch:+.3f}; "
      f"{'archetype ABSORBS the gate (confound confirmed)' if abs(e_arch) < 0.5*abs(e_base) else 'gate survives archetype control'}")

# ============================================================ (b) spatial field model
print("\n=== (b) SPATIAL FIELD MODEL (anisotropic, no engagement gate) ===", flush=True)
s_sp = QBSpatialFieldModel().run_svi_inference(data, num_steps=8000, lr=5e-3)
for k in ["l_perp", "l_par", "c0", "c_a", "sigma"]:
    v = np.asarray(s_sp[k]).ravel()
    print(f"  {k:7s} mean={v.mean():+.4f}  95%=[{np.percentile(v,2.5):+.4f}, {np.percentile(v,97.5):+.4f}]")
lp, lpar = float(np.asarray(s_sp["l_perp"]).mean()), float(np.asarray(s_sp["l_par"]).mean())
R2_sp = r2(spatial_mean(s_sp))
R2_pt = r2(point_mean(s_base))
print(f"\n  anisotropy l_par/l_perp = {lpar/lp:.2f}  (>1 => field reaches further along rusher motion)")
print(f"  P(c_a>0) = {np.mean(np.asarray(s_sp['c_a'])>0):.3f}  (driving rushers carry more charge)")
print(f"\n  incremental R^2:  spatial-field {R2_sp:.3f}   vs   point-force {R2_pt:.3f}")
pickle.dump((data, enc), open("qb_force_design_wk0.pkl", "wb"))
pickle.dump(s_sp, open("qb_spatial_samples_wk0.pkl", "wb"))
print("\nwrote qb_force_design_wk0.pkl, qb_spatial_samples_wk0.pkl")
