"""Fit the improved spatial-field model (exp vs Gaussian radial profile, + closing-speed charge),
compare incremental R^2 to the point-force model (0.163), and save the best for viz/attribution."""
import sys, pickle
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "model")
from models import QBSpatialFieldModel

data, enc = pickle.load(open("qb_force_design_wk0.pkl", "rb"))
a = data["a_qb"]
r2 = lambda m: 1 - ((a - m) ** 2).sum() / ((a - a.mean(0)) ** 2).sum()
softplus = lambda x: np.logaddexp(0.0, x)


def spatial_mean(s, profile):
    p = {k: float(np.asarray(s[k]).mean()) for k in ["l_perp", "l_par", "c0", "c_a", "c_int"]}
    d = data["dist"][..., None] * data["uhat"]
    v = data["rvel"]; vhat = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 0.5)
    ip2, ia2 = 1 / p["l_perp"] ** 2, 1 / p["l_par"] ** 2
    proj = (d * vhat).sum(-1)
    Sinv_d = ip2 * d + (ia2 - ip2) * proj[..., None] * vhat
    quad = ip2 * (d * d).sum(-1) + (ia2 - ip2) * proj ** 2
    q = softplus(p["c0"] + p["c_a"] * np.maximum(data["aclose"], 0) + p["c_int"] * data["is_interior"])
    if profile == "exp":
        r = np.sqrt(quad + 1e-6); psi = np.exp(-r); grad = Sinv_d / r[..., None]
    else:
        psi = np.exp(-0.5 * quad); grad = Sinv_d
    return ((q * psi * data["rmask"])[..., None] * grad).sum(1) + np.asarray(s["drift"]).mean(0)


print(f"N={len(a):,}  (point-force baseline R^2 = 0.163)\n", flush=True)
results = {}
for prof in ["exp", "gauss"]:
    s = QBSpatialFieldModel(profile=prof).run_svi_inference(data, num_steps=8000, lr=5e-3)
    R = r2(spatial_mean(s, prof)); results[prof] = (s, R)
    g = lambda k: float(np.asarray(s[k]).mean())
    print(f"  profile={prof:5s}  R^2={R:.3f}  l_par/l_perp={g('l_par')/g('l_perp'):.2f}  "
          f"c_a={g('c_a'):+.2f} c_int={g('c_int'):+.2f}  exp(c_int)={np.exp(g('c_int')):.2f}x")

best = max(results, key=lambda k: results[k][1])
print(f"\nbest profile = {best}  (R^2 {results[best][1]:.3f}  vs point-force 0.163)")
pickle.dump({"samples": results[best][0], "profile": best}, open("qb_spatial_best.pkl", "wb"))
print("wrote qb_spatial_best.pkl")
