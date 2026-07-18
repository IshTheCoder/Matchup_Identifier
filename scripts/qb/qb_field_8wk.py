"""Fit the spatial-field model on all 8 weeks and produce a stable per-rusher attribution
leaderboard (force exerted on the QB), with a snap threshold to remove low-snap blitzers."""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import feature_engineering as fe
from models import QBSpatialFieldModel

MIN_FRAMES = 200
print("building 8-week QB force design ...", flush=True)
data, enc = fe.build_qb_force_design_from_files(weeks=range(8))
a = data["a_qb"]
print(f"  N={len(a):,} QB-frames  Rmax={data['Rmax']}  dose coverage={enc['dose_coverage']:.3f}", flush=True)
pickle.dump((data, enc), open("qb_force_design_8wk.pkl", "wb"))

s = QBSpatialFieldModel(profile="gauss").run_svi_inference(data, num_steps=10000, lr=5e-3)
pickle.dump(s, open("qb_spatial_samples_8wk.pkl", "wb"))
P = {k: float(np.asarray(s[k]).mean()) for k in ["l_perp", "l_par", "c0", "c_a", "c_int"]}
softplus = lambda x: np.logaddexp(0.0, x)

uhat, dist, rvel = data["uhat"], data["dist"], data["rvel"]
aclose, is_int, rmask = data["aclose"], data["is_interior"], data["rmask"]
vhat = rvel / (np.linalg.norm(rvel, axis=-1, keepdims=True) + 0.5)
ip2, ia2 = 1 / P["l_perp"] ** 2, 1 / P["l_par"] ** 2
d = dist[..., None] * uhat
proj = (d * vhat).sum(-1)
Sinv_d = ip2 * d + (ia2 - ip2) * proj[..., None] * vhat
quad = ip2 * (d * d).sum(-1) + (ia2 - ip2) * proj ** 2
psi = np.exp(-0.5 * quad)
q = softplus(P["c0"] + P["c_a"] * np.maximum(aclose, 0) + P["c_int"] * is_int) * rmask
F = (q * psi)[..., None] * Sinv_d
Fmag = np.linalg.norm(F, axis=-1) * rmask
mean = F.sum(1) + np.asarray(s["drift"]).mean(0)
R2 = 1 - ((a - mean) ** 2).sum() / ((a - a.mean(0)) ** 2).sum()
lp, lpar = P["l_perp"], P["l_par"]
print(f"\n  R^2={R2:.3f}   l_par/l_perp={lpar/lp:.2f}   c_a={P['c_a']:+.2f}   "
      f"c_int={P['c_int']:+.2f} (exp {np.exp(P['c_int']):.1f}x)   sigma={float(np.asarray(s['sigma']).mean()):.3f}")

# -------- per-rusher attribution (mean force on QB per frame), thresholded --------
players = pd.read_csv("data/players.csv").set_index("nflId")
inv = {v: k for k, v in enc["rusher"].items()}
gid = data["rusher_slot_id"]
nG = len(enc["rusher"])
tot = np.zeros(nG); cnt = np.zeros(nG)
np.add.at(tot, gid[rmask > 0], Fmag[rmask > 0]); np.add.at(cnt, gid[rmask > 0], 1.0)
att = pd.DataFrame({"force": np.where(cnt > 0, tot / np.maximum(cnt, 1), 0), "frames": cnt,
                    "nflId": [inv[i] for i in range(nG)]})
att["name"] = att["nflId"].map(players["displayName"]); att["pos"] = att["nflId"].map(players["officialPosition"])
att = att[att["frames"] >= MIN_FRAMES].sort_values("force", ascending=False).reset_index(drop=True)
att.to_csv("qb_force_attribution_8wk.csv", index=False)
print(f"\n=== top 15 rushers by force exerted on the QB (>= {MIN_FRAMES} frames, n={len(att)}) ===")
print(att.head(15)[["name", "pos", "force", "frames"]].round(3).to_string(index=False))
print("\n=== top 5 by position group ===")
for grp, ps in [("Edge", ["DE", "OLB"]), ("Interior", ["DT", "NT"])]:
    d5 = att[att["pos"].isin(ps)].head(5)
    print(f"  {grp}: " + ", ".join(f"{r['name']} {r['force']:.3f}" for _, r in d5.iterrows()))
print("\nwrote qb_force_design_8wk.pkl, qb_spatial_samples_8wk.pkl, qb_force_attribution_8wk.csv")
