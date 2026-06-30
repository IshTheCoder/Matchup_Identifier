"""Forward-coned, attribute-centered PER-PLAYER spatial-field model.
Usage: python3 qb_cone.py [n_weeks]   (default 8; use 1 for a fast smoke test)
Reports cone width (one-sidedness), a cone-only R^2 sanity check, and the attribute-centered
per-player charge rating u_j (shrunk toward position/height/weight archetype)."""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import feature_engineering as fe
from models import QBConeFieldModel

W = int(sys.argv[1]) if len(sys.argv) > 1 else 8
print(f"building {W}-week design ...", flush=True)
data, enc = fe.build_qb_force_design_from_files(weeks=range(W))
a = data["a_qb"]; Rc = data["rusher_covariates"]
print(f"  N={len(a):,}  N_rushers={data['N_rushers']}  attr-dim={Rc.shape[1]}", flush=True)
pickle.dump((data, enc), open(f"qb_force_design_{W}wk.pkl", "wb"))

uhat, dist, rvel = data["uhat"], data["dist"], data["rvel"]
aclose, rmask, sid = data["aclose"], data["rmask"], data["rusher_slot_id"]
softplus = lambda x: np.logaddexp(0.0, x)
r2 = lambda m: 1 - ((a - m) ** 2).sum() / ((a - a.mean(0)) ** 2).sum()
vmag = np.linalg.norm(rvel, axis=-1)
vhat = rvel / (vmag[..., None] + 0.5)
cos = (uhat * vhat).sum(-1)
sw = vmag / (vmag + 1.0)


def field_mean(s, u=None):
    P = {k: float(np.asarray(s[k]).mean()) for k in ["ell0", "ell_v", "kappa", "c0", "c_a", "c_rad"]}
    A = 1.0 - sw * (1.0 - 1.0 / (1.0 + np.exp(-P["kappa"] * cos)))
    ell = P["ell0"] + P["ell_v"] * vmag
    rho = np.exp(-0.5 * (dist / ell) ** 2)
    arg = P["c0"] + P["c_a"] * np.maximum(aclose, 0)
    if u is not None:
        arg = arg + u[sid]
    fmag = softplus(arg) * rho * A * rmask
    vec = rvel + P["c_rad"] * uhat                  # force along motion (+ radial contact)
    return np.einsum("nr,nrd->nd", fmag, vec) + np.asarray(s["drift"]).mean(0), P


# -------- cone-only sanity check (no per-player) --------
s0 = QBConeFieldModel(profile="gauss", use_player=False).run_svi_inference(data, num_steps=8000, lr=5e-3)
m0, P0 = field_mean(s0)
print(f"\n  [cone-only]  R^2={r2(m0):.3f}  ell0={P0['ell0']:.2f}+{P0['ell_v']:.2f}*v yd  c_rad={P0['c_rad']:.2f}  kappa={P0['kappa']:.2f} "
      f"(behind={100/(1+np.exp(P0['kappa']))*(1+np.exp(-P0['kappa'])):.1f}%)  c_a={P0['c_a']:+.2f}", flush=True)

# -------- attribute-centered per-player --------
s = QBConeFieldModel(profile="gauss", use_player=True).run_svi_inference(data, num_steps=12000, lr=5e-3)
pickle.dump(s, open(f"qb_cone_samples_{W}wk.pkl", "wb"))
aR = np.asarray(s["rusher_weight"]); z = np.asarray(s["z_rusher"]); tau = np.asarray(s["tau"])
u_draws = aR @ Rc.T + tau[:, None] * z                 # (draws, N_rushers) attribute-centered log-charge
u_mean = u_draws.mean(0)
m, P = field_mean(s, u_mean)
print(f"  [per-player] R^2={r2(m):.3f}  ell0={P['ell0']:.2f}+{P['ell_v']:.2f}*v yd  c_rad={P['c_rad']:.2f}  kappa={P['kappa']:.2f} "
      f"(behind={100/(1+np.exp(P['kappa']))*(1+np.exp(-P['kappa'])):.1f}%)  tau={float(tau.mean()):.3f}", flush=True)

players = pd.read_csv("data/players.csv").set_index("nflId")
inv = {v: k for k, v in enc["rusher"].items()}
nG = data["N_rushers"]; cnt = np.zeros(nG); np.add.at(cnt, sid[rmask > 0], 1.0)
rate = pd.DataFrame({"nflId": [inv[i] for i in range(nG)], "frames": cnt, "u": u_mean,
                     "lo": np.percentile(u_draws, 2.5, 0), "hi": np.percentile(u_draws, 97.5, 0)})
rate["name"] = rate["nflId"].map(players["displayName"]); rate["pos"] = rate["nflId"].map(players["officialPosition"])
rate = rate[rate["frames"] >= 200].sort_values("u", ascending=False).reset_index(drop=True)
rate.to_csv(f"qb_cone_rating_{W}wk.csv", index=False)
fmt = lambda d: "\n".join(f"   {r['name']:22s} {r['pos']:3s} u={r['u']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}]"
                          f"  ({int(r['frames'])} fr)" for _, r in d.iterrows())
print(f"\n=== TOP 15 by attribute-centered charge u_j (>=200 fr, n={len(rate)}) ===\n{fmt(rate.head(15))}")
print(f"\n=== BOTTOM 6 ===\n{fmt(rate.tail(6))}")
for grp, ps in [("Edge", ["DE", "OLB"]), ("Interior", ["DT", "NT"])]:
    print(f"  {grp}: " + ", ".join(f"{r['name']} {r['u']:+.2f}" for _, r in rate[rate.pos.isin(ps)].head(5).iterrows()))
print(f"\nwrote qb_cone_samples_{W}wk.pkl, qb_cone_rating_{W}wk.csv")
