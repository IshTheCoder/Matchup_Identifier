"""QB-specific danger-zone escape model: each QB has a Gaussian zone (QB-specific radius), each
rusher a velocity-anisotropic density; their overlap T_k drives the per-frame release hazard.
Reports fit (vs fixed-radius escape 0.177), per-QB zone radii, and the per-player rating."""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import feature_engineering as fe
from models import QBZoneEscapeModel

W = int(sys.argv[1]) if len(sys.argv) > 1 else 8
pkl = f"qb_force_design_{W}wk.pkl"
try:
    data, enc = pickle.load(open(pkl, "rb")); assert "quarterback_ids" in data
    print(f"loaded {pkl}", flush=True)
except Exception:
    data, enc = fe.build_qb_force_design_from_files(weeks=range(W)); pickle.dump((data, enc), open(pkl, "wb"))
N = len(data["a_qb"]); ps = data["play_start"]; pr = data["play_row"]
t_since = np.arange(N) - ps[pr]
data["release"] = np.zeros(N, np.float32); data["release"][ps[1:] - 1] = 1.0
data["t_since_std"] = ((t_since - t_since.mean()) / (t_since.std() + 1e-9)).astype(np.float32)
print(f"  N={N:,}  releases={int(data['release'].sum()):,}", flush=True)

s = QBZoneEscapeModel(use_situation=True, aniso=True).run_svi_inference(data, num_steps=12000, lr=5e-3)
pickle.dump(s, open(f"qb_zone_samples_{W}wk.pkl", "wb"))
g = lambda k: float(np.asarray(s[k]).mean())
uhat, dr, rvel, rmask, sid, Rc = (data["uhat"], data["dist"], data["rvel"], data["rmask"],
                                  data["rusher_slot_id"], data["rusher_covariates"])
qb = data["quarterback_ids"]; sig = lambda x: 1 / (1 + np.exp(-x))

# QB-specific zone radius
sigq_qb = np.exp(g("log_sig0") + g("s_sig") * np.asarray(s["z_sig_qb"]).mean(0))   # (Nqb,)
sigma_q = sigq_qb[qb]                                                              # (N,)
a = sigma_q[:, None] ** 2 + g("lperp") ** 2
vmag = np.linalg.norm(rvel, axis=-1); vhat = rvel / (vmag[..., None] + 0.5)
proj = ((dr[..., None] * uhat) * vhat).sum(-1)
b = (g("lperp") + g("dl")) ** 2 - g("lperp") ** 2
T = np.exp(-0.5 * (dr ** 2 - (b / (a + b)) * proj ** 2) / a) * rmask
u_draws = np.asarray(s["rusher_weight"]) @ Rc.T + np.asarray(s["tau"])[:, None] * np.asarray(s["z_rusher"])
u_mean = u_draws.mean(0)

base = g("b0") + g("gt1") * data["t_since_std"] + g("gt2") * data["t_since_std"] ** 2
for nm, idk in [("qb", "quarterback_ids"), ("down", "down_ids"), ("qtr", "quarter_ids"),
                ("cov", "coverage_ids"), ("form", "form_ids")]:
    if "s_" + nm in s:
        base = base + (g("s_" + nm) * np.asarray(s["z_" + nm]).mean(0))[data[idk]]
for gg, c in [("g_score", "score_diff_cov"), ("g_time", "time_cov"), ("g_yard", "yard_cov")]:
    base = base + g(gg) * data[c]
y = data["release"]
ll = lambda p: (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).sum()
ll0 = (y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean())).sum()
p_full = sig(base + np.einsum("nr,nr->n", u_mean[sid], T))
print(f"\n  McFadden pseudo-R^2: full {1 - ll(p_full)/ll0:.3f}   (no rusher {1 - ll(sig(base))/ll0:.3f})", flush=True)
print(f"  zone radius sigma_q: mean {sigq_qb.mean():.2f}yd  range [{sigq_qb.min():.2f},{sigq_qb.max():.2f}]  "
      f"lperp={g('lperp'):.2f} lpar={g('lperp')+g('dl'):.2f}  tau={g('tau'):.3f}", flush=True)

players = pd.read_csv("data/players.csv").set_index("nflId")
invq = {v: k for k, v in enc["qb"].items()}
qf = pd.DataFrame({"nflId": [invq[i] for i in range(len(sigq_qb))], "sigma_q": sigq_qb})
qf["name"] = qf["nflId"].map(players["displayName"])
qf = qf[qf["nflId"].isin([invq[i] for i in np.unique(qb)])].sort_values("sigma_q")
print("  smallest danger zones (calm pocket):", ", ".join(f"{r['name']} {r['sigma_q']:.2f}" for _, r in qf.head(4).iterrows()))
print("  largest danger zones (pressure-sensitive):", ", ".join(f"{r['name']} {r['sigma_q']:.2f}" for _, r in qf.tail(4).iterrows()))

inv = {v: k for k, v in enc["rusher"].items()}
nG = data["N_rushers"]; cnt = np.zeros(nG); np.add.at(cnt, sid[rmask > 0], 1.0)
rate = pd.DataFrame({"nflId": [inv[i] for i in range(nG)], "frames": cnt, "u": u_mean,
                     "lo": np.percentile(u_draws, 2.5, 0), "hi": np.percentile(u_draws, 97.5, 0)})
rate["name"] = rate["nflId"].map(players["displayName"]); rate["pos"] = rate["nflId"].map(players["officialPosition"])
rate = rate[rate["frames"] >= 200].sort_values("u", ascending=False).reset_index(drop=True)
rate.to_csv(f"qb_zone_rating_{W}wk.csv", index=False)
f = lambda d: "\n".join(f"   {r['name']:22s} {r['pos']:3s} u={r['u']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]  ({int(r['frames'])} fr)" for _, r in d.iterrows())
print(f"\n=== TOP 15 rushers by zone-overlap charge u_j (>=200 fr, n={len(rate)}) ===\n{f(rate.head(15))}")
print(f"\n=== BOTTOM 6 ===\n{f(rate.tail(6))}")
for grp, ps2 in [("Edge", ["DE", "OLB"]), ("Interior", ["DT", "NT"])]:
    print(f"  {grp}: " + ", ".join(f"{r['name']} {r['u']:+.3f}" for _, r in rate[rate.pos.isin(ps2)].head(5).iterrows()))
print(f"\nwrote qb_zone_samples_{W}wk.pkl, qb_zone_rating_{W}wk.csv")
