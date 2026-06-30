"""Kramers escape model: per-frame QB-release hazard rises with rusher pressure. Discrete-time
logistic survival on QB-frames. Per-rusher attribute-centered charge u_j = how much he accelerates
the forced release (edge-inclusive). Usage: python3 qb_escape.py [n_weeks] (default 8)."""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import feature_engineering as fe
from models import QBEscapeModel

W = int(sys.argv[1]) if len(sys.argv) > 1 else 8
pkl = f"qb_force_design_{W}wk.pkl"
try:
    data, enc = pickle.load(open(pkl, "rb")); assert "dropback_ids" in data
    print(f"loaded {pkl}", flush=True)
except Exception:
    print(f"building {W}-week design ...", flush=True)
    data, enc = fe.build_qb_force_design_from_files(weeks=range(W)); pickle.dump((data, enc), open(pkl, "wb"))

# discrete-time survival targets: event = the play's LAST frame (ball out / sack); t = frames since snap
N = len(data["a_qb"]); ps = data["play_start"]; pr = data["play_row"]
t_since = np.arange(N) - ps[pr]
release = np.zeros(N, np.float32); release[ps[1:] - 1] = 1.0
data["release"] = release
data["t_since_std"] = ((t_since - t_since.mean()) / (t_since.std() + 1e-9)).astype(np.float32)
print(f"  N={N:,}  releases={int(release.sum()):,}  base rate={release.mean():.4f}  "
      f"mean hold={t_since[ps[1:]-1].mean():.1f} frames", flush=True)

s = QBEscapeModel(use_situation=True).run_svi_inference(data, num_steps=12000, lr=5e-3)
pickle.dump(s, open(f"qb_escape_samples_{W}wk.pkl", "wb"))
Rc, sid, rmask, dist, sclose = (data["rusher_covariates"], data["rusher_slot_id"], data["rmask"],
                                data["dist"], data["sclose"])
g = lambda k: float(np.asarray(s[k]).mean())
u_draws = np.asarray(s["rusher_weight"]) @ Rc.T + np.asarray(s["tau"])[:, None] * np.asarray(s["z_rusher"])
u_mean = u_draws.mean(0)

# fitted release prob + pseudo-R^2 / AUC (does rusher pressure predict the release clock?)
ts = data["t_since_std"]; sig = lambda x: 1 / (1 + np.exp(-x))
base = g("b0") + g("gt1") * ts + g("gt2") * ts ** 2
for nm, idk in [("qb", "quarterback_ids"), ("down", "down_ids"), ("qtr", "quarter_ids"),
                ("cov", "coverage_ids"), ("form", "form_ids"), ("dropback", "dropback_ids"), ("cov2", "cov2_ids")]:
    if "s_" + nm in s:
        base = base + (g("s_" + nm) * np.asarray(s["z_" + nm]).mean(0))[data[idk]]
for gg, c in [("g_score", "score_diff_cov"), ("g_time", "time_cov"), ("g_yard", "yard_cov"), ("g_box", "box_cov"), ("g_pa", "pa_cov"), ("g_depth", "depth_cov")]:
    base = base + g(gg) * data[c]
phi = np.exp(-dist / g("ell")) * rmask     # proximity (matches QBEscapeModel default)
if "eta_block" in s:
    phi = phi * np.exp(-g("eta_block") * data["dose"])
    eb = np.asarray(s["eta_block"])
    print(f"  blocking attenuation eta={eb.mean():+.3f}  P(eta>0)={np.mean(eb > 0):.3f}  "
          f"(eta>0 => a blocked rusher exerts LESS escape pressure)", flush=True)
p_full = sig(base + np.einsum("nr,nr->n", u_mean[sid], phi))
p_base = sig(base)
y = release
ll = lambda p: (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).sum()
ll0 = (y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean())).sum()
print(f"\n  McFadden pseudo-R^2:  full {1 - ll(p_full)/ll0:.3f}   (QB/situation only {1 - ll(p_base)/ll0:.3f})", flush=True)
order = np.argsort(p_full); auc = (y[order].cumsum() * (1 - y[order])).sum() / (y.sum() * (1 - y).sum())
print(f"  AUC(predict release frame): {auc:.3f}   ell={g('ell'):.2f}yd  tau={g('tau'):.3f}", flush=True)

players = pd.read_csv("data/players.csv").set_index("nflId")
inv = {v: k for k, v in enc["rusher"].items()}
nG = data["N_rushers"]; cnt = np.zeros(nG); np.add.at(cnt, sid[rmask > 0], 1.0)
rate = pd.DataFrame({"nflId": [inv[i] for i in range(nG)], "frames": cnt, "u": u_mean,
                     "lo": np.percentile(u_draws, 2.5, 0), "hi": np.percentile(u_draws, 97.5, 0)})
rate["name"] = rate["nflId"].map(players["displayName"]); rate["pos"] = rate["nflId"].map(players["officialPosition"])
rate = rate[rate["frames"] >= 200].sort_values("u", ascending=False).reset_index(drop=True)
rate.to_csv(f"qb_escape_rating_{W}wk.csv", index=False)
f = lambda d: "\n".join(f"   {r['name']:22s} {r['pos']:3s} u={r['u']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]  ({int(r['frames'])} fr)" for _, r in d.iterrows())
print(f"\n=== TOP 15 rushers by release-hazard charge u_j (>=200 fr, n={len(rate)}) ===\n{f(rate.head(15))}")
print(f"\n=== BOTTOM 6 ===\n{f(rate.tail(6))}")
for grp, ps2 in [("Edge", ["DE", "OLB"]), ("Interior", ["DT", "NT"])]:
    print(f"  {grp}: " + ", ".join(f"{r['name']} {r['u']:+.3f}" for _, r in rate[rate.pos.isin(ps2)].head(5).iterrows()))
print(f"\nwrote qb_escape_samples_{W}wk.pkl, qb_escape_rating_{W}wk.csv")
