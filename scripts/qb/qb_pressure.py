"""Direction-agnostic pressure model: |a_QB| ~ sum of per-rusher pressures (charge x proximity x
closing). Credits ANY rusher who forces the QB to move (edge or interior). Per-player attribute-
centered charge = rating.  Usage: python3 qb_pressure.py [n_weeks]  (default 8; 1 for smoke)."""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import feature_engineering as fe
from models import QBPressureModel

W = int(sys.argv[1]) if len(sys.argv) > 1 else 8
print(f"building {W}-week design ...", flush=True)
data, enc = fe.build_qb_force_design_from_files(weeks=range(W))
a = data["a_qb"]; Rc = data["rusher_covariates"]; dist = data["dist"]; sclose = data["sclose"]
rmask = data["rmask"]; sid = data["rusher_slot_id"]
m = np.linalg.norm(a, axis=-1)
print(f"  N={len(a):,}  N_rushers={data['N_rushers']}  mean|a_qb|={m.mean():.2f}", flush=True)
pickle.dump((data, enc), open(f"qb_force_design_{W}wk.pkl", "wb"))
softplus = lambda x: np.logaddexp(0.0, x)
r2 = lambda yh: 1 - ((m - yh) ** 2).sum() / ((m - m.mean()) ** 2).sum()


def baseline_np(s):
    b = float(np.asarray(s["b0"]).mean())
    for nm, idk in [("qb", "quarterback_ids"), ("down", "down_ids"), ("qtr", "quarter_ids"),
                    ("cov", "coverage_ids"), ("form", "form_ids")]:
        if "s_" + nm in s:
            eff = float(np.asarray(s["s_" + nm]).mean()) * np.asarray(s["z_" + nm]).mean(0)
            b = b + eff[data[idk]]
    for g, c in [("g_score", "score_diff_cov"), ("g_time", "time_cov"), ("g_yard", "yard_cov")]:
        if g in s:
            b = b + float(np.asarray(s[g]).mean()) * data[c]
    return b


def mean_pred(s, u=None):
    ell = float(np.asarray(s["ell"]).mean()); c0 = float(np.asarray(s["c0"]).mean())
    q = softplus(c0 + (u[sid] if u is not None else 0.0))
    pressure = q * np.exp(-dist / ell) * np.maximum(sclose, 0) * rmask
    return baseline_np(s) + pressure.sum(1)


s0 = QBPressureModel(use_player=False).run_svi_inference(data, num_steps=8000, lr=5e-3)
print(f"\n  [pressure, no player] R^2(|a_qb|)={r2(mean_pred(s0)):.3f}  "
      f"ell={float(np.asarray(s0['ell']).mean()):.2f}yd  b0={float(np.asarray(s0['b0']).mean()):.2f}", flush=True)

s = QBPressureModel(use_player=True).run_svi_inference(data, num_steps=12000, lr=5e-3)
pickle.dump(s, open(f"qb_pressure_samples_{W}wk.pkl", "wb"))
u_draws = np.asarray(s["rusher_weight"]) @ Rc.T + np.asarray(s["tau"])[:, None] * np.asarray(s["z_rusher"])
u_mean = u_draws.mean(0)
print(f"  [pressure, per-player] R^2={r2(mean_pred(s, u_mean)):.3f}  tau={float(np.asarray(s['tau']).mean()):.3f}", flush=True)

players = pd.read_csv("data/players.csv").set_index("nflId")
inv = {v: k for k, v in enc["rusher"].items()}
nG = data["N_rushers"]; cnt = np.zeros(nG); np.add.at(cnt, sid[rmask > 0], 1.0)
rate = pd.DataFrame({"nflId": [inv[i] for i in range(nG)], "frames": cnt, "u": u_mean,
                     "lo": np.percentile(u_draws, 2.5, 0), "hi": np.percentile(u_draws, 97.5, 0)})
rate["name"] = rate["nflId"].map(players["displayName"]); rate["pos"] = rate["nflId"].map(players["officialPosition"])
rate = rate[rate["frames"] >= 200].sort_values("u", ascending=False).reset_index(drop=True)
rate.to_csv(f"qb_pressure_rating_{W}wk.csv", index=False)
fmt = lambda d: "\n".join(f"   {r['name']:22s} {r['pos']:3s} u={r['u']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}]  ({int(r['frames'])} fr)"
                          for _, r in d.iterrows())
print(f"\n=== TOP 15 by forced-movement charge u_j (>=200 fr, n={len(rate)}) ===\n{fmt(rate.head(15))}")
print(f"\n=== BOTTOM 6 ===\n{fmt(rate.tail(6))}")
for grp, ps in [("Edge", ["DE", "OLB"]), ("Interior", ["DT", "NT"])]:
    print(f"  {grp}: " + ", ".join(f"{r['name']} {r['u']:+.2f}" for _, r in rate[rate.pos.isin(ps)].head(5).iterrows()))
print(f"\nwrote qb_pressure_samples_{W}wk.pkl, qb_pressure_rating_{W}wk.csv")
