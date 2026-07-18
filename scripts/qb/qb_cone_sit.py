"""Apples-to-apples: the radial cone force-field model WITH the QB + game-situation 2-D drift,
vs without it. Tests whether the situational controls (which rescued the pressure model on week 0)
change the directional model's edge sign-bias, or just raise R^2."""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "model")
from models import QBConeFieldModel

data, enc = pickle.load(open("qb_force_design_8wk.pkl", "rb"))
a = data["a_qb"]; uhat, dist, rvel = data["uhat"], data["dist"], data["rvel"]
aclose, rmask, sid, Rc = data["aclose"], data["rmask"], data["rusher_slot_id"], data["rusher_covariates"]
vmag = np.linalg.norm(rvel, axis=-1); vhat = rvel / (vmag[..., None] + 0.5)
cos = (uhat * vhat).sum(-1); sw = vmag / (vmag + 1.0)
softplus = lambda x: np.logaddexp(0.0, x)
r2 = lambda m: 1 - ((a - m) ** 2).sum() / ((a - a.mean(0)) ** 2).sum()
r2c = lambda j, m: 1 - ((a[:, j] - m[:, j]) ** 2).sum() / ((a[:, j] - a[:, j].mean()) ** 2).sum()
print(f"N={len(a):,}  N_rushers={data['N_rushers']}  N_qb={data['N_qb']}", flush=True)


def baseline2d(s):
    b = np.broadcast_to(np.asarray(s["drift"]).mean(0)[None, :], (len(a), 2)).copy()
    for nm, idk in [("qb", "quarterback_ids"), ("down", "down_ids"), ("qtr", "quarter_ids"),
                    ("cov", "coverage_ids"), ("form", "form_ids")]:
        if "s_" + nm in s:
            b = b + (float(np.asarray(s["s_" + nm]).mean()) * np.asarray(s["z_" + nm]).mean(0))[data[idk]]
    for g, c in [("g_score", "score_diff_cov"), ("g_time", "time_cov"), ("g_yard", "yard_cov")]:
        if g in s:
            b = b + np.asarray(s[g]).mean(0)[None, :] * data[c][:, None]
    return b


def cone_mean(s, u=None):
    P = {k: float(np.asarray(s[k]).mean()) for k in ["ell0", "ell_v", "kappa", "c0", "c_a"]}
    A = 1 - sw * (1 - 1 / (1 + np.exp(-P["kappa"] * cos)))
    rho = np.exp(-0.5 * (dist / (P["ell0"] + P["ell_v"] * vmag)) ** 2)
    arg = P["c0"] + P["c_a"] * np.maximum(aclose, 0) + (u[sid] if u is not None else 0.0)
    fmag = softplus(arg) * rho * A * rmask
    return np.einsum("nr,nrd->nd", fmag, uhat) + baseline2d(s)   # radial direction


for use_sit in [False, True]:
    s = QBConeFieldModel(profile="gauss", direction="radial", use_player=True,
                         use_situation=use_sit).run_svi_inference(data, num_steps=12000, lr=5e-3)
    u_draws = np.asarray(s["rusher_weight"]) @ Rc.T + np.asarray(s["tau"])[:, None] * np.asarray(s["z_rusher"])
    u_mean = u_draws.mean(0)
    m = cone_mean(s, u_mean)
    tag = "radial + QB/situation" if use_sit else "radial, no situation"
    print(f"\n=== {tag} ===  R^2={r2(m):.3f}  (x {r2c(0,m):.3f}, y {r2c(1,m):.3f})  tau={float(np.asarray(s['tau']).mean()):.2f}", flush=True)
    if use_sit:
        players = pd.read_csv("data/players.csv").set_index("nflId")
        inv = {v: k for k, v in enc["rusher"].items()}
        nG = data["N_rushers"]; cnt = np.zeros(nG); np.add.at(cnt, sid[rmask > 0], 1.0)
        rate = pd.DataFrame({"nflId": [inv[i] for i in range(nG)], "frames": cnt, "u": u_mean})
        rate["name"] = rate["nflId"].map(players["displayName"]); rate["pos"] = rate["nflId"].map(players["officialPosition"])
        rate = rate[rate["frames"] >= 200].sort_values("u", ascending=False).reset_index(drop=True)
        rate.to_csv("qb_cone_sit_rating_8wk.csv", index=False)
        f = lambda d: "\n".join(f"   {r['name']:22s} {r['pos']:3s} u={r['u']:+.2f} ({int(r['frames'])} fr)" for _, r in d.iterrows())
        print(f"TOP 12:\n{f(rate.head(12))}\nBOTTOM 5:\n{f(rate.tail(5))}")
        for grp, ps in [("Edge", ["DE", "OLB"]), ("Interior", ["DT", "NT"])]:
            print(f"  {grp}: " + ", ".join(f"{r['name']} {r['u']:+.2f}" for _, r in rate[rate.pos.isin(ps)].head(5).iterrows()))
        pickle.dump(s, open("qb_cone_sit_samples_8wk.pkl", "wb"))
print("\nwrote qb_cone_sit_rating_8wk.csv, qb_cone_sit_samples_8wk.pkl")
