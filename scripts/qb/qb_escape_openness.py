"""S2: condition the QB release-hazard model on downfield receiver OPENNESS and test whether it fixes
the documented escape-rating inversion (rusher charge u_j correlated NEGATIVELY with PFF pressure
because the model couldn't tell a forced release from a throw-to-an-open-man release). Fits the SAME
QBEscapeModel on IDENTICAL frames with use_openness False vs True and compares corr(u, PFF pressure)."""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import feature_engineering as fe
from models import QBEscapeModel

W = 8
pkl = f"qb_force_design_{W}wk_open.pkl"
try:
    data, enc = pickle.load(open(pkl, "rb")); assert "open_cov" in data and "dropback_ids" in data
    print(f"loaded {pkl}", flush=True)
except Exception:
    print(f"building {W}-week design with openness ...", flush=True)
    data, enc = fe.build_qb_force_design_from_files(weeks=range(W), openness_path="openness_spacecontrol.csv")
    pickle.dump((data, enc), open(pkl, "wb"))
assert "open_cov" in data, "open_cov missing"

N = len(data["a_qb"]); ps = data["play_start"]; pr = data["play_row"]
t_since = np.arange(N) - ps[pr]
release = np.zeros(N, np.float32); release[ps[1:] - 1] = 1.0
data["release"] = release
data["t_since_std"] = ((t_since - t_since.mean()) / (t_since.std() + 1e-9)).astype(np.float32)
print(f"N={N:,}  releases={int(release.sum()):,}  open_cov mean={data['open_cov'].mean():+.2f} std={data['open_cov'].std():.2f}", flush=True)

Rc, sid, rmask = data["rusher_covariates"], data["rusher_slot_id"], data["rmask"]
players = pd.read_csv("data/players.csv").set_index("nflId"); inv = {v: k for k, v in enc["rusher"].items()}
nG = data["N_rushers"]; cnt = np.zeros(nG); np.add.at(cnt, sid[rmask > 0], 1.0)
pff = pd.read_csv("data/pffScoutingData.csv"); prr = pff[pff.pff_role == "Pass Rush"].copy()
prr["pressure"] = prr[["pff_sack", "pff_hit", "pff_hurry"]].max(axis=1)
rpff = prr.groupby("nflId").agg(pff_snaps=("pressure", "size"), pff_press=("pressure", "mean"),
                                pff_sack=("pff_sack", "mean"), pff_hurry=("pff_hurry", "mean")).reset_index()


def fit_rate(use_open):
    s = QBEscapeModel(use_situation=True, use_openness=use_open).run_svi_inference(data, num_steps=12000, lr=5e-3)
    ud = np.asarray(s["rusher_weight"]) @ Rc.T + np.asarray(s["tau"])[:, None] * np.asarray(s["z_rusher"])
    r = pd.DataFrame({"nflId": [inv[i] for i in range(nG)], "frames": cnt, "u": ud.mean(0)})
    r["pos"] = r["nflId"].map(players["officialPosition"]); r["name"] = r["nflId"].map(players["displayName"])
    return s, r


def validate(r, label):
    m = r[r.frames >= 200].merge(rpff[rpff.pff_snaps >= 50], on="nflId", how="inner")
    ei = m[m.pos.isin(["DE", "OLB", "DT", "NT"])]
    out = []
    for name, d in [("ALL", m), ("EdgeDL", ei)]:
        for tgt in ["pff_press", "pff_sack", "pff_hurry"]:
            dd = d[["u", tgt]].dropna(); rr, p = pearsonr(dd.u, dd[tgt]); rho, _ = spearmanr(dd.u, dd[tgt])
            out.append((name, tgt, rr, rho, len(dd)))
    print(f"\n=== {label}: rusher charge u vs PFF (n_all={len(m)}, edge/DL={len(ei)}) ===", flush=True)
    for name, tgt, rr, rho, n in out:
        print(f"  [{name:6}] u vs {tgt:<10} Pearson {rr:+.3f}  Spearman {rho:+.3f}")
    return m


print("\n--- fitting BASELINE (no openness) ---", flush=True)
sA, rA = fit_rate(False); validate(rA, "BASELINE")
print("\n--- fitting +OPENNESS ---", flush=True)
sB, rB = fit_rate(True)
go = np.asarray(sB["g_open"]); print(f"\n  g_open mean={go.mean():+.3f}  P(g_open>0)={np.mean(go > 0):.3f}  "
                                     f"(>0 => QB releases when a receiver is open)", flush=True)
validate(rB, "+OPENNESS")
rB.merge(rpff, on="nflId", how="left").sort_values("u", ascending=False).to_csv("qb_escape_openness_rating.csv", index=False)
print("\nwrote qb_escape_openness_rating.csv", flush=True)
