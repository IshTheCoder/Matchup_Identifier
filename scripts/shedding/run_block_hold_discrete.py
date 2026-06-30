"""Discrete-time logistic hazard of block failure (BlockHoldDiscreteModel). Expands the null-state
engagement spells to per-frame Bernoulli trials, opponent-adjusts via the engaged rusher's STRAIN
plus-minus (fixed covariate), and fits a tightly-regularized blocker hold frailty. KEY TEST: does the
opponent-adjusted hold rating validate vs PFF pressures-allowed WITHIN position (where marginal
blocker metrics fail)?"""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, "src"); sys.path.insert(0, "model")
from models import BlockHoldDiscreteModel

MIN_SPELLS = 40
ev = pd.read_csv("block_failure_events.csv")
players = pd.read_csv("data/players.csv").set_index("nflId")
rr = pd.read_csv("rusher_rankings_phase25.csv").set_index("nflId")["effect"]
ev["ropp"] = ev.primary_rusher.map(rr).fillna(0.0)
ev["ropp"] = (ev.ropp - ev.ropp.mean()) / (ev.ropp.std() + 1e-9)
benc = {b: i for i, b in enumerate(np.sort(ev.nflId.unique()))}
ev["bid"] = ev.nflId.map(benc)

d = np.maximum(ev.time.to_numpy(int), 1)                          # frames engaged per spell (>=1)
idx = np.repeat(np.arange(len(ev)), d)                            # spell index per frame-row
fr = np.arange(d.sum()) - np.repeat(np.cumsum(d) - d, d) + 1      # frame-in-spell 1..d
beaten = ((ev.event.to_numpy()[idx] == 1) & (fr == d[idx])).astype(np.float32)
t = ev.time.to_numpy()                                           # for tstd scale we standardize fr
tstd = (fr - fr.mean()) / (fr.std() + 1e-9)
data = {"t": tstd.astype(np.float32), "t2": (tstd ** 2).astype(np.float32),
        "r_opp": ev.ropp.to_numpy()[idx].astype(np.float32), "blocker_id": ev.bid.to_numpy()[idx],
        "beaten": beaten, "N_blockers": len(benc)}
print(f"spells={len(ev):,} -> frame-rows={len(beaten):,}  beats={int(beaten.sum()):,} "
      f"({beaten.mean():.3%}/frame)  blockers={len(benc)}", flush=True)

m = BlockHoldDiscreteModel()
s = m.run_svi_inference(data, num_steps=15000, lr=5e-3)
pickle.dump(s, open("block_hold_discrete_samples.pkl", "wb"))
print(f"  alpha={s['alpha'].mean():+.3f}  g1={s['g1'].mean():+.3f} g2={s['g2'].mean():+.3f}  "
      f"beta_opp={s['beta_opp'].mean():+.3f} (P>0={np.mean(s['beta_opp']>0):.2f})  sigma_b={s['sigma_b'].mean():.3f}", flush=True)

ud = s["sigma_b"][:, None] * s["z_blocker"]                      # blocker log-hazard shift
invb = {v: k for k, v in benc.items()}
nsp = ev.groupby("bid").size()
hold = pd.DataFrame({"nflId": [invb[i] for i in range(len(benc))], "hold": -ud.mean(0),
                     "lo": -np.percentile(ud, 97.5, 0), "hi": -np.percentile(ud, 2.5, 0),
                     "spells": [int(nsp.get(i, 0)) for i in range(len(benc))]})
hold["name"] = hold.nflId.map(players.displayName); hold["pos"] = hold.nflId.map(players.officialPosition)
pff = pd.read_csv("data/pffScoutingData.csv"); pb = pff[pff.pff_role == "Pass Block"].copy()
pb["pa"] = pb[["pff_sackAllowed", "pff_hitAllowed", "pff_hurryAllowed"]].max(axis=1)
bpff = pb.groupby("nflId").agg(pb_snaps=("pa", "size"), press_allowed=("pa", "mean")).reset_index()
hold = hold.merge(bpff, on="nflId", how="left"); hold.to_csv("block_hold_discrete_ratings.csv", index=False)

q = hold[(hold.spells >= MIN_SPELLS)].dropna(subset=["press_allowed"])
print(f"\n=== Opponent-adjusted HOLD rating vs PFF pressures-allowed (n={len(q)}; expect NEGATIVE) ===", flush=True)
print(f"  overall: Pearson={pearsonr(q.hold, q.press_allowed)[0]:+.3f}  Spearman={spearmanr(q.hold, q.press_allowed)[0]:+.3f}")
q2 = q.copy(); q2["h_dm"] = q2.hold - q2.groupby("pos").hold.transform("mean")
q2["p_dm"] = q2.press_allowed - q2.groupby("pos").press_allowed.transform("mean")
print(f"  WITHIN-position: Pearson={pearsonr(q2.h_dm, q2.p_dm)[0]:+.3f}  (p_fail marginal was T+0.21/C-0.06)")
for pos in ["T", "G", "C"]:
    sdf = q[q.pos == pos]
    if len(sdf) > 10:
        print(f"    {pos} (n={len(sdf)}): Spearman={spearmanr(sdf.hold, sdf.press_allowed)[0]:+.3f}")
print("\nbest opponent-adjusted pass protectors (>=40 spells):", flush=True)
for _, x in q.sort_values("hold", ascending=False).head(12).iterrows():
    print(f"  {x['name']:22s} {x.pos:2s} hold={x.hold:+.3f} [{x.lo:+.3f},{x.hi:+.3f}] pa={x.press_allowed:.2f} ({x.spells} sp)")
print("worst:", flush=True)
for _, x in q.sort_values("hold").head(6).iterrows():
    print(f"  {x['name']:22s} {x.pos:2s} hold={x.hold:+.3f} pa={x.press_allowed:.2f}")
