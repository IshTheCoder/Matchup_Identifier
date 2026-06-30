"""Matchup block-failure survival model (BlockFailureHazardModel). Fits an exponential frailty hazard
to the null-state 'beaten' spells (block_failure_events.csv) with crossed blocker (hold) + rusher
(shed) random effects, so each side is OPPONENT-ADJUSTED. Outputs opponent-adjusted hold/shed
ratings, matchup expected hold-times, and validates vs PFF. Usage: python3 run_block_survival_model.py"""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, "src"); sys.path.insert(0, "model")
from models import BlockFailureHazardModel

DT = 0.1; MIN_SPELLS = 40
ev = pd.read_csv("block_failure_events.csv")
players = pd.read_csv("data/players.csv").set_index("nflId")
benc = {b: i for i, b in enumerate(np.sort(ev.nflId.unique()))}
renc = {r: i for i, r in enumerate(np.sort(ev.primary_rusher.unique()))}
ev["bid"] = ev.nflId.map(benc); ev["rid"] = ev.primary_rusher.map(renc)
ev["expo"] = (ev.time + 1) * DT                                  # seconds of exposure per spell
base = ev.event.sum() / ev.expo.sum()
data = {"time": ev.expo.to_numpy(float), "event": ev.event.to_numpy(float),
        "blocker_id": ev.bid.to_numpy(), "rusher_id": ev.rid.to_numpy(),
        "N_blockers": len(benc), "N_rushers": len(renc), "base_hazard": float(base)}
print(f"spells={len(ev):,}  beaten={int(ev.event.sum()):,} ({ev.event.mean():.1%})  "
      f"blockers={len(benc)} rushers={len(renc)}  base hazard={base:.3f}/s (mean hold {1/base:.1f}s)", flush=True)

m = BlockFailureHazardModel()
s = m.run_svi_inference(data, num_steps=15000, lr=5e-3)
pickle.dump(s, open("block_survival_samples.pkl", "wb"))
al = s["alpha"]; ud = s["sigma_b"][:, None] * s["z_blocker"]; vd = s["sigma_r"][:, None] * s["z_rusher"]
print(f"  alpha={al.mean():+.3f}  sigma_b={s['sigma_b'].mean():.3f}  sigma_r={s['sigma_r'].mean():.3f}", flush=True)

invb = {v: k for k, v in benc.items()}; invr = {v: k for k, v in renc.items()}
nspell_b = ev.groupby("bid").size(); nspell_r = ev.groupby("rid").size()
# blocker HOLD rating = -u (high = holds longer); baseline hold time vs an average rusher = 1/exp(alpha+u)
hold = pd.DataFrame({"nflId": [invb[i] for i in range(len(benc))],
                     "hold": -ud.mean(0), "lo": -np.percentile(ud, 97.5, 0), "hi": -np.percentile(ud, 2.5, 0),
                     "hold_time_s": (1.0 / np.exp(al[:, None] + ud)).mean(0), "spells": [int(nspell_b.get(i, 0)) for i in range(len(benc))]})
hold["name"] = hold.nflId.map(players.displayName); hold["pos"] = hold.nflId.map(players.officialPosition)
# rusher SHED rating = v (high = beats blocks faster)
shed = pd.DataFrame({"nflId": [invr[i] for i in range(len(renc))],
                     "shed": vd.mean(0), "lo": np.percentile(vd, 2.5, 0), "hi": np.percentile(vd, 97.5, 0),
                     "spells": [int(nspell_r.get(i, 0)) for i in range(len(renc))]})
shed["name"] = shed.nflId.map(players.displayName); shed["pos"] = shed.nflId.map(players.officialPosition)
hold.to_csv("block_hold_ratings.csv", index=False); shed.to_csv("rusher_shed_ratings.csv", index=False)

# ---- validation vs PFF ----
pff = pd.read_csv("data/pffScoutingData.csv")
pb = pff[pff.pff_role == "Pass Block"].copy(); pb["pa"] = pb[["pff_sackAllowed", "pff_hitAllowed", "pff_hurryAllowed"]].max(axis=1)
bpff = pb.groupby("nflId").agg(pb_snaps=("pa", "size"), press_allowed=("pa", "mean")).reset_index()
pr = pff[pff.pff_role == "Pass Rush"].copy(); pr["p"] = pr[["pff_sack", "pff_hit", "pff_hurry"]].max(axis=1)
rpff = pr.groupby("nflId").agg(pr_snaps=("p", "size"), press=("p", "mean")).reset_index()
sb = pd.read_csv("survival_by_blocker.csv")[["nflId", "beat_rate"]]
H = hold[hold.spells >= MIN_SPELLS].merge(bpff, on="nflId", how="left").merge(sb, on="nflId", how="left")
S = shed[shed.spells >= MIN_SPELLS].merge(rpff, on="nflId", how="left")
Hp = H.dropna(subset=["press_allowed"]); Sp = S.dropna(subset=["press"])
print(f"\n=== BLOCKER hold rating (opponent-adjusted) vs PFF (n={len(Hp)}; expect NEGATIVE) ===", flush=True)
print(f"  hold vs press_allowed: Pearson={pearsonr(Hp.hold, Hp.press_allowed)[0]:+.3f} Spearman={spearmanr(Hp.hold, Hp.press_allowed)[0]:+.3f}")
print(f"  hold vs raw beat_rate: Pearson={pearsonr(H.dropna(subset=['beat_rate']).hold, H.dropna(subset=['beat_rate']).beat_rate)[0]:+.3f}")
print(f"=== RUSHER shed rating (opponent-adjusted) vs PFF pressure (n={len(Sp)}; expect POSITIVE) ===", flush=True)
print(f"  shed vs press: Pearson={pearsonr(Sp.shed, Sp.press)[0]:+.3f} Spearman={spearmanr(Sp.shed, Sp.press)[0]:+.3f}")

print("\nbest pass protectors (longest opponent-adjusted hold, >=40 spells):", flush=True)
for _, x in H.sort_values("hold", ascending=False).head(10).iterrows():
    print(f"  {x['name']:22s} {x.pos:2s} hold={x.hold:+.3f} (~{x.hold_time_s:.1f}s vs avg) pa={x.press_allowed:.2f} ({x.spells} spells)")
print("best rushers (fastest opponent-adjusted shed):", flush=True)
for _, x in S.sort_values("shed", ascending=False).head(10).iterrows():
    print(f"  {x['name']:22s} {x.pos:3s} shed={x.shed:+.3f} press={x.press:.2f} ({x.spells} spells)")

# ---- matchup expected hold-time grid: top tackles x top edge rushers ----
TK = H[H.pos.isin(["T"])].sort_values("hold", ascending=False).head(5)
ED = S[S.pos.isin(["DE", "OLB"])].sort_values("shed", ascending=False).head(5)
print("\nMATCHUP expected hold time (s) -- top tackles (rows) vs top edge rushers (cols):", flush=True)
print("  " + "".join(f"{n.split()[-1][:9]:>10s}" for n in ED.name))
for _, b in TK.iterrows():
    ub = -b.hold                                                # u_b (log-hazard shift)
    cells = [1.0 / np.exp(al.mean() + ub + sv) for sv in ED.shed]   # 1/h_{bj} = expected hold (s)
    print(f"  {b['name'].split()[-1][:14]:14s}" + "".join(f"{c:10.1f}" for c in cells))
