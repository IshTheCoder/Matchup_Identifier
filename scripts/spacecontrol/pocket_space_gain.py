"""Driver for the pocket Space Occupation Gain / Loss (Bornn analog), computed entirely from the
space-control density (no HMM assignment):
 - RUSHER Space Occupation GAIN (SOG): wins valuable near-QB ground -> validate vs PFF pressure.
 - LINEMAN Space Occupation LOSS (SOL): loses control of the protected pocket when beaten -> validate
   vs PFF pressures allowed (expect POSITIVE; also report WITHIN-position). Usage: [pkl]"""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, "src")
import space_gain as sg

pkl = sys.argv[1] if len(sys.argv) > 1 else "pocket_design_8wk.pkl"
data, enc = pickle.load(open(pkl, "rb"))
invr = {v: k for k, v in enc["rusher"].items()}; invb = {v: k for k, v in enc["blocker"].items()}
print(f"loaded {pkl}: N={len(data['x_qb']):,} frames, {len(data['play_start'])-1} plays", flush=True)
players = pd.read_csv("data/players.csv")[["nflId", "displayName", "officialPosition"]]
pff = pd.read_csv("data/pffScoutingData.csv")

rdf, bdf = sg.compute_pocket_gains(data, rad=5.0, lam=3.0, eps=0.0)

# ---------- RUSHER: Space Occupation Gain ----------
agg = sg.aggregate_by_rusher(rdf); agg["nflId"] = agg.rid.map(invr)
prr = pff[pff.pff_role == "Pass Rush"].copy(); prr["pr"] = prr[["pff_sack", "pff_hit", "pff_hurry"]].max(axis=1)
rpff = prr.groupby("nflId").agg(pff_snaps=("pr", "size"), pff_press=("pr", "mean")).reset_index()
agg = agg.merge(players, on="nflId", how="left").merge(rpff, on="nflId", how="left")
agg.to_csv("pocket_space_gain_rusher.csv", index=False)
q = agg[(agg.frames >= 200) & (agg.pff_snaps >= 50)].dropna(subset=["pff_press"])
print(f"\n=== RUSHER Space Occupation GAIN vs PFF pressure (n={len(q)}) ===", flush=True)
for m in ["sumSOG", "muSOG", "rate"]:
    print(f"  {m:7s} Pearson={pearsonr(q[m], q.pff_press)[0]:+.3f}  Spearman={spearmanr(q[m], q.pff_press)[0]:+.3f}")

# ---------- LINEMAN: Space Occupation Loss ----------
bag = sg.aggregate_by_blocker(bdf); bag["nflId"] = bag.bid.map(invb)
pbb = pff[pff.pff_role == "Pass Block"].copy(); pbb["pa"] = pbb[["pff_sackAllowed", "pff_hitAllowed", "pff_hurryAllowed"]].max(axis=1)
bpff = pbb.groupby("nflId").agg(pb_snaps=("pa", "size"), press_allowed=("pa", "mean")).reset_index()
bag = bag.merge(players, on="nflId", how="left").merge(bpff, on="nflId", how="left")
bag.to_csv("pocket_space_lost_blocker.csv", index=False)
b = bag[(bag.frames >= 200) & (bag.pb_snaps >= 50)].dropna(subset=["press_allowed"])
print(f"\n=== LINEMAN Space Occupation LOSS vs PFF pressures allowed (n={len(b)}; expect POSITIVE) ===", flush=True)
for m in ["sumSOL", "muSOL", "rate"]:
    print(f"  {m:7s} Pearson={pearsonr(b[m], b.press_allowed)[0]:+.3f}  Spearman={spearmanr(b[m], b.press_allowed)[0]:+.3f}")
b2 = b.copy(); b2["r_dm"] = b2["rate"] - b2.groupby("officialPosition")["rate"].transform("mean")
b2["pa_dm"] = b2.press_allowed - b2.groupby("officialPosition").press_allowed.transform("mean")
print(f"  WITHIN-position (rate vs pressures-allowed, demeaned): Pearson={pearsonr(b2.r_dm, b2.pa_dm)[0]:+.3f}")
for pos in ["T", "G", "C"]:
    s = b[b.officialPosition == pos]
    if len(s) > 10:
        print(f"    {pos}-only (n={len(s)}): Spearman={spearmanr(s['rate'], s.press_allowed)[0]:+.3f}")

print(f"\ntop 12 RUSHERS by Space Occupation Gain (>=200 fr):", flush=True)
for _, x in q.sort_values("sumSOG", ascending=False).head(12).iterrows():
    print(f"  {x.displayName:22s} {x.officialPosition:3s} sumSOG={x.sumSOG:6.1f} muSOG={x.muSOG:.3f} active={x.active:.2f} press={x.pff_press:.2f}")
ol = b[b.officialPosition.isin(["T", "G", "C"]) & (b.frames >= 400)]
print(f"\nLINEMEN, MOST space lost / worst (OL, >=400 fr):", flush=True)
for _, x in ol.sort_values("rate", ascending=False).head(8).iterrows():
    print(f"  {x.displayName:22s} {x.officialPosition:2s} rate={x['rate']:.4f} muSOL={x.muSOL:.3f} pa={x.press_allowed:.2f}")
print(f"LINEMEN, LEAST space lost / best:", flush=True)
for _, x in ol.sort_values("rate").head(8).iterrows():
    print(f"  {x.displayName:22s} {x.officialPosition:2s} rate={x['rate']:.4f} muSOL={x.muSOL:.3f} pa={x.press_allowed:.2f}")
