"""M2: per-rusher SPACE-WON = the rusher's geometric control share integrated over the QB-danger
region, averaged over his frames. Quick checks: (1) space-won vs PFF per-rusher pressure rate (expect
clearly positive); (2) proximity-circularity: corr(space-won, mean dist-to-QB) must be |r|<0.9 (the
skew/lead/speed terms add beyond raw distance)."""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, "src")
import pitch_control as pc

data, enc = pickle.load(open("pocket_design_1wk.pkl", "rb"))
N = len(data["x_qb"]); RAD = 5.0
inv_rush = {v: k for k, v in enc["rusher"].items()}              # encoded idx -> nflId

space_sum = {}; space_cnt = {}; dist_sum = {}
for n in range(N):
    G, gx, gy, cell = pc.qb_grid(data["x_qb"][n], half=6.0, n=40)
    inreg = ((G[:, 0] - data["x_qb"][n, 0]) ** 2 + (G[:, 1] - data["x_qb"][n, 1]) ** 2) <= RAD ** 2
    Ir, ri, Ib, bi, Iqb = pc.frame_fields(G, data, n)
    if len(ri) == 0:
        continue
    wr, _, _ = pc.control_shares(Ir, Ib, Iqb, include_qb=True)
    sp = cell * wr[:, inreg].sum(1)                              # space-won per present rusher (yd^2)
    for j, slot in enumerate(ri):
        rid = int(data["rush_slot_id"][n, slot])
        space_sum[rid] = space_sum.get(rid, 0.0) + sp[j]
        space_cnt[rid] = space_cnt.get(rid, 0) + 1
        dist_sum[rid] = dist_sum.get(rid, 0.0) + float(data["d_rush"][n, slot])

rows = [(inv_rush[r], space_sum[r] / space_cnt[r], dist_sum[r] / space_cnt[r], space_cnt[r])
        for r in space_sum]
rl = pd.DataFrame(rows, columns=["nflId", "space_won", "mean_dist", "frames"])

pff = pd.read_csv("data/pffScoutingData.csv"); prr = pff[pff.pff_role == "Pass Rush"].copy()
prr["pr"] = prr[["pff_sack", "pff_hit", "pff_hurry"]].max(axis=1)
prate = prr.groupby("nflId").agg(press_rate=("pr", "mean"), snaps=("pr", "size")).reset_index()
players = pd.read_csv("data/players.csv")[["nflId", "displayName"]]
rl = rl.merge(prate, on="nflId", how="left").merge(players, on="nflId", how="left")
rl.to_csv("pocket_rusher_space_1wk.csv", index=False)

q = rl[rl.frames >= 50].dropna(subset=["press_rate"])
print(f"rushers (>=50 frames, PFF matched): {len(q)}", flush=True)
rp, pp = pearsonr(q.space_won, q.press_rate); rs, ps = spearmanr(q.space_won, q.press_rate)
print(f"\n=== M2: space-won vs PFF pressure rate ===\n  Pearson r={rp:+.3f} (p={pp:.1e})   Spearman={rs:+.3f} (p={ps:.1e})")
rc, pc_ = pearsonr(q.space_won, q.mean_dist)
print(f"\n=== M2: proximity-circularity  corr(space-won, mean dist-to-QB) ===\n"
      f"  Pearson r={rc:+.3f} (p={pc_:.1e})   {'OK (<0.9 -> beyond proximity)' if abs(rc)<0.9 else 'TOO HIGH -> ~proximity'}")
print("\ntop 12 space-won (>=50 frames):")
for _, x in q.sort_values("space_won", ascending=False).head(12).iterrows():
    print(f"  {x.displayName:22s} space={x.space_won:5.2f}  dist={x.mean_dist:4.1f}  press={x.press_rate:.2f}  fr={int(x.frames)}")
