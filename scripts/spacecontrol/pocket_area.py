"""M1 gate (revised): pocket = OFFENSE-controlled area in a QB-centered disk (protected pocket),
plus the QB's own control-share area. Play-length-robust summaries (min/mean/end) tested vs PFF
PRESSURE (did the pocket fail) and sack. Gate: pocket area must predict pressure (collapse -> pressure)."""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import pointbiserialr
sys.path.insert(0, "src")
import pitch_control as pc

data, enc = pickle.load(open("pocket_design_1wk.pkl", "rb"))
N = len(data["x_qb"]); ps = data["play_start"]
print(f"N={N:,} frames, {len(ps)-1} plays", flush=True)

RAD = 5.0
off_area = np.zeros(N); qb_area = np.zeros(N)
for n in range(N):
    G, gx, gy, cell = pc.qb_grid(data["x_qb"][n], half=6.0, n=40)
    inreg = ((G[:, 0] - data["x_qb"][n, 0]) ** 2 + (G[:, 1] - data["x_qb"][n, 1]) ** 2) <= RAD ** 2
    Ir, ri, Ib, bi, Iqb = pc.frame_fields(G, data, n)
    PC = pc.pitch_control(Ir, Ib, Iqb, include_qb=False)        # P(offense controls)
    off_area[n] = cell * PC[inreg].sum()                        # protected pocket area (yd^2)
    _, _, wqb = pc.control_shares(Ir, Ib, Iqb, include_qb=True)
    qb_area[n] = cell * wqb[inreg].sum()
np.save("pocket_areas_1wk.npy", np.stack([off_area, qb_area]))

rows = []
for p in range(len(ps) - 1):
    fr = np.arange(ps[p], ps[p + 1])
    if len(fr) < 3:
        continue
    o, q = off_area[fr], qb_area[fr]
    rows.append((data["gameId"][fr[0]], data["playId"][fr[0]],
                 o.mean(), o.min(), o[-1], q.mean(), q.min(), q[-1], len(fr)))
df = pd.DataFrame(rows, columns=["gameId", "playId", "off_mean", "off_min", "off_end",
                                 "qb_mean", "qb_min", "qb_end", "frames"])
plays = pd.read_csv("data/plays.csv")
df = df.merge(plays[["gameId", "playId", "passResult"]], on=["gameId", "playId"], how="left")
df["sack"] = (df["passResult"] == "S").astype(int)
pff = pd.read_csv("data/pffScoutingData.csv"); prr = pff[pff.pff_role == "Pass Rush"]
prr = prr.assign(pr=prr[["pff_sack", "pff_hit", "pff_hurry"]].max(axis=1))
press = prr.groupby(["gameId", "playId"])["pr"].max().rename("pressure").reset_index()
df = df.merge(press, on=["gameId", "playId"], how="left"); df["pressure"] = df["pressure"].fillna(0).astype(int)
df.to_csv("pocket_area_by_play.csv", index=False)

print(f"\n=== M1 GATE (n={len(df)} plays, {int(df.sack.sum())} sacks, {int(df.pressure.sum())} pressures) ===", flush=True)
for tgt in ["pressure", "sack"]:
    print(f"-- vs {tgt} --")
    for m in ["off_mean", "off_min", "off_end", "qb_mean", "qb_min", "qb_end"]:
        d = df[[m, tgt]].dropna(); r, pv = pointbiserialr(d[tgt], d[m])
        print(f"   {m:9s} r={r:+.3f} p={pv:.1e}   {tgt}={d[d[tgt]==1][m].mean():.3f}  no={d[d[tgt]==0][m].mean():.3f}")
print("\n(expect pressured/sacked plays to have LOWER pocket area)")
