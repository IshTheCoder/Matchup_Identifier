"""S1 (clean openness): a receiver's openness = how uncontested his CATCH POINT (lead point
x+lead*v) is versus COVERAGE ONLY (teammates don't cover you): c_i = 1/(1 + sum_d I_d(catch_i)),
evaluated pointwise (no grid; exploits the skewed influence -- a trailing defender's suppressed
forward reach => receiver more open). Realizable = c_i * throwability exp(-|catch_i - QB|/D0).
Validates: per-receiver construct vs separation, frame completion, target identity."""
import sys, pickle
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pointbiserialr, pearsonr, spearmanr
sys.path.insert(0, "src")
import pitch_control as pc

COV_R, D0, LEAD = 3.0, 15.0, 0.5
data, enc = pickle.load(open("alltwentytwo_design_1wk.pkl", "rb"))
N = len(data["x_qb"]); inv_rec = {v: k for k, v in enc["receiver"].items()}
print(f"N={N:,}  COV_R={COV_R} D0={D0} LEAD={LEAD}", flush=True)

o_cmax = np.zeros(N); o_real = np.zeros(N); o_sum = np.zeros(N); o_arg = np.zeros(N, int); msep = np.full(N, np.nan)
pair_c, pair_sep = [], []                                       # per-(frame,receiver) for construct validity
for n in range(N):
    ri = np.where(data["rec_mask"][n] > 0)[0]; ci = np.where(data["cov_mask"][n] > 0)[0]
    if ri.size == 0 or ci.size == 0:
        continue
    centers = data["x_rec"][n][ri] + LEAD * data["v_rec"][n][ri]         # catch points (K,2)
    Id = np.stack([pc.influence(centers, data["x_cov"][n][d], data["v_cov"][n][d],
                                data["a_cov"][n][d], 0.0, radius=COV_R) for d in ci])  # (D,K)
    c = 1.0 / (1.0 + Id.sum(0))                                          # openness vs coverage (K,)
    dq = np.hypot(centers[:, 0] - data["x_qb"][n, 0], centers[:, 1] - data["x_qb"][n, 1])
    real = c * np.exp(-dq / D0)
    j = int(np.argmax(real))
    o_cmax[n] = c.max(); o_real[n] = real[j]; o_sum[n] = real.sum(); o_arg[n] = int(data["rec_slot_id"][n, ri[j]])
    sep = cdist(data["x_rec"][n][ri], data["x_cov"][n][ci]).min(1)       # per-receiver separation
    msep[n] = sep.max(); pair_c.extend(c.tolist()); pair_sep.extend(sep.tolist())

df = pd.DataFrame({"gameId": data["gameId"], "playId": data["playId"], "frameId": data["frame_id"],
                   "o_cmax": o_cmax, "o_real": o_real, "o_sum": o_sum, "o_arg": o_arg,
                   "max_sep": msep, "play_row": data["play_row"]})
df.to_csv("openness_week1.csv", index=False)

pc_, ps_ = pearsonr(pair_c, pair_sep), spearmanr(pair_c, pair_sep)
print(f"\n(b) PER-RECEIVER construct: openness c_i vs separation_i (n={len(pair_c):,} pairs) "
      f"Pearson={pc_[0]:+.3f} Spearman={ps_[0]:+.3f}", flush=True)
rel = df.groupby("play_row").tail(1).merge(
    pd.read_csv("data/plays.csv")[["gameId", "playId", "passResult"]], on=["gameId", "playId"], how="left")
rel = rel[rel.passResult.isin(["C", "I", "IN"])]; rel["c"] = (rel.passResult == "C").astype(int)
print(f"(a) openness@release vs completion (n={len(rel)}):", flush=True)
for m in ["o_cmax", "o_real", "o_sum"]:
    print(f"   {m:7s} r={pointbiserialr(rel.c, rel[m])[0]:+.3f}  C={rel[rel.c==1][m].mean():.2f} I={rel[rel.c==0][m].mean():.2f}")

trk = pd.read_csv("data/week1.csv", usecols=["gameId", "playId", "nflId", "frameId", "x", "y", "team", "event"])
route = pd.read_csv("data/pffScoutingData.csv").query("pff_role=='Pass Route'")[["gameId", "playId", "nflId"]]
af = trk[trk.event == "pass_arrived"].groupby(["gameId", "playId"]).frameId.min().rename("af").reset_index()
ball = trk[trk.team == "football"].merge(af, on=["gameId", "playId"]); ball = ball[ball.frameId == ball.af][["gameId", "playId", "x", "y"]].rename(columns={"x": "bx", "y": "by"})
rec = trk.dropna(subset=["nflId"]).merge(route, on=["gameId", "playId", "nflId"]).merge(af, on=["gameId", "playId"])
rec = rec[rec.frameId == rec.af].merge(ball, on=["gameId", "playId"]); rec["d"] = np.hypot(rec.x - rec.bx, rec.y - rec.by)
tgt = rec.loc[rec.groupby(["gameId", "playId"]).d.idxmin()][["gameId", "playId", "nflId"]].rename(columns={"nflId": "target"})
df["arg_nfl"] = df.o_arg.map(inv_rec)
pred = df[df.o_real > 0].groupby(["gameId", "playId"]).arg_nfl.agg(lambda s: s.value_counts().index[0]).rename("pred").reset_index()
nr = route.groupby(["gameId", "playId"]).size().rename("nrec").reset_index()
mt = tgt.merge(pred, on=["gameId", "playId"]).merge(nr, on=["gameId", "playId"]); mt["hit"] = (mt.target == mt.pred).astype(int)
print(f"(c) target identity (n={len(mt)}): argmax-realizable==target {mt.hit.mean():.3f}  baseline {1/mt.nrec.mean():.3f}", flush=True)
