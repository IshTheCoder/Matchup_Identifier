"""S1 (revised): openness = receiver control share integrated over OCCUPIED downfield space
(ΣI>τ), so empty field (where the share is numerically ill-defined) contributes nothing. Computes
raw/realizable/per-receiver openness AND validates in one pass: (a) completion, (b) construct vs raw
separation, (c) target identity. Iterate params at the top."""
import sys, pickle
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pointbiserialr, pearsonr, spearmanr
sys.path.insert(0, "src")
import pitch_control as pc

REC_R, COV_R, FWD, D0, OCC = 2.5, 3.0, 35.0, 15.0, 0.15
data, enc = pickle.load(open("alltwentytwo_design_1wk.pkl", "rb"))
N = len(data["x_qb"]); ps = data["play_start"]; inv_rec = {v: k for k, v in enc["receiver"].items()}
print(f"N={N:,} frames  params REC_R={REC_R} COV_R={COV_R} FWD={FWD} D0={D0} OCC={OCC}", flush=True)

o_raw = np.zeros(N); o_real = np.zeros(N); o_max = np.zeros(N); o_arg = np.zeros(N, int); msep = np.full(N, np.nan)
for n in range(N):
    los = float(data["los_x"][n])
    G, gx, gy, cell = pc.field_grid(los, forward=FWD, back=2.0, nx=80, ny=52)
    down = G[:, 0] > los
    Irec, ri = pc.team_influences(G, data["x_rec"][n], data["v_rec"][n], data["a_rec"][n], None, data["rec_mask"][n], radius=REC_R)
    Icov, ci = pc.team_influences(G, data["x_cov"][n], data["v_cov"][n], data["a_cov"][n], None, data["cov_mask"][n], radius=COV_R)
    if len(ri) == 0:
        continue
    tot = Irec.sum(0) + Icov.sum(0)
    occ = (tot > OCC).astype(np.float32)                       # only count occupied space
    w = Irec / (tot + 1e-9)                                    # per-receiver share (Kr,P)
    wtot = w.sum(0)
    kern = np.exp(-np.hypot(G[:, 0] - data["x_qb"][n, 0], G[:, 1] - data["x_qb"][n, 1]) / D0)
    m = down * occ
    o_raw[n] = cell * (wtot * m).sum()
    o_real[n] = cell * (wtot * m * kern).sum()
    sp = cell * (w * m).sum(1)
    j = int(np.argmax(sp)); o_max[n] = sp[j]; o_arg[n] = int(data["rec_slot_id"][n, ri[j]])
    rm = data["rec_mask"][n] > 0; cm = data["cov_mask"][n] > 0
    if rm.sum() and cm.sum():
        msep[n] = cdist(data["x_rec"][n][rm], data["x_cov"][n][cm]).min(1).max()

df = pd.DataFrame({"gameId": data["gameId"], "playId": data["playId"], "frameId": data["frame_id"],
                   "open_raw": o_raw, "open_real": o_real, "open_max": o_max, "open_arg": o_arg,
                   "max_sep": msep, "play_row": data["play_row"]})
df.to_csv("openness_week1.csv", index=False)

rel = df.groupby("play_row").tail(1).merge(
    pd.read_csv("data/plays.csv")[["gameId", "playId", "passResult"]], on=["gameId", "playId"], how="left")
rel = rel[rel.passResult.isin(["C", "I", "IN"])]; rel["c"] = (rel.passResult == "C").astype(int)
print(f"\n(a) openness@release vs completion (n={len(rel)}):", flush=True)
for m in ["open_raw", "open_real", "open_max"]:
    print(f"   {m:9s} r={pointbiserialr(rel.c, rel[m])[0]:+.3f}  C={rel[rel.c==1][m].mean():.1f} I={rel[rel.c==0][m].mean():.1f}")
v = df.dropna(subset=["max_sep"])
print(f"\n(b) construct vs raw separation (n={len(v):,}):", flush=True)
for m in ["open_raw", "open_real", "open_max"]:
    print(f"   {m:9s} Pearson={pearsonr(v[m], v.max_sep)[0]:+.3f}  Spearman={spearmanr(v[m], v.max_sep)[0]:+.3f}")

trk = pd.read_csv("data/week1.csv", usecols=["gameId", "playId", "nflId", "frameId", "x", "y", "team", "event"])
route = pd.read_csv("data/pffScoutingData.csv").query("pff_role=='Pass Route'")[["gameId", "playId", "nflId"]]
af = trk[trk.event == "pass_arrived"].groupby(["gameId", "playId"]).frameId.min().rename("af").reset_index()
ball = trk[trk.team == "football"].merge(af, on=["gameId", "playId"]); ball = ball[ball.frameId == ball.af][["gameId", "playId", "x", "y"]].rename(columns={"x": "bx", "y": "by"})
rec = trk.dropna(subset=["nflId"]).merge(route, on=["gameId", "playId", "nflId"]).merge(af, on=["gameId", "playId"])
rec = rec[rec.frameId == rec.af].merge(ball, on=["gameId", "playId"]); rec["d"] = np.hypot(rec.x - rec.bx, rec.y - rec.by)
tgt = rec.loc[rec.groupby(["gameId", "playId"]).d.idxmin()][["gameId", "playId", "nflId"]].rename(columns={"nflId": "target"})
df["arg_nfl"] = df.open_arg.map(inv_rec)
pred = df[df.open_max > 0].groupby(["gameId", "playId"]).arg_nfl.agg(lambda s: s.value_counts().index[0]).rename("pred").reset_index()
nr = route.groupby(["gameId", "playId"]).size().rename("nrec").reset_index()
mt = tgt.merge(pred, on=["gameId", "playId"]).merge(nr, on=["gameId", "playId"]); mt["hit"] = (mt.target == mt.pred).astype(int)
print(f"\n(c) target identity (n={len(mt)}): argmax-openness==target {mt.hit.mean():.3f}  baseline {1/mt.nrec.mean():.3f}", flush=True)
