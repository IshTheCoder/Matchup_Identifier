"""S1 validity (stronger than completion): (b) CONSTRUCT validity -- space-control openness vs raw
receiver-nearest-defender separation (computed from the design); (c) TARGET identity -- is the
argmax-openness receiver the one actually thrown to (target = Pass Route receiver nearest the football
at the 'pass_arrived' event). Baseline match rate = 1/n_receivers (~0.20)."""
import sys, pickle
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, "src")

data, enc = pickle.load(open("alltwentytwo_design_1wk.pkl", "rb"))
df = pd.read_csv("openness_week1.csv")
N = len(data["x_qb"]); inv_rec = {v: k for k, v in enc["receiver"].items()}

# (b) construct validity: per-frame max receiver separation from nearest coverage defender
max_sep = np.full(N, np.nan)
for n in range(N):
    rm = data["rec_mask"][n] > 0; cm = data["cov_mask"][n] > 0
    if rm.sum() == 0 or cm.sum() == 0:
        continue
    d = cdist(data["x_rec"][n][rm], data["x_cov"][n][cm]).min(1)   # nearest defender per receiver
    max_sep[n] = d.max()
df["max_sep"] = max_sep
v = df.dropna(subset=["max_sep"])
print(f"=== S1(b) construct validity: space-control openness vs raw separation (n={len(v):,} frames) ===", flush=True)
for m in ["open_raw", "open_real", "open_max"]:
    print(f"  {m:9s} Pearson(.,max_sep)={pearsonr(v[m], v.max_sep)[0]:+.3f}  Spearman={spearmanr(v[m], v.max_sep)[0]:+.3f}")

# (c) target identity: target = Pass Route receiver nearest the football at pass_arrived
trk = pd.read_csv("data/week1.csv", usecols=["gameId", "playId", "nflId", "frameId", "x", "y", "team", "event"])
pff = pd.read_csv("data/pffScoutingData.csv")
route = pff[pff.pff_role == "Pass Route"][["gameId", "playId", "nflId"]]
arr = (trk[trk.event == "pass_arrived"].groupby(["gameId", "playId"]).frameId.min().rename("af").reset_index())
ball = trk[trk.team == "football"].merge(arr, on=["gameId", "playId"])
ball = ball[ball.frameId == ball.af][["gameId", "playId", "x", "y"]].rename(columns={"x": "bx", "y": "by"})
rec = trk.dropna(subset=["nflId"]).merge(route, on=["gameId", "playId", "nflId"]).merge(arr, on=["gameId", "playId"])
rec = rec[rec.frameId == rec.af].merge(ball, on=["gameId", "playId"])
rec["d2ball"] = np.hypot(rec.x - rec.bx, rec.y - rec.by)
target = rec.loc[rec.groupby(["gameId", "playId"]).d2ball.idxmin()][["gameId", "playId", "nflId"]] \
    .rename(columns={"nflId": "target"})

# our prediction: modal argmax-openness receiver over each play's window
df["arg_nfl"] = df.open_arg.map(inv_rec)
pred = (df[df.open_max > 0].groupby(["gameId", "playId"]).arg_nfl
        .agg(lambda s: s.value_counts().index[0]).rename("pred").reset_index())
nrec = df.groupby(["gameId", "playId"]).open_arg.first().reset_index()  # placeholder for n
nr = (pd.read_csv("data/pffScoutingData.csv").query("pff_role=='Pass Route'")
      .groupby(["gameId", "playId"]).size().rename("nrec").reset_index())
mt = target.merge(pred, on=["gameId", "playId"]).merge(nr, on=["gameId", "playId"])
mt["hit"] = (mt.target == mt.pred).astype(int)
base = (1.0 / mt.nrec).mean()
print(f"\n=== S1(c) target identity (n={len(mt)} pass plays) ===", flush=True)
print(f"  argmax-openness == target: {mt.hit.mean():.3f}   (random baseline 1/n_rec = {base:.3f})")
