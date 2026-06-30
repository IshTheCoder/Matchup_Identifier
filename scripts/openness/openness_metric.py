"""S1: per-frame OPENNESS over week 1 from the all-22 space-control field.
 - open_raw       = receiver control share integrated over the downfield region (x>LOS), yd^2.
 - open_real      = same, weighted by a throwability kernel exp(-dist_to_QB/D0) ("realizable openness").
 - open_max       = best single receiver's controlled downfield space; open_argmax = his nflId.
VALIDITY GATE: (a) openness at release predicts completion (passResult); (b) realizable openness tracks
the separation-based receiver_openness.csv max_sep; (c) argmax-openness receiver = ball-arrival target."""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import pointbiserialr, pearsonr, spearmanr
sys.path.insert(0, "src")
import pitch_control as pc

REC_R, COV_R, FWD, D0 = 2.5, 3.0, 35.0, 15.0
data, enc = pickle.load(open("alltwentytwo_design_1wk.pkl", "rb"))
N = len(data["x_qb"]); ps = data["play_start"]
inv_rec = {v: k for k, v in enc["receiver"].items()}
print(f"N={N:,} frames, {len(ps)-1} plays", flush=True)

open_raw = np.zeros(N); open_real = np.zeros(N); open_max = np.zeros(N); open_arg = np.zeros(N, int)
for n in range(N):
    los = float(data["los_x"][n])
    G, gx, gy, cell = pc.field_grid(los, forward=FWD, back=2.0, nx=80, ny=52)
    down = G[:, 0] > los
    Irec, ri = pc.team_influences(G, data["x_rec"][n], data["v_rec"][n], data["a_rec"][n],
                                  None, data["rec_mask"][n], radius=REC_R)
    Icov, ci = pc.team_influences(G, data["x_cov"][n], data["v_cov"][n], data["a_cov"][n],
                                  None, data["cov_mask"][n], radius=COV_R)
    if len(ri) == 0:
        continue
    tot = Irec.sum(0) + Icov.sum(0) + 1e-9
    wrec = Irec / tot                                          # per-receiver share (Kr,P)
    kern = np.exp(-np.hypot(G[:, 0] - data["x_qb"][n, 0], G[:, 1] - data["x_qb"][n, 1]) / D0)
    wtot = wrec.sum(0)
    open_raw[n] = cell * wtot[down].sum()
    open_real[n] = cell * (wtot * kern)[down].sum()
    sp = cell * wrec[:, down].sum(1)                           # per-receiver downfield space
    j = int(np.argmax(sp)); open_max[n] = sp[j]; open_arg[n] = int(data["rec_slot_id"][n, ri[j]])

df = pd.DataFrame({"gameId": data["gameId"], "playId": data["playId"], "frameId": data["frame_id"],
                   "open_raw": open_raw, "open_real": open_real, "open_max": open_max,
                   "open_arg": open_arg, "play_row": data["play_row"]})
df.to_csv("openness_week1.csv", index=False)

# (a) completion gate: openness at the RELEASE frame (last frame of window) vs passResult==C
rel = df.groupby("play_row").tail(1).copy()
plays = pd.read_csv("data/plays.csv")[["gameId", "playId", "passResult"]]
rel = rel.merge(plays, on=["gameId", "playId"], how="left")
rel = rel[rel.passResult.isin(["C", "I", "IN"])]; rel["complete"] = (rel.passResult == "C").astype(int)
print(f"\n=== S1(a) openness at release vs completion (n={len(rel)}, {int(rel.complete.sum())} complete) ===", flush=True)
for m in ["open_raw", "open_real", "open_max"]:
    r, pv = pointbiserialr(rel.complete, rel[m])
    print(f"  {m:9s} r={r:+.3f} p={pv:.1e}   complete={rel[rel.complete==1][m].mean():.1f}  inc={rel[rel.complete==0][m].mean():.1f}")

# (b) realizable openness vs separation-based max_sep
try:
    sep = pd.read_csv("receiver_openness.csv")
    j = df.merge(sep, on=["gameId", "playId", "frameId"], how="inner")
    print(f"\n=== S1(b) realizable openness vs separation max_sep (n={len(j):,} matched frames) ===", flush=True)
    print(f"  Pearson(open_real, max_sep)={pearsonr(j.open_real, j.max_sep)[0]:+.3f}  "
          f"Spearman={spearmanr(j.open_real, j.max_sep)[0]:+.3f}")
    print(f"  Pearson(open_max,  max_sep)={pearsonr(j.open_max, j.max_sep)[0]:+.3f}")
except FileNotFoundError:
    print("\n(receiver_openness.csv not found; skipping S1(b))")
