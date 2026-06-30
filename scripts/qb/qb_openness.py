"""Derive downfield-coverage openness per QB-frame from the raw all-22 tracking + PFF role labels:
for each receiver (PFF 'Pass Route'), separation = distance to the nearest coverage defender (PFF
'Coverage'); aggregate per (game,play,frame) to max_sep (best open receiver), mean_sep, n_open.
Restricted to the QB-frames the escape model uses. -> receiver_openness.csv"""
import pickle
import numpy as np, pandas as pd

data, _ = pickle.load(open("qb_force_design_8wk.pkl", "rb"))
keys = pd.DataFrame({"gameId": data["gameId"], "playId": data["playId"],
                     "frameId": data["frame_id"]}).drop_duplicates()
print(f"target QB-frames: {len(keys):,}", flush=True)
pff = pd.read_csv("data/pffScoutingData.csv")[["gameId", "playId", "nflId", "pff_role"]]
route = pff[pff.pff_role == "Pass Route"][["gameId", "playId", "nflId"]]
cover = pff[pff.pff_role == "Coverage"][["gameId", "playId", "nflId"]]

out = []
for w in range(1, 9):
    trk = pd.read_csv(f"data/week{w}.csv", usecols=["gameId", "playId", "nflId", "frameId", "x", "y"])
    trk = trk.merge(keys, on=["gameId", "playId", "frameId"])      # only window frames we model
    rec = trk.merge(route, on=["gameId", "playId", "nflId"])
    dfd = trk.merge(cover, on=["gameId", "playId", "nflId"])
    m = rec.merge(dfd, on=["gameId", "playId", "frameId"], suffixes=("_r", "_d"))
    m["sep"] = np.hypot(m.x_r - m.x_d, m.y_r - m.y_d)
    sep = m.groupby(["gameId", "playId", "frameId", "nflId_r"])["sep"].min().reset_index()  # nearest defender
    sep["open"] = (sep["sep"] > 3).astype(int)
    agg = sep.groupby(["gameId", "playId", "frameId"]).agg(
        max_sep=("sep", "max"), mean_sep=("sep", "mean"), n_open=("open", "sum")).reset_index()
    out.append(agg); print(f"  week {w}: {len(agg):,} frames", flush=True)

op = pd.concat(out, ignore_index=True)
op.to_csv("receiver_openness.csv", index=False)
print(f"wrote receiver_openness.csv ({len(op):,} frames; coverage of QB-frames "
      f"{len(op)/len(keys):.2f})  mean max_sep={op.max_sep.mean():.2f}yd", flush=True)
