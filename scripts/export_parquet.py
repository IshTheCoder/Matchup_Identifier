"""One-off: export the already-fit MCMC models (their pickled posterior samples + designs) to parquet
in model_outputs/ -- no model re-run. Future fits also write parquet via src/model_io (wired into the
drivers). Run from the repo root:  python3 scripts/export_parquet.py"""
import sys
import pickle
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import model_io as mio

players = pd.read_csv("data/players.csv").set_index("nflId")
ENC_KEY = {"rusher": "rusher", "blocker": "blocker", "quarterback": "qb"}


def _ids(enc, kind, n):
    inv = {v: k for k, v in enc[ENC_KEY[kind]].items()}
    return [inv[i] for i in range(n)]


# ---- play-level plus-minus (phase 2.5): rusher / blocker / quarterback effects ----
data, enc = pickle.load(open("play_design_phase25.pkl", "rb"))
s = pickle.load(open("play_model_samples_phase25.pkl", "rb"))
mask = np.asarray(data["mask"]); rid = np.asarray(data["rusher_ids"])
bid = np.asarray(data["blocker_ids"]); asg = np.asarray(data["assignment"])
snaps = {
    "rusher": np.bincount(rid[mask > 0].ravel(), minlength=int(data["N_rushers"])),
    "blocker": np.bincount(bid[(asg.sum(1) > 0)].ravel(), minlength=int(data["N_blockers"])),
    "quarterback": np.bincount(np.asarray(data["quarterback_ids"]), minlength=int(data["N_quarterbacks"])),
}
for kind in ("rusher", "blocker", "quarterback"):
    draws = mio.reconstruct_draws(s, data, kind)
    mio.export_effect(f"playpm_{kind}_phase25", draws, _ids(enc, kind, draws.shape[1]), players, snaps=snaps[kind])
    print(f"  playpm_{kind}: draws {draws.shape}", flush=True)

# ---- continuous dose model (blocker, baseline-conditioned, converged MCMC) ----
dd, denc = pickle.load(open("continuous_design_delta_filtered_8wk.pkl", "rb"))
ds = pickle.load(open("blocker_dose_samples_filtered_baseline_mcmc.pkl", "rb"))
draws = mio.reconstruct_draws(ds, dd, "blocker")              # 'blocker_effect' direct site
mio.export_effect("dose_blocker_baseline", draws, _ids(denc, "blocker", draws.shape[1]), players)
print(f"  dose_blocker: draws {draws.shape}", flush=True)

import os
print("wrote:", ", ".join(sorted(os.listdir(mio.MODEL_DIR))))
