"""Phase 2.5 block-failure survival metrics from the structured-null assignments.

For each (game, play, blocker) we find the first frame the blocker enters the null/disengaged
state AND it is a genuine BEAT (not a drive-win or release), using geometry + STRAIN:
  beaten  = in null AND rusher goal-side (d(rusher,QB) < d(blocker,QB)) AND rusher STRAIN > thr
  drive-win = in null AND rusher pushed away (not goal-side, STRAIN <= 0)
  release   = in null otherwise (no threatening rusher)
Time-to-beat is a survival quantity (most blocks never fail -> right-censored at the throw).
We report, with a Kaplan-Meier product-limit estimator (no lifelines dependency):
  - blocker ENGAGEMENT: KM median + restricted-mean-survival-time (RMST) per blocker, plus an
    opponent-adjusted residual (vs the rusher-group RMST of the rushers actually faced);
  - rusher SHEDDING: KM median + RMST grouped by the blocker's primary engaged rusher
    (shorter time-to-beat => that rusher beats blocks faster);
  - per-blocker drive-win rate (a positive complement).
Writes survival_by_blocker.csv, survival_by_rusher.csv, tables/block_engagement.tex,
tables/rusher_shedding.tex.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import feature_engineering as fe  # noqa: E402

NULL_RUSHER = -1
STRAIN_THR = 0.0       # rusher "closing" threshold for a beat (STRAIN>0 => moving toward QB)
DT = 0.1               # seconds per frame
MIN_PLAYS = 40         # min blocks for a ranked player
RMST_HORIZON = 40      # frames (~4 s) for restricted-mean-survival-time


# ----------------------------------------------------------- KM / RMST --
def km(times, events, horizon=RMST_HORIZON):
    """product-limit survival of (times, events 1=beat/0=censored). Returns (median, rmst)."""
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    order = np.argsort(times)
    times, events = times[order], events[order]
    n = len(times)
    if n == 0:
        return np.nan, np.nan
    S, prev_t, med, rmst = 1.0, 0.0, np.nan, 0.0
    for t in np.unique(times):
        at_risk = (times >= t).sum()
        d = events[times == t].sum()
        rmst += S * (min(t, horizon) - min(prev_t, horizon))  # area of the prior step
        if d > 0 and at_risk > 0:
            S *= (1.0 - d / at_risk)
        if np.isnan(med) and S <= 0.5:
            med = t
        prev_t = t
    rmst += S * (horizon - min(prev_t, horizon))  # tail to the horizon
    return med, rmst


# ------------------------------------------------- assignment reduction --
def per_blocker_frame(assignment_path):
    """-> DataFrame (gameId, playId, frameId, nflId, primary_rusher, in_null)"""
    ad = pd.read_csv(assignment_path,
                     usecols=["gameId", "playId", "frameId", "nflId", "nflId_pr", "assignment_probs"],
                     dtype={"assignment_probs": "float64"})
    key = ["gameId", "playId", "frameId", "nflId"]
    nullrows = ad[ad.nflId_pr == NULL_RUSHER][key + ["assignment_probs"]].rename(
        columns={"assignment_probs": "null_prob"})
    rush = ad[ad.nflId_pr != NULL_RUSHER]
    prim = (rush.sort_values("assignment_probs").drop_duplicates(key, keep="last")
            .rename(columns={"nflId_pr": "primary_rusher", "assignment_probs": "primary_prob"}))
    bf = prim.merge(nullrows, on=key, how="left")
    bf["null_prob"] = bf["null_prob"].fillna(0.0)
    bf["in_null"] = bf["null_prob"] > bf["primary_prob"]
    return bf[key + ["primary_rusher", "in_null"]]


def label_beats(bf, weeks=range(8)):
    """join blocker/rusher/QB geometry + rusher STRAIN; label each frame beaten/drive/release"""
    sample = pd.concat([pd.read_csv(f"data/sample_data_week_{i}.csv") for i in weeks],
                       ignore_index=True)
    rf, _ = fe._strain_per_rusher_frame(sample)  # rusher xy, QB xy, d=dist(rusher,QB), strain
    rcols = ["gameId", "playId", "nflId_pr", "frameId", "x_smooth_pr", "y_smooth_pr",
             "x_smooth_qb", "y_smooth_qb", "d", "strain"]
    rf = rf[rcols].drop_duplicates(["gameId", "playId", "nflId_pr", "frameId"])
    blk = (sample.drop_duplicates(["gameId", "playId", "nflId", "frameId"])
           [["gameId", "playId", "nflId", "frameId", "x_smooth", "y_smooth"]])

    bf = bf.merge(rf, left_on=["gameId", "playId", "primary_rusher", "frameId"],
                  right_on=["gameId", "playId", "nflId_pr", "frameId"], how="inner")
    bf = bf.merge(blk, on=["gameId", "playId", "nflId", "frameId"], how="inner")
    d_bq = np.hypot(bf.x_smooth - bf.x_smooth_qb, bf.y_smooth - bf.y_smooth_qb)
    goal_side = bf["d"] < d_bq                       # rusher closer to QB than blocker
    bf["beaten"] = bf.in_null & goal_side & (bf.strain > STRAIN_THR)
    bf["drive_win"] = bf.in_null & (~goal_side) & (bf.strain <= 0)
    return bf


# ------------------------------------------------ first-passage per block --
def first_passage(bf):
    """per (game,play,blocker): time-to-first-beat (frames), event flag, primary rusher, drive flag"""
    bf = bf.sort_values(["gameId", "playId", "nflId", "frameId"])
    rows = []
    for (g, p, b), x in bf.groupby(["gameId", "playId", "nflId"]):
        fr = x.frameId.to_numpy()
        t = np.arange(len(fr))  # 0-based frame index from first tracked frame
        beat = x.beaten.to_numpy()
        # primary rusher for the block = modal engaged rusher over non-null frames
        eng = x[~x.in_null]
        prim = (eng.primary_rusher.mode().iloc[0] if len(eng)
                else x.primary_rusher.mode().iloc[0])
        if beat.any():
            te = int(t[beat][0]); event = 1
        else:
            te = int(t[-1]); event = 0       # censored at the throw
        rows.append((g, p, int(b), int(prim), te, event, int(x.drive_win.any())))
    return pd.DataFrame(rows, columns=["gameId", "playId", "nflId", "primary_rusher",
                                       "time", "event", "drive_win"])


# --------------------------------------------------------------- ranking --
def _km_table(fp, by, players, name_col):
    out = []
    for pid, x in fp.groupby(by):
        med, rmst = km(x.time, x.event)
        out.append({by: pid, "n_blocks": len(x), "beat_rate": x.event.mean(),
                    "km_median_s": med * DT if med == med else np.nan, "rmst_s": rmst * DT})
    df = pd.DataFrame(out)
    df["name"] = df[by].map(players["displayName"])
    df["pos"] = df[by].map(players["officialPosition"]).replace({"DE": "Edge", "OLB": "Edge"})  # merge edge
    return df[df.n_blocks >= MIN_PLAYS].reset_index(drop=True)


def run(assignment_path="assignment_data.csv"):
    players = pd.read_csv("data/players.csv").set_index("nflId")
    print("reducing assignments ...", flush=True)
    bf = per_blocker_frame(assignment_path)
    print("labeling beats (geometry + STRAIN) ...", flush=True)
    bf = label_beats(bf)
    fp = first_passage(bf)
    fp.to_csv("block_failure_events.csv", index=False)

    # rusher RMST first (for opponent adjustment of blockers)
    rush = _km_table(fp, "primary_rusher", players, "name").rename(columns={"primary_rusher": "nflId"})
    rusher_rmst = dict(zip(rush.nflId, rush.rmst_s))
    fp["opp_rmst"] = fp.primary_rusher.map(rusher_rmst)

    blk = _km_table(fp, "nflId", players, "name")
    # opponent-adjusted engagement: blocker mean (time vs faced-rusher difficulty)
    adj = (fp.dropna(subset=["opp_rmst"]).assign(resid=lambda d: d.time * DT - d.opp_rmst)
           .groupby("nflId").resid.mean().rename("engage_resid_s"))
    blk = blk.merge(adj, on="nflId", how="left")
    drive = fp.groupby("nflId").drive_win.mean().rename("drive_win_rate")
    blk = blk.merge(drive, on="nflId", how="left")

    blk.to_csv("survival_by_blocker.csv", index=False)
    rush.to_csv("survival_by_rusher.csv", index=False)

    # ---- tex tables (tackles by engagement; rushers by shedding) ----
    os.makedirs("tables", exist_ok=True)
    T = blk[blk.pos == "T"].copy()
    _tex_block(T.sort_values("rmst_s", ascending=False),
               "tables/block_engagement.tex", "tab:block_engagement",
               "Tackle pass-block engagement (longer survival-to-beat = better), "
               f"min {MIN_PLAYS} blocks", ["name", "rmst_s", "beat_rate", "engage_resid_s"],
               ["Name", "RMST (s)", "Beat rate", "Adj resid (s)"])
    RU = rush.copy()
    _tex_block(RU[RU.pos.isin(["Edge", "DT", "NT"])].sort_values("rmst_s").head(15),
               "tables/rusher_shedding.tex", "tab:rusher_shedding",
               "Pass rushers that beat blocks fastest (shortest survival-to-beat), "
               f"min {MIN_PLAYS} blocks", ["name", "pos", "rmst_s", "beat_rate"],
               ["Name", "Pos", "RMST (s)", "Beat rate"])

    print(f"\nblockers ranked: {len(blk)}  rushers ranked: {len(rush)}")
    print("\nstickiest tackles (engagement, top RMST):")
    print(T.sort_values("rmst_s", ascending=False).head(8)[
        ["name", "rmst_s", "beat_rate", "engage_resid_s", "n_blocks"]].to_string(index=False))
    print("\nmost beatable tackles (lowest RMST):")
    print(T.sort_values("rmst_s").head(8)[
        ["name", "rmst_s", "beat_rate", "n_blocks"]].to_string(index=False))
    print("\nfastest-shedding rushers (lowest RMST):")
    print(RU[RU.pos.isin(["DE", "DT", "NT", "OLB"])].sort_values("rmst_s").head(10)[
        ["name", "pos", "rmst_s", "beat_rate", "n_blocks"]].to_string(index=False))


def _tex_block(df, path, label, caption, cols, headers):
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    body = " \\\\\n".join(" & ".join(fmt(r[c]) for c in cols) for _, r in df.head(15).iterrows())
    spec = "l" + "c" * (len(cols) - 1)
    with open(path, "w") as f:
        f.write("\\begin{table}[h!]\n\\centering\n\\footnotesize\n\\begin{tabular}{" + spec + "}\n"
                "\\toprule\n" + " & ".join(headers) + " \\\\\n\\midrule\n" + body
                + " \\\\\n\\bottomrule\n\\end{tabular}\n\\caption{" + caption + "}\n\\label{"
                + label + "}\n\\end{table}\n")


if __name__ == "__main__":
    run()
