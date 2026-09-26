"""Tidy posterior draws for the ridgeline (caterpillar) figures -> model_outputs/ridge_draws.parquet

Every paper table that reports a credible interval gets a companion density-ridge figure showing the
same players it tabulates. This script does the SELECTION (which players, which position facet, the
same snap thresholds the table scripts use) and emits one long-format frame; src/make_ridge_figures.R
does the drawing. Keeping selection here means the figures and the tables cannot drift apart.

Columns: table, metric, pos (facet key), player_pos (for colour), nflId, name,
         grp (Top/Bottom), rank, draw, value
Run from the repo root:  python3 scripts/export_ridge_draws.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
MODEL_DIR = "model_outputs"
OUT = f"{MODEL_DIR}/ridge_draws.parquet"

# thresholds mirror the table generators exactly
MIN_SNAPS = 50        # PFF snaps (pass rush / pass block / dropbacks)
MIN_CONT_SNAPS = 100  # compare_blocker_metrics.py leaderboard
MIN_SPELLS = 40       # block_hold_tables.py
RUSH_POS = ["Edge", "DT", "NT"]
BLK_POS = ["T", "G", "C"]

frames = []


def _add(table, metric, sel, draws_of, n_top, pos_col="pos"):
    """sel: DataFrame with nflId/name/pos/effect (posterior mean) already filtered.
    draws_of: nflId -> 1-D array of draws. Takes the top and bottom n_top per position.

    `pos` is the facet key (a position for the faceted tables, "All" otherwise); `player_pos`
    is the player's own position, which drives the colour so one hue means one position in
    every figure -- including the unfaceted QB and continuous-time panels."""
    for pos, d in sel.groupby(pos_col):
        d = d.sort_values("effect", ascending=False)
        picks = [("Top", d.head(n_top)), ("Bottom", d.tail(n_top).iloc[::-1])]
        if len(d) <= 2 * n_top:  # too few players to split without overlap; show them all as one block
            picks = [("Top", d)]
        for grp, dd in picks:
            for rank, (_, r) in enumerate(dd.iterrows(), 1):
                v = np.asarray(draws_of[r.nflId], dtype=float)
                frames.append(pd.DataFrame({
                    "table": table, "metric": metric, "pos": pos,
                    "player_pos": r.get("player_pos", r["pos"]), "nflId": r.nflId,
                    "name": r["name"], "grp": grp, "rank": rank,
                    "draw": np.arange(len(v)), "value": v,
                }))


def _center_by_position(sel, draws_of, pos_col="pos"):
    """Per draw, subtract that position's mean effect -> {nflId: centred draws}.

    Applied to the continuous-time plus-minus effects, whose scale carries a
    position-level offset that is not a statement about any individual. Subtracting it draw by
    draw also removes the shared uncertainty in that offset, so each panel is centred on zero =
    the average player at that position.

    The centring pool is every player passing the table's snap filter for that position, NOT just
    the plotted top/bottom: those are selected for being extreme, so centring on them would be
    circular. Within a position the subtraction is a common shift per draw, so the within-position
    ordering (and hence which players are plotted) is unchanged."""
    out = {}
    for _, d in sel.groupby(pos_col):
        ids = d.nflId.tolist()
        M = np.stack([np.asarray(draws_of[i], dtype=float) for i in ids])   # (n_players, n_draws)
        c = M.mean(0)                                                        # per-draw position mean
        out.update({i: M[k] - c for k, i in enumerate(ids)})
    return out


def _parquet_draws(name):
    """model_outputs/{name}_draws.parquet -> (summary DataFrame, {nflId: draws})"""
    dr = pd.read_parquet(f"{MODEL_DIR}/{name}_draws.parquet")
    su = pd.read_parquet(f"{MODEL_DIR}/{name}_summary.parquet")
    return su, {k: g["value"].to_numpy() for k, g in dr.groupby("nflId")}


# PFF snap counts per role: the continuous design carries no play ids, so its exports have no
# snap column. The continuous-time leaderboard below already filters on PFF pass-block snaps.
_pff = pd.read_csv("data/pffScoutingData.csv", usecols=["nflId", "pff_role"])
PFF_SNAPS = {role: _pff[_pff.pff_role == role].groupby("nflId").size()
             for role in ("Pass Rush", "Pass Block", "Pass")}

# ------------------------------------------------ continuous-time plus-minus, by position --
for kind, table, metric, keep_pos, role in [
        ("rusher", "rusher_cont", "R[j]^Delta", RUSH_POS, "Pass Rush"),
        ("blocker", "blocker_cont", "B[b]^Delta", BLK_POS, "Pass Block")]:
    su, dmap = _parquet_draws(f"dose_{kind}_baseline")
    su["snaps"] = su.nflId.map(PFF_SNAPS[role]).fillna(0)
    if kind == "rusher":
        su["pos"] = su["pos"].replace({"DE": "Edge", "OLB": "Edge"})
    su = su[(su.snaps >= MIN_SNAPS) & su.pos.isin(keep_pos)].rename(columns={"mean": "effect"})
    dmap = _center_by_position(su, dmap)                 # centre on the per-draw position mean
    su["effect"] = su.nflId.map(lambda i: dmap[i].mean())
    _add(table, metric, su, dmap, 5)
    print(f"  {table} (position-centred): {su.pos.value_counts().to_dict()}", flush=True)

# ----------------------------------------------------- continuous-time QB strain suppression --
su, dmap = _parquet_draws("dose_quarterback_baseline")
su["snaps"] = su.nflId.map(PFF_SNAPS["Pass"]).fillna(0)  # PFF dropbacks
su = su[su.snaps >= MIN_SNAPS].copy()
su["effect"] = -su["mean"]                      # suppression = -Q^Delta_q
su["player_pos"] = "QB"
su["pos"] = "All"
_add("qb_cont", "-Q[q]^Delta", su, {k: -v for k, v in dmap.items()}, 10)
print(f"  qb_cont: {len(su)} QBs", flush=True)

# ------------------------------------------------------------ continuous-time leaderboard --
su, dmap = _parquet_draws("dose_blocker_baseline")
snaps = pd.read_csv("blocker_dose_rankings_filtered_baseline_mcmc.csv")[["nflId", "pb_snaps"]]
su = su.merge(snaps, on="nflId", how="inner").query("pb_snaps >= @MIN_CONT_SNAPS").copy()
su = su[su.pos.isin(BLK_POS)].copy()
# centre within the player's OWN position before selecting: this panel mixes T/G/C, so without it
# the leaderboard would rank across positions on an uncentred scale.
dmap = _center_by_position(su, dmap)
su["effect"] = su.nflId.map(lambda i: dmap[i].mean())
su["player_pos"] = su["pos"]
su["pos"] = "All"
_add("blocker_continuous", "B[b]", su, dmap, 12)
print(f"  blocker_continuous (position-centred): {len(su)} blockers >= {MIN_CONT_SNAPS} snaps",
      flush=True)

# ------------------------------------------------------ opponent-adjusted block-hold rating --
hs = pickle.load(open("block_hold_discrete_samples.pkl", "rb"))
hold_draws = -(np.asarray(hs["sigma_b"])[:, None] * np.asarray(hs["z_blocker"]))   # hold = -u_b
ev = pd.read_csv("block_failure_events.csv")
benc = {b: i for i, b in enumerate(np.sort(ev.nflId.unique()))}                    # same encoder as the fit
hr = pd.read_csv("block_hold_discrete_ratings.csv")
hr = hr[(hr.spells >= MIN_SPELLS) & hr.pos.isin(BLK_POS)].rename(columns={"hold": "effect"})
_add("block_hold", "-u[b]", hr, {r.nflId: hold_draws[:, benc[r.nflId]] for _, r in hr.iterrows()}, 5)
print(f"  block_hold: {len(hr)} blockers >= {MIN_SPELLS} spells", flush=True)

# ------------------------------------------------------------------------------- write --
os.makedirs(MODEL_DIR, exist_ok=True)
out = pd.concat(frames, ignore_index=True)
out.to_parquet(OUT, index=False)
print(f"\nwrote {OUT}: {len(out):,} rows, "
      f"{out.groupby(['table','pos']).ngroups} (table,position) panels")
print(out.groupby(["table", "pos"]).name.nunique().rename("players").to_string())
