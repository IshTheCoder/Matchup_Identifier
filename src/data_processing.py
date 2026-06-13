from typing import Tuple

import numpy as np
import pandas as pd

# pre-snap matchup-prior features, in a fixed order. All are PAIR-relational (vary
# across the candidate rusher k) so they survive the softmax-over-k; play-level
# covariates (down, yardsToGo) appear ONLY as interactions with a pair feature.
PRIOR_FEATURE_NAMES = [
    "lat_off",            # y_blocker - y_rusher (signed)
    "abs_lat",            # |y_blocker - y_rusher|
    "lon_off",            # x_blocker - x_rusher (signed)
    "dist",               # snap distance blocker<->rusher
    "same_side",          # blocker and rusher on the same side of the qb (indicator)
    "edge_on_side",       # rusher is the outermost rusher on the blocker's side (indicator)
    "rel_cos",            # cos(dir_rusher - dir_blocker) at snap
    "rusher_edge",        # rusher lined up as an edge defender (indicator, pff_positionLinedUp)
    "rusher_edge_x_side", # rusher_edge * same_side
    "down_x_edge",        # down * edge_on_side
    "ytg_x_abslat",       # yardsToGo * abs_lat
]

# defensive alignments treated as "edge" (outside) rushers
_EDGE_ALIGN = {"LE", "RE", "LEO", "REO", "LOLB", "ROLB", "RUSH"}


def possession_to_prior_features(
    poss: pd.DataFrame, plays_df: pd.DataFrame, pff_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pre-snap pair features X (j, k, F) for one possession's initial-matchup prior

    Axes are aligned to possession_to_voxel: blockers (j) sorted by nflId, rushers (k)
    sorted by nflId_pr. F == len(PRIOR_FEATURE_NAMES). Returns (X, blocker_ids,
    rusher_ids). Uses only pre-snap (ball_snap frame) geometry + alignment; play-level
    down/yardsToGo enter only as interactions. pff_blockType / pff_nflIdBlockedPlayer
    are intentionally NOT used (outcome leakage).
    """
    game_id = poss["gameId"].iloc[0]
    play_id = poss["playId"].iloc[0]
    rusher_ids = np.sort(poss["nflId_pr"].unique())
    blocker_ids = np.sort(poss["nflId"].unique())
    k, j = len(rusher_ids), len(blocker_ids)

    snap = poss[poss["event"] == "ball_snap"]
    if snap.empty:  # fall back to the earliest frame
        snap = poss[poss["frameId"] == poss["frameId"].min()]

    def _by(group_col, cols, ids):
        first = snap.groupby(group_col)[cols].first()
        return first.reindex(ids).to_numpy(dtype=float)

    rush = _by("nflId_pr", ["x_smooth_pr", "y_smooth_pr", "dir"], rusher_ids)  # (k,3)
    blk = _by("nflId", ["x_smooth", "y_smooth", "dir_pb"], blocker_ids)        # (j,3)
    y_qb = float(snap["y_smooth_qb"].iloc[0])

    rx, ry, rdir = rush[:, 0], rush[:, 1], rush[:, 2]
    bx, by, bdir = blk[:, 0], blk[:, 1], blk[:, 2]

    # side relative to the qb (treat 0 as +1)
    rush_side = np.where(ry - y_qb >= 0, 1.0, -1.0)            # (k,)
    blk_side = np.where(by - y_qb >= 0, 1.0, -1.0)            # (j,)
    rush_lat = np.abs(ry - y_qb)                              # (k,)

    # outermost rusher on each side
    edge_idx = {}
    for s in (1.0, -1.0):
        on = np.where(rush_side == s)[0]
        if on.size:
            edge_idx[s] = on[np.argmax(rush_lat[on])]

    # rusher alignment (edge?) from pff_positionLinedUp
    pff_play = pff_df[(pff_df.gameId == game_id) & (pff_df.playId == play_id)]
    align = pff_play.set_index("nflId")["pff_positionLinedUp"]
    rusher_edge = np.array(
        [1.0 if align.get(rid, "") in _EDGE_ALIGN else 0.0 for rid in rusher_ids]
    )  # (k,)

    play = plays_df[(plays_df.gameId == game_id) & (plays_df.playId == play_id)]
    down = float(play["down"].iloc[0]) if len(play) else 0.0
    ytg = float(play["yardsToGo"].iloc[0]) if len(play) else 0.0

    # broadcast pair features: rows=blocker j, cols=rusher k
    lat_off = by[:, None] - ry[None, :]                       # (j,k)
    lon_off = bx[:, None] - rx[None, :]
    dist = np.hypot(lat_off, lon_off)
    abs_lat = np.abs(lat_off)
    same_side = (blk_side[:, None] == rush_side[None, :]).astype(float)
    rel_cos = np.cos(np.deg2rad(rdir[None, :] - bdir[:, None]))
    edge_on_side = np.zeros((j, k))
    for jj in range(j):
        kk = edge_idx.get(blk_side[jj])
        if kk is not None:
            edge_on_side[jj, kk] = 1.0
    rusher_edge_b = np.broadcast_to(rusher_edge[None, :], (j, k))

    X = np.stack(
        [
            lat_off,
            abs_lat,
            lon_off,
            dist,
            same_side,
            edge_on_side,
            rel_cos,
            rusher_edge_b,
            rusher_edge_b * same_side,
            down * edge_on_side,
            ytg * abs_lat,
        ],
        axis=-1,
    )  # (j, k, F)
    return X, blocker_ids, rusher_ids


def possession_to_voxel(
    data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """takes one possession and turns it into a volume for E-M algorithm

    Args:
        data (pd.DataFrame): dataframe of one possession

    Returns:
        np.ndarray: (t x 2 array of qb position), (t x k x 2 array of pass rush positions), (t x j x 2 array of pass blocker positions)
    """

    B = np.array(
        [
            val
            for val in data.groupby("time")
            .apply(lambda x: x[["x_smooth_qb", "y_smooth_qb"]].head(1).values.flatten())
            .reset_index()
            .sort_values("time")[0]
        ]
    )

    D = np.stack(
        data.groupby(["nflId", "time"])
        .apply(lambda x: x[["x_smooth", "y_smooth", "dir_pb"]].head(1).values.flatten())
        .reset_index()
        .groupby("nflId")
        .apply(lambda x: np.array(x.sort_values("time")[0].values.tolist()))
        .reset_index()[0]
        .values.tolist(),
        axis=-1,
    ).swapaxes(-2, -1)

    O = np.stack(
        data.groupby(["nflId_pr", "time"])
        .apply(
            lambda x: x[["x_smooth_pr", "y_smooth_pr", "dir"]].head(1).values.flatten()
        )
        .reset_index()
        .groupby("nflId_pr")
        .apply(lambda x: np.array(x.sort_values("time")[0].values.tolist()))
        .reset_index()[0]
        .values.tolist(),
        axis=-1,
    ).swapaxes(-2, -1)
    return B, O, D


def voxels_to_design_response(
    D: np.ndarray, O: np.ndarray, B: np.ndarray, I: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """turns

    Args:
        D (np.ndarray): t x j x 2 matrix of defender positions
        O (np.ndarray): t x k x 2 matrix of pass rusher positions
        B (np.ndarray): t  x 2 array of quarterback positions
        I (np.ndarray): (t) x j x k array of probable assignments

    Returns:
        np.ndarray: tuple of n x 2 array, and n x 1 array, n = 2 * j x k * t
    """
    _, j, _ = D.shape
    _, k, _ = O.shape
    D = np.repeat(D[:, np.newaxis, :, :], k, axis=1)
    O = np.repeat(O[:, :, np.newaxis, :], j, axis=2)
    B = np.repeat(
        np.repeat(B[:, np.newaxis, :], j, axis=1)[:, np.newaxis, :, :], k, axis=1
    )
    # D, O, B above are all broadcast to (t, k, j, 2); X/y are flattened in C-order
    # so the rows run (t, k, j, coord) with the x/y coordinate FASTEST (the two
    # coordinates of one (t, k, j) triple are adjacent rows). Two fixes vs the
    # original:
    #   1. The responsibilities I arrive as (t, j, k); transpose to (t, k, j) so they
    #      match the (t, k, j) ordering of X/y (mattered whenever j != k).
    #   2. Each triple owns two adjacent rows (x then y) sharing the same weight, so
    #      repeat each weight twice (interleaved). The old hstack([I, I]) laid the
    #      weights out as two halves, which only aligns if coords are blocked, not
    #      interleaved -> it mismatched every y-row.
    I = np.transpose(I, (0, 2, 1))
    I = np.repeat(I.flatten(), 2)
    X = np.column_stack([O[:, :, :, 0:2].flatten(), B.flatten()])
    y = D[:, :, :, 0:2].flatten()[:, np.newaxis]
    return X, y, I
