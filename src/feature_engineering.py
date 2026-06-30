import pandas as pd
import numpy as np
from itertools import chain
from typing import Tuple


weeks_used = 8
raw_files = ["data/week" + str(i) + ".csv" for i in range(1, weeks_used + 1)]
pass_rush_files = [f"data/sample_data_week_{i}.csv" for i in range(weeks_used)]


def generate_acceleration_data():
    """
    generates acceleration towards qb in y direction hmm data
    """
    data_list = []
    for raw_file, pass_rush_file in zip(raw_files, pass_rush_files):
        data = pd.read_csv(raw_file)
        processed_week_data = pd.read_csv(pass_rush_file)

        ### get timesteps between snap and first event
        event_instances = (
            data.groupby(["gameId", "playId"])
            .apply(
                lambda x: x[x.event != "None"].drop_duplicates(["time"])[
                    ["time", "event"]
                ]
            )
            .reset_index()
        )
        event_timestamp = (
            event_instances.groupby(["gameId", "playId"])
            .apply(
                lambda x: x[
                    x.event.isin(
                        ["pass_forward", "run", "qb_sack", "qb_strip_sack", "ball_snap"]
                    )
                ][["time", "event"]]
            )
            .reset_index()
        )
        event_timestamp["event"] = event_timestamp["event"].apply(
            lambda x: "action" if x != "ball_snap" else x
        )
        event_timestamp["time"] = pd.to_datetime(event_timestamp["time"])
        event_wide = pd.pivot_table(
            event_timestamp[["gameId", "time", "playId", "event"]],
            columns="event",
            values=["time"],
            index=["gameId", "playId"],
        ).reset_index()
        event_wide.columns = ["gameId", "playId", "action", "ball_snap"]

        ### now merge
        merged_event_processed = pd.merge(event_wide, processed_week_data)
        filtered_processed = merged_event_processed[
            (
                pd.to_datetime(merged_event_processed.time)
                <= merged_event_processed.action
            )
            & (
                pd.to_datetime(merged_event_processed.time)
                >= merged_event_processed.ball_snap
            )
        ]
        pr_filtered = filtered_processed

        #### create physics features

        pr_filtered["x_pr_qb"] = pr_filtered["x_smooth_qb"] - pr_filtered["x_smooth_pr"]
        pr_filtered["y_pr_qb"] = pr_filtered["y_smooth_qb"] - pr_filtered["y_smooth_pr"]

        pr_filtered["pr_qb_norm"] = (
            pr_filtered["x_pr_qb"] ** 2 + pr_filtered["y_pr_qb"] ** 2
        )
        pr_filtered["scalar_projection"] = (
            pr_filtered["x_pr_qb"] * pr_filtered["d2x_smooth_pr"]
            + pr_filtered["y_pr_qb"] * pr_filtered["d2y_smooth_pr"]
        ) / pr_filtered["pr_qb_norm"]
        pr_filtered["d2y_pr_qb"] = (
            pr_filtered["scalar_projection"] * pr_filtered["y_pr_qb"]
        )
        pr_filtered["d2x_pr_qb"] = (
            pr_filtered["scalar_projection"] * pr_filtered["x_pr_qb"]
        )

        ### calculate strain
        pr_filtered["d_ij"] = np.sqrt(
            np.square(pr_filtered["x_smooth_pr"] - pr_filtered["x_smooth_qb"])
            + np.square(pr_filtered["y_smooth_pr"] - pr_filtered["y_smooth_qb"])
        )
        pr_filtered["strain_rate"] = pr_filtered.groupby(
            ["gameId", "playId", "nflId_pr", "nflId"]
        )["d_ij"].transform(lambda x: -np.gradient(x, 0.1))
        pr_filtered["strain_rate"] /= pr_filtered["d_ij"]
        pr_filtered["strain_acceleration"] = pr_filtered.groupby(
            ["gameId", "playId", "nflId_pr", "nflId"]
        )["strain_rate"].transform(lambda x: np.gradient(x, 0.1))

        ### append to list
        data_list.append(pr_filtered)

    pd.concat(data_list).to_csv("processed_data/pass_rusher_features.csv", index=False)


def generate_features(
    assignment_data: pd.DataFrame, acceleration_data: pd.DataFrame
) -> Tuple[pd.DataFrame, dict, dict]:
    """_summary_

    Args:
        assignment_data (pd.DataFrame): prob of assignment
        acceleration_data (pd.DataFrame): acceleration based data

    Returns:
        pd.DataFrame: clean df to use for analysis as well as dictionary params
    """
    assignment_data_agg = (
        assignment_data.groupby(["nflId_pr", "playId", "frameId", "gameId"])
        .apply(lambda x: {i: val for val, i in zip(x.assignment_probs, x.nflId)})
        .reset_index()
    )
    assignment_data_agg.rename(axis=1, mapper={0: "assignment_dict"}, inplace=True)
    final_feature_data = assignment_data_agg.merge(
        acceleration_data.drop_duplicates(["time", "nflId_pr"])
    )
    rusher_encode_map = {
        index: val for index, val in enumerate(set(final_feature_data["nflId_pr"]))
    }
    rusher_encode_inverse_map = {
        rusher_encode_map[index]: index for index in rusher_encode_map
    }
    final_feature_data["rusher_id_model"] = final_feature_data["nflId_pr"].apply(
        lambda x: rusher_encode_inverse_map[x]
    )
    blocker_encode_map = {
        index: val
        for index, val in enumerate(
            set(
                chain.from_iterable(
                    [
                        list(item.keys())
                        for item in final_feature_data["assignment_dict"]
                    ]
                )
            )
        )
    }

    qb_encode_map = {
        index: val for index, val in enumerate(set(final_feature_data["nflId_qb"]))
    }
    final_feature_data_na = final_feature_data.dropna()
    final_feature_data_na = final_feature_data[
        (~np.isinf(final_feature_data.strain_acceleration))
        & (~np.isinf(final_feature_data.strain_rate))
    ]
    final_feature_data_na["y_jt"] = final_feature_data_na[
        "strain_acceleration"
    ] * final_feature_data_na["assignment_dict"].apply(lambda x: sum(x.values()))
    return final_feature_data_na, blocker_encode_map, rusher_encode_map, qb_encode_map


# --------------------------------------------------------------------------- #
# play-level max-strain plus-minus design (the `data` dict for model/models.py)
# --------------------------------------------------------------------------- #
ACTION_EVENTS = ("pass_forward", "autoevent_passforward", "qb_sack",
                 "qb_strip_sack", "run")


def _height_to_inches(h):
    try:
        ft, inch = str(h).split("-")
        return int(ft) * 12 + int(inch)
    except Exception:
        return np.nan


def _clock_seconds(s):
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except Exception:
        return np.nan


def _encode(values):
    """contiguous integer encoding {value: idx}, sorted by value"""
    uniq = sorted(set(np.asarray(list(values)).tolist()))
    return {v: i for i, v in enumerate(uniq)}


def _strain_per_rusher_frame(sample_data):
    """one row per (gameId, playId, nflId_pr, frameId) with STRAIN = -v/d, filtered
    to the snap->action window."""
    rf = sample_data.drop_duplicates(["gameId", "playId", "nflId_pr", "frameId"]).copy()
    rf["d"] = np.hypot(rf["x_smooth_pr"] - rf["x_smooth_qb"],
                       rf["y_smooth_pr"] - rf["y_smooth_qb"])
    rf = rf.sort_values(["gameId", "playId", "nflId_pr", "frameId"])
    rf["strain"] = rf.groupby(["gameId", "playId", "nflId_pr"])["d"].transform(
        lambda x: -np.gradient(x.to_numpy(), 0.1)) / rf["d"]
    action = (sample_data[sample_data.event.isin(ACTION_EVENTS)]
              .groupby(["gameId", "playId"])["frameId"].min().rename("action_frame"))
    rf = rf.merge(action, on=["gameId", "playId"], how="left")
    rf = rf[rf["action_frame"].isna() | (rf["frameId"] <= rf["action_frame"])]
    rf = rf.replace([np.inf, -np.inf], np.nan).dropna(subset=["strain"])
    return rf, action


def build_play_design(sample_data, assignment_data, plays, games, players,
                      normalize_attention=False):
    """assemble the play-level model `data` dict (N = plays, padded rushers/blockers).

    normalize_attention (Phase 2.6): if True, normalize attention per (play, rusher) by the
    effective-blocker count n_j = sum_b theta(b,j), so the countering term becomes the
    AVERAGE engaged blocker effect rather than a count-scaled sum.
    """
    # drop the disengaged/null sentinel (nflId_pr == -1) so attention/effective-blocker
    # sums reflect genuine engagement only (Phase 2/2.5 de-biasing; no-op pre-null)
    assignment_data = assignment_data[assignment_data["nflId_pr"] != -1]
    # --- play-level situation ---
    home = plays["gameId"].map(games.set_index("gameId")["homeTeamAbbr"])
    home_diff = plays["preSnapHomeScore"] - plays["preSnapVisitorScore"]
    plays = plays.assign(
        score_diff=np.where(plays["possessionTeam"] == home, home_diff, -home_diff),
        time_remaining=(4 - plays["quarter"]).clip(lower=0) * 900
        + plays["gameClock"].map(_clock_seconds),
        yard_line=plays["absoluteYardlineNumber"],
    )
    sit = plays.set_index(["gameId", "playId"])[[
        "down", "quarter", "offenseFormation", "pff_passCoverageType",
        "possessionTeam", "defensiveTeam", "score_diff", "time_remaining", "yard_line"]]

    # --- per-rusher strain (outcome) and per-(rusher,blocker) time-avg attention ---
    rf, action = _strain_per_rusher_frame(sample_data)
    rusher_out = (rf.groupby(["gameId", "playId", "nflId_pr"])["strain"].max()
                  .rename("outcome").reset_index())
    ad = assignment_data.merge(action, on=["gameId", "playId"], how="left")
    ad = ad[ad["action_frame"].isna() | (ad["frameId"] <= ad["action_frame"])]
    att = (ad.groupby(["gameId", "playId", "nflId_pr", "nflId"])["assignment_probs"]
           .mean().rename("theta_bar").reset_index())
    qb_of_play = (sample_data.drop_duplicates(["gameId", "playId"])
                  .set_index(["gameId", "playId"])["nflId_qb"])

    # keep plays with an outcome and a COMPLETE situation row (drop any missing
    # score_diff/time_remaining/yard_line -- negligible at the play-aggregate level)
    sit_valid = sit.reset_index().dropna(
        subset=["score_diff", "time_remaining", "yard_line"])[["gameId", "playId"]]
    plays_keep = (rusher_out[["gameId", "playId"]].drop_duplicates()
                  .merge(sit_valid, on=["gameId", "playId"]))
    plays_keep = [tuple(x) for x in plays_keep.itertuples(index=False)]

    # --- global encodings ---
    enc = {
        "rusher": _encode(rusher_out["nflId_pr"]),
        "blocker": _encode(att["nflId"]),
        "qb": _encode([qb_of_play.get(p) for p in plays_keep]),
        "offense": _encode(sit["possessionTeam"]),
        "defense": _encode(sit["defensiveTeam"]),
        "down": _encode(sit["down"].fillna(-1)),
        "quarter": _encode(sit["quarter"].fillna(-1)),
        "oform": _encode(sit["offenseFormation"].fillna("UNK")),
        "dform": _encode(sit["pff_passCoverageType"].fillna("UNK")),
    }

    N = len(plays_keep)
    N_rush = max(rusher_out.groupby(["gameId", "playId"]).size())
    N_blk = max(att.groupby(["gameId", "playId"])["nflId"].nunique())

    outcome = np.zeros((N, N_rush))
    mask = np.zeros((N, N_rush))
    rusher_ids = np.zeros((N, N_rush), int)
    blocker_ids = np.zeros((N, N_blk), int)
    assignment = np.zeros((N, N_rush, N_blk))
    qb_ids = np.zeros(N, int)
    offense_ids = np.zeros(N, int); defense_ids = np.zeros(N, int)
    down_ids = np.zeros(N, int); quarter_ids = np.zeros(N, int)
    oform_ids = np.zeros(N, int); dform_ids = np.zeros(N, int)
    score_diff = np.zeros(N); time_remaining = np.zeros(N); yard_line = np.zeros(N)

    ro = rusher_out.set_index(["gameId", "playId"])
    at = att.set_index(["gameId", "playId"])
    for n, p in enumerate(plays_keep):
        srow = sit.loc[p]
        rushers = ro.loc[[p]].sort_values("nflId_pr") if p in ro.index else ro.iloc[0:0]
        r_list = rushers["nflId_pr"].tolist()
        r_slot = {rid: s for s, rid in enumerate(r_list)}
        blk_rows = at.loc[[p]] if p in at.index else at.iloc[0:0]
        b_list = sorted(blk_rows["nflId"].unique())
        b_slot = {bid: s for s, bid in enumerate(b_list)}
        for s, rid in enumerate(r_list):
            outcome[n, s] = rushers["outcome"].iloc[s]
            mask[n, s] = 1.0
            rusher_ids[n, s] = enc["rusher"][rid]
        for s, bid in enumerate(b_list):
            blocker_ids[n, s] = enc["blocker"][bid]
        for _, row in blk_rows.iterrows():
            if row["nflId_pr"] in r_slot:
                assignment[n, r_slot[row["nflId_pr"]], b_slot[row["nflId"]]] = row["theta_bar"]
        qb_ids[n] = enc["qb"].get(qb_of_play.get(p), 0)
        offense_ids[n] = enc["offense"][srow["possessionTeam"]]
        defense_ids[n] = enc["defense"][srow["defensiveTeam"]]
        down_ids[n] = enc["down"][srow["down"] if not pd.isna(srow["down"]) else -1]
        quarter_ids[n] = enc["quarter"][srow["quarter"] if not pd.isna(srow["quarter"]) else -1]
        oform_ids[n] = enc["oform"][srow["offenseFormation"] if not pd.isna(srow["offenseFormation"]) else "UNK"]
        dform_ids[n] = enc["dform"][srow["pff_passCoverageType"] if not pd.isna(srow["pff_passCoverageType"]) else "UNK"]
        score_diff[n], time_remaining[n], yard_line[n] = (
            srow["score_diff"], srow["time_remaining"], srow["yard_line"])

    # Phase 2.6: per-(play,rusher) attention normalization -> AVERAGE engaged blocker
    # effect (weights sum to 1). Unblocked rushers (n_j=0) keep a zero countering term.
    if normalize_attention:
        s = assignment.sum(axis=2, keepdims=True)
        assignment = np.where(s > 0, assignment / s, 0.0)

    # --- covariate matrices (attribute-centered priors) ---
    pl = players.copy()
    pl["height_in"] = pl["height"].map(_height_to_inches)
    pl = pl.dropna(subset=["height_in", "weight"]).set_index("nflId")

    def _attrs(id_map, positions=True):
        ids = sorted(id_map, key=id_map.get)
        rows = []
        pos_levels = None
        if positions:
            pos = [_posgroup(pl["officialPosition"].get(i, "UNK")) for i in ids]
            pos_levels = sorted(set(pos))[1:]  # drop first as reference
        h = np.array([pl["height_in"].get(i, np.nan) for i in ids], float)
        w = np.array([pl["weight"].get(i, np.nan) for i in ids], float)
        h = np.nan_to_num((h - np.nanmean(h)) / (np.nanstd(h) + 1e-9))
        w = np.nan_to_num((w - np.nanmean(w)) / (np.nanstd(w) + 1e-9))
        cols = []
        if positions:
            for lev in pos_levels:
                cols.append(np.array([1.0 if _posgroup(pl["officialPosition"].get(i, "UNK")) == lev else 0.0
                                      for i in ids]))
        cols += [h, w]
        return np.column_stack(cols)

    rusher_cov = _attrs(enc["rusher"])
    blocker_cov = _attrs(enc["blocker"])
    qb_cov = _attrs(enc["qb"], positions=False)

    def _std(x):
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)

    data = {
        "outcome": outcome, "mask": mask, "assignment": assignment,
        "rusher_ids": rusher_ids, "blocker_ids": blocker_ids,
        "quarterback_ids": qb_ids, "offense_ids": offense_ids, "defense_ids": defense_ids,
        "down_ids": down_ids, "quarter_ids": quarter_ids,
        "offensive_formation_ids": oform_ids, "defensive_formation_ids": dform_ids,
        "rusher_covariates": rusher_cov, "blocker_covariates": blocker_cov,
        "quarterback_covariates": qb_cov,
        "score_diff_covariates": _std(score_diff),
        "time_remaining_covariates": _std(time_remaining),
        "yard_line_covariates": _std(yard_line),
        "N_rushers": len(enc["rusher"]), "N_blockers": len(enc["blocker"]),
        "N_quarterbacks": len(enc["qb"]), "N_offense": len(enc["offense"]),
        "N_defense": len(enc["defense"]), "N_down": len(enc["down"]),
        "N_quarter": len(enc["quarter"]),
        "N_offensive_formation": len(enc["oform"]),
        "N_defensive_formation": len(enc["dform"]),
    }
    return data, enc


def _situation_table(plays, games):
    """per (gameId, playId) game-situation columns incl. score_diff/time_remaining/yard_line"""
    home = plays["gameId"].map(games.set_index("gameId")["homeTeamAbbr"])
    home_diff = plays["preSnapHomeScore"] - plays["preSnapVisitorScore"]
    p = plays.assign(
        score_diff=np.where(plays["possessionTeam"] == home, home_diff, -home_diff),
        time_remaining=(4 - plays["quarter"]).clip(lower=0) * 900
        + plays["gameClock"].map(_clock_seconds),
        yard_line=plays["absoluteYardlineNumber"],
    )
    return p.set_index(["gameId", "playId"])[[
        "down", "quarter", "offenseFormation", "pff_passCoverageType",
        "possessionTeam", "defensiveTeam", "score_diff", "time_remaining", "yard_line"]]


def _posgroup(p):
    """Merge 4-3 DE and 3-4 OLB into one EDGE position group (same edge-rush role) for the hierarchical
    attribute-centering; leaves interior DL (DT/NT) and all blocker positions untouched."""
    return {"DE": "EDGE", "OLB": "EDGE"}.get(p, p)


def _attr_matrix(id_map, players_indexed, positions=True):
    """standardized [position dummies (drop ref), height_z, weight_z] per encoded entity"""
    ids = sorted(id_map, key=id_map.get)
    pos_get = players_indexed["officialPosition"].get
    h = np.array([players_indexed["height_in"].get(i, np.nan) for i in ids], float)
    w = np.array([players_indexed["weight"].get(i, np.nan) for i in ids], float)
    h = np.nan_to_num((h - np.nanmean(h)) / (np.nanstd(h) + 1e-9))
    w = np.nan_to_num((w - np.nanmean(w)) / (np.nanstd(w) + 1e-9))
    cols = []
    if positions:
        levels = sorted({_posgroup(pos_get(i, "UNK")) for i in ids})[1:]  # drop reference level
        for lev in levels:
            cols.append(np.array([1.0 if _posgroup(pos_get(i, "UNK")) == lev else 0.0 for i in ids]))
    cols += [h, w]
    return np.column_stack(cols)


def build_continuous_design(sample_data, assignment_data, plays, games, players):
    """long (play, rusher, frame) data dict: forecast next-frame strain from current (AR(1))"""
    sit = _situation_table(plays, games)
    rf, _ = _strain_per_rusher_frame(sample_data)
    n0 = len(rf)
    # response = the rusher's OWN strain at the NEXT frame t+1 (a coherent per-rusher AR(1)
    # target); predictor strain_current = the same rusher's OWN strain at t. The play's peak
    # pressure (max over rushers) is a REPORTED aggregate, NOT the regression target: targeting
    # the shared play-frame max from every rusher row collapses to a frame-level average and
    # makes the own-strain AR term incoherent for all but the arg-max rusher.
    nextown = (rf[["gameId", "playId", "nflId_pr", "frameId", "strain"]]
               .rename(columns={"strain": "outcome"}).copy())
    nextown["frameId"] = nextown["frameId"] - 1   # this t+1 own-strain joins onto frame t
    sit_valid = sit.reset_index().dropna(
        subset=["score_diff", "time_remaining", "yard_line"])[["gameId", "playId"]]
    rf = (rf.rename(columns={"strain": "strain_current"})
          .merge(nextown, on=["gameId", "playId", "nflId_pr", "frameId"], how="left")
          .dropna(subset=["outcome"])
          .merge(sit_valid, on=["gameId", "playId"])
          .reset_index(drop=True))
    rf["row_id"] = np.arange(len(rf))
    N_obs = len(rf)
    print(f"[continuous] post-snap+strain rows {n0} -> after t+1 shift {N_obs}", flush=True)

    plays_list = list(dict.fromkeys(zip(rf["gameId"], rf["playId"])))
    play_idx = {p: i for i, p in enumerate(plays_list)}
    rf_play = rf[["gameId", "playId"]].apply(lambda r: play_idx[(r.iloc[0], r.iloc[1])], axis=1).to_numpy()

    # frame-level attention restricted to the kept rows
    ad = assignment_data.merge(
        rf[["gameId", "playId", "nflId_pr", "frameId", "row_id"]],
        on=["gameId", "playId", "nflId_pr", "frameId"])
    blk_enc = _encode(ad["nflId"])
    play_blk = ad.groupby(["gameId", "playId"])["nflId"].apply(lambda s: sorted(s.unique()))
    N_blk = int(play_blk.map(len).max())
    play_blk_ids = np.zeros((len(plays_list), N_blk), int)
    slot_map = {}
    for p, bl in play_blk.items():
        for s, b in enumerate(bl):
            slot_map[(p[0], p[1], b)] = s
            play_blk_ids[play_idx[p], s] = blk_enc[b]
    adm_slot = np.array([slot_map[(g, pp, b)] for g, pp, b
                         in zip(ad["gameId"], ad["playId"], ad["nflId"])])
    assignment = np.zeros((N_obs, N_blk))
    assignment[ad["row_id"].to_numpy(), adm_slot] = ad["assignment_probs"].to_numpy()
    blocker_ids = play_blk_ids[rf_play]  # (N_obs, N_blk)

    # NEXT-frame attention theta(.,j,t+1) on the SAME row (frame t): merge assignment on frameId+1
    rf_next = rf[["gameId", "playId", "nflId_pr", "frameId", "row_id"]].copy()
    rf_next["frameId"] = rf_next["frameId"] + 1                       # look up t+1 attention
    ad_n = assignment_data.merge(rf_next, on=["gameId", "playId", "nflId_pr", "frameId"])
    ad_n = ad_n[[(g, pp, b) in slot_map for g, pp, b in zip(ad_n["gameId"], ad_n["playId"], ad_n["nflId"])]]
    adm_slot_n = np.array([slot_map[(g, pp, b)] for g, pp, b
                           in zip(ad_n["gameId"], ad_n["playId"], ad_n["nflId"])])
    assignment_next = np.zeros((N_obs, N_blk))
    if len(ad_n):
        assignment_next[ad_n["row_id"].to_numpy(), adm_slot_n] = ad_n["assignment_probs"].to_numpy()

    qb_of_play = (sample_data.drop_duplicates(["gameId", "playId"])
                  .set_index(["gameId", "playId"])["nflId_qb"])
    enc = {
        "rusher": _encode(rf["nflId_pr"]), "blocker": blk_enc,
        "qb": _encode([qb_of_play.get(p) for p in plays_list]),
        "offense": _encode(sit["possessionTeam"]), "defense": _encode(sit["defensiveTeam"]),
        "down": _encode(sit["down"].fillna(-1)), "quarter": _encode(sit["quarter"].fillna(-1)),
        "oform": _encode(sit["offenseFormation"].fillna("UNK")),
        "dform": _encode(sit["pff_passCoverageType"].fillna("UNK")),
    }

    def _pa(fn):  # per-play array -> broadcast to rows
        return np.array([fn(p) for p in plays_list])[rf_play]

    g = lambda p, c: sit.loc[p, c]
    data = {
        "outcome": rf["outcome"].to_numpy(),
        "strain_current": rf["strain_current"].to_numpy(),
        "assignment": assignment, "assignment_next": assignment_next, "blocker_ids": blocker_ids,
        "rusher_ids": rf["nflId_pr"].map(enc["rusher"]).to_numpy(),
        "quarterback_ids": _pa(lambda p: enc["qb"].get(qb_of_play.get(p), 0)),
        "offense_ids": _pa(lambda p: enc["offense"][g(p, "possessionTeam")]),
        "defense_ids": _pa(lambda p: enc["defense"][g(p, "defensiveTeam")]),
        "down_ids": _pa(lambda p: enc["down"][g(p, "down") if not pd.isna(g(p, "down")) else -1]),
        "quarter_ids": _pa(lambda p: enc["quarter"][g(p, "quarter") if not pd.isna(g(p, "quarter")) else -1]),
        "offensive_formation_ids": _pa(lambda p: enc["oform"][g(p, "offenseFormation") if not pd.isna(g(p, "offenseFormation")) else "UNK"]),
        "defensive_formation_ids": _pa(lambda p: enc["dform"][g(p, "pff_passCoverageType") if not pd.isna(g(p, "pff_passCoverageType")) else "UNK"]),
        "score_diff_covariates": _pa(lambda p: g(p, "score_diff")),
        "time_remaining_covariates": _pa(lambda p: g(p, "time_remaining")),
        "yard_line_covariates": _pa(lambda p: g(p, "yard_line")),
    }
    # standardize the numeric situation covariates (over rows)
    for k in ("score_diff_covariates", "time_remaining_covariates", "yard_line_covariates"):
        x = data[k].astype(float)
        data[k] = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)

    pl = players.copy()
    pl["height_in"] = pl["height"].map(_height_to_inches)
    pl = pl.dropna(subset=["height_in", "weight"]).set_index("nflId")
    data["rusher_covariates"] = _attr_matrix(enc["rusher"], pl)
    data["blocker_covariates"] = _attr_matrix(enc["blocker"], pl)
    data["quarterback_covariates"] = _attr_matrix(enc["qb"], pl, positions=False)
    data.update({
        "N_rushers": len(enc["rusher"]), "N_blockers": len(enc["blocker"]),
        "N_quarterbacks": len(enc["qb"]), "N_offense": len(enc["offense"]),
        "N_defense": len(enc["defense"]), "N_down": len(enc["down"]),
        "N_quarter": len(enc["quarter"]),
        "N_offensive_formation": len(enc["oform"]),
        "N_defensive_formation": len(enc["dform"]),
    })
    return data, enc


def build_continuous_design_from_files(weeks=range(8), assignment_path="assignment_data_phase25.csv"):
    sample = pd.concat([pd.read_csv(f"data/sample_data_week_{i}.csv") for i in weeks],
                       ignore_index=True)
    return build_continuous_design(sample, pd.read_csv(assignment_path),
                                   pd.read_csv("data/plays.csv"), pd.read_csv("data/games.csv"),
                                   pd.read_csv("data/players.csv"))


def build_play_design_from_files(weeks=range(8), assignment_path="assignment_data.csv",
                                 normalize_attention=False):
    """load week sample-data + assignment_data + situation tables, build the data dict"""
    sample = pd.concat([pd.read_csv(f"data/sample_data_week_{i}.csv") for i in weeks],
                       ignore_index=True)
    assignment = pd.read_csv(assignment_path)
    plays = pd.read_csv("data/plays.csv")
    games = pd.read_csv("data/games.csv")
    players = pd.read_csv("data/players.csv")
    return build_play_design(sample, assignment, plays, games, players,
                             normalize_attention=normalize_attention)


# --------------------------------------------------------------------------- #
# QB force-field design (per-QB-frame ragged data dict for QBForceFieldModel)
# --------------------------------------------------------------------------- #
def build_qb_force_design(sample_data, assignment_data, plays, games, players,
                          dmin=0.5, dropback_only=True, openness=None):
    """One row per (gameId, playId, frameId): observed QB acceleration as the target, plus the
    per-rusher geometry (rusher->QB unit vector, distance), closing kinematics (closing speed and
    acceleration toward the QB), and engagement dose, padded to Rmax rushers with rmask.
    See QBForceFieldModel in model/models.py."""
    rf, _ = _strain_per_rusher_frame(sample_data)   # per (g,p,frame,rusher), snap->action window
    dx = (rf["x_smooth_qb"] - rf["x_smooth_pr"]).to_numpy()
    dy = (rf["y_smooth_qb"] - rf["y_smooth_pr"]).to_numpy()
    dist = np.hypot(dx, dy)
    dclip = np.clip(dist, dmin, None)
    ux, uy = dx / dclip, dy / dclip                 # rusher->QB unit vector (repulsion direction)
    rf = rf.assign(
        dist=dist, ux=ux, uy=uy,
        sclose=ux * rf["dx_smooth_pr"].to_numpy() + uy * rf["dy_smooth_pr"].to_numpy(),
        aclose=ux * rf["d2x_smooth_pr"].to_numpy() + uy * rf["d2y_smooth_pr"].to_numpy(),
    )
    # engagement dose = sum_b theta(b, rusher) (drop the disengaged/null sentinel)
    ad = assignment_data[assignment_data["nflId_pr"] != -1]
    dose = (ad.groupby(["gameId", "playId", "frameId", "nflId_pr"])["assignment_probs"]
            .sum().rename("dose").reset_index())
    rf = rf.merge(dose, on=["gameId", "playId", "frameId", "nflId_pr"], how="left")
    dose_coverage = float(rf["dose"].notna().mean())
    rf["dose"] = rf["dose"].fillna(0.0)
    # dropback-only for the milestone (rollouts are scheme movement, not rusher-driven)
    if dropback_only and "playAction" in plays.columns:
        pa = plays[["gameId", "playId", "playAction"]].drop_duplicates()
        rf = rf.merge(pa, on=["gameId", "playId"], how="left")
        rf = rf[rf["playAction"].fillna(0) != 1].copy()
    # play-contiguous frame spine -> integer row index n
    frames = (rf[["gameId", "playId", "frameId"]].drop_duplicates()
              .sort_values(["gameId", "playId", "frameId"]).reset_index(drop=True))
    frames["n"] = np.arange(len(frames))
    rf = rf.merge(frames, on=["gameId", "playId", "frameId"], how="left")
    rf["is_int"] = rf["officialPosition"].isin(["DT", "NT"]).astype(float)   # interior DL
    rf["is_edge"] = rf["officialPosition"].isin(["DE", "OLB"]).astype(float)  # edge rusher
    rf = rf.sort_values(["n", "nflId_pr"])
    rf["slot"] = rf.groupby("n").cumcount()
    N, Rmax = len(frames), int(rf["slot"].max() + 1)
    uhat = np.zeros((N, Rmax, 2), np.float32)
    rvel = np.zeros((N, Rmax, 2), np.float32)        # rusher velocity vector (for field anisotropy)
    dist_a = np.zeros((N, Rmax), np.float32); sclose = np.zeros((N, Rmax), np.float32)
    aclose = np.zeros((N, Rmax), np.float32); dose_a = np.zeros((N, Rmax), np.float32)
    is_int = np.zeros((N, Rmax), np.float32); is_edge = np.zeros((N, Rmax), np.float32)
    rmask = np.zeros((N, Rmax), np.float32)
    enc_rusher = _encode(rf["nflId_pr"])
    rusher_slot_id = np.zeros((N, Rmax), int)        # encoded nflId per slot (for attribution)
    ni, si = rf["n"].to_numpy(), rf["slot"].to_numpy()
    uhat[ni, si, 0] = rf["ux"]; uhat[ni, si, 1] = rf["uy"]
    rvel[ni, si, 0] = rf["dx_smooth_pr"]; rvel[ni, si, 1] = rf["dy_smooth_pr"]
    dist_a[ni, si] = rf["dist"]; sclose[ni, si] = rf["sclose"]; aclose[ni, si] = rf["aclose"]
    dose_a[ni, si] = rf["dose"]; is_int[ni, si] = rf["is_int"]; is_edge[ni, si] = rf["is_edge"]
    rusher_slot_id[ni, si] = rf["nflId_pr"].map(enc_rusher).to_numpy()
    rmask[ni, si] = 1.0
    # QB-frame target + position (QB columns are identical across rushers within a frame)
    qb = rf.drop_duplicates("n").sort_values("n")
    a_qb = qb[["d2x_smooth_qb", "d2y_smooth_qb"]].to_numpy(np.float32)
    x_qb = qb[["x_smooth_qb", "y_smooth_qb"]].to_numpy(np.float32)
    # play index / starts / snap anchor / formation (anchor+formation for later drift modes)
    play_key = frames["gameId"].astype(str) + "_" + frames["playId"].astype(str)
    play_row = pd.factorize(play_key)[0].astype(int)
    play_start = np.concatenate([[0], np.where(np.diff(play_row) != 0)[0] + 1, [N]])
    x_anchor = x_qb[play_start[:-1]][play_row]      # QB position at each play's first frame
    enc_form = _encode(plays["offenseFormation"].fillna("UNK"))
    fmap = (plays.drop_duplicates(["gameId", "playId"])
            .assign(offenseFormation=lambda d: d["offenseFormation"].fillna("UNK"))
            .set_index(["gameId", "playId"])["offenseFormation"].to_dict())
    form_ids = np.array([enc_form.get(fmap.get((g, p), "UNK"), 0)
                         for g, p in zip(frames["gameId"], frames["playId"])], int)
    # rusher attribute matrix [position dummies, height_z, weight_z] for attribute-centered charge
    pl = players.copy()
    pl["height_in"] = pl["height"].map(_height_to_inches)
    pl = pl.dropna(subset=["height_in", "weight"]).set_index("nflId")
    rusher_cov = _attr_matrix(enc_rusher, pl)
    # QB identity + game situation per frame (controls for volitional/scheme QB movement)
    sit = _situation_table(plays, games)
    qb_of_play = (sample_data.drop_duplicates(["gameId", "playId"])
                  .set_index(["gameId", "playId"])["nflId_qb"])
    sf = frames[["gameId", "playId"]].merge(sit.reset_index(), on=["gameId", "playId"], how="left")
    sf["qb"] = [qb_of_play.get((g, p), -1) for g, p in zip(sf["gameId"], sf["playId"])]
    enc_qb = _encode(sf["qb"]); enc_down = _encode(sf["down"].fillna(-1))
    enc_qtr = _encode(sf["quarter"].fillna(-1)); enc_cov = _encode(sf["pff_passCoverageType"].fillna("UNK"))
    qb_ids = sf["qb"].map(enc_qb).to_numpy()
    down_ids = sf["down"].fillna(-1).map(enc_down).to_numpy()
    quarter_ids = sf["quarter"].fillna(-1).map(enc_qtr).to_numpy()
    cov_ids = sf["pff_passCoverageType"].fillna("UNK").map(enc_cov).to_numpy()
    std = lambda c: np.nan_to_num((sf[c].to_numpy(float) - np.nanmean(sf[c])) / (np.nanstd(sf[c]) + 1e-9))
    # play-concept controls: dropback type, granular coverage, box count, play-action, QB drop depth
    extra = (plays.drop_duplicates(["gameId", "playId"])
             [["gameId", "playId", "dropBackType", "pff_passCoverage", "defendersInBox", "pff_playAction"]])
    sf = sf.merge(extra, on=["gameId", "playId"], how="left")
    enc_drop = _encode(sf["dropBackType"].fillna("UNK")); enc_cov2 = _encode(sf["pff_passCoverage"].fillna("UNK"))
    _z = lambda x: np.nan_to_num((x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)).astype(np.float32)
    dropdepth = np.hypot(x_qb[:, 0] - x_anchor[:, 0], x_qb[:, 1] - x_anchor[:, 1])   # QB retreat from snap spot
    open_cov = None
    if openness is not None:                         # per-frame receiver openness (space-control)
        of = frames[["gameId", "playId", "frameId"]].merge(
            openness[["gameId", "playId", "frameId", "open_sum"]], on=["gameId", "playId", "frameId"], how="left")
        cov = float((of["open_sum"].notna()).mean())
        print(f"  openness join coverage of QB-frames: {cov:.2f}", flush=True)
        open_cov = _z(of["open_sum"].fillna(of["open_sum"].median()).to_numpy(float))
    data = {
        "a_qb": a_qb, "uhat": uhat, "rvel": rvel, "dist": dist_a, "sclose": sclose, "aclose": aclose,
        "dose": dose_a, "is_interior": is_int, "is_edge": is_edge, "rmask": rmask,
        "rusher_slot_id": rusher_slot_id, "rusher_covariates": rusher_cov, "N_rushers": len(enc_rusher),
        "x_qb": x_qb, "x_anchor": x_anchor,
        "quarterback_ids": qb_ids, "down_ids": down_ids, "quarter_ids": quarter_ids,
        "coverage_ids": cov_ids, "score_diff_cov": std("score_diff"),
        "time_cov": std("time_remaining"), "yard_cov": std("yard_line"),
        "N_qb": len(enc_qb), "N_down": len(enc_down), "N_quarter": len(enc_qtr), "N_cov": len(enc_cov),
        "dropback_ids": sf["dropBackType"].fillna("UNK").map(enc_drop).to_numpy(),
        "cov2_ids": sf["pff_passCoverage"].fillna("UNK").map(enc_cov2).to_numpy(),
        "box_cov": _z(sf["defendersInBox"].to_numpy(float)),
        "pa_cov": sf["pff_playAction"].fillna(0).to_numpy(np.float32),
        "depth_cov": _z(dropdepth), "N_dropback": len(enc_drop), "N_cov2": len(enc_cov2),
        "form_ids": form_ids, "play_row": play_row, "frame_id": frames["frameId"].to_numpy(int),
        "gameId": frames["gameId"].to_numpy(), "playId": frames["playId"].to_numpy(),
        "play_start": play_start, "N_form": len(enc_form), "Rmax": Rmax,
    }
    if open_cov is not None:
        data["open_cov"] = open_cov
    enc = {"form": enc_form, "rusher": enc_rusher, "qb": enc_qb, "dose_coverage": dose_coverage}
    return data, enc


def build_pocket_design(sample_data, plays, games, players, dropback_only=False, assignment=None):
    """Per-(gameId,playId,frameId) pocket design for Fernandez-Bornn space control: QB + every
    pass-RUSHER (defense) + every pass-BLOCKER (offense), each with position/velocity/acceleration,
    distance-to-QB, mask, and encoded id, padded to per-team max slots. Snap->action windowed.
    Feeds src/pitch_control.py. (Receivers/coverage DBs are absent from sample_data; this is pocket
    control among pocket participants.)"""
    rf, action = _strain_per_rusher_frame(sample_data)        # rusher rows, windowed; has QB+rusher kinematics
    bcols = ["gameId", "playId", "frameId", "nflId", "x_smooth", "y_smooth",
             "dx_smooth", "dy_smooth", "d2x_smooth", "d2y_smooth", "officialPosition_pb"]
    bf = sample_data.drop_duplicates(["gameId", "playId", "nflId", "frameId"])[bcols].copy()
    bf = bf.merge(action, on=["gameId", "playId"], how="left")
    bf = bf[bf["action_frame"].isna() | (bf["frameId"] <= bf["action_frame"])]

    frames = (rf[["gameId", "playId", "frameId"]].drop_duplicates()
              .sort_values(["gameId", "playId", "frameId"]).reset_index(drop=True))
    frames["n"] = np.arange(len(frames)); N = len(frames)
    rf = rf.merge(frames, on=["gameId", "playId", "frameId"])
    bf = bf.merge(frames, on=["gameId", "playId", "frameId"])
    qb = rf.drop_duplicates("n").sort_values("n")
    x_qb = qb[["x_smooth_qb", "y_smooth_qb"]].to_numpy(np.float32)
    v_qb = qb[["dx_smooth_qb", "dy_smooth_qb"]].to_numpy(np.float32)
    a_qb = qb[["d2x_smooth_qb", "d2y_smooth_qb"]].to_numpy(np.float32)

    enc_rush, enc_blk = _encode(rf["nflId_pr"]), _encode(bf["nflId"])
    rf = rf.sort_values(["n", "nflId_pr"]); rf["slot"] = rf.groupby("n").cumcount()
    bf = bf.sort_values(["n", "nflId"]); bf["slot"] = bf.groupby("n").cumcount()
    R, B = int(rf["slot"].max() + 1), int(bf["slot"].max() + 1)

    def fill(df, cols, K):
        arr = np.zeros((N, K, len(cols)), np.float32)
        ni, si = df["n"].to_numpy(), df["slot"].to_numpy()
        for k, c in enumerate(cols):
            arr[ni, si, k] = df[c].to_numpy()
        return arr

    x_rush = fill(rf, ["x_smooth_pr", "y_smooth_pr"], R); v_rush = fill(rf, ["dx_smooth_pr", "dy_smooth_pr"], R)
    a_rush = fill(rf, ["d2x_smooth_pr", "d2y_smooth_pr"], R)
    x_blk = fill(bf, ["x_smooth", "y_smooth"], B); v_blk = fill(bf, ["dx_smooth", "dy_smooth"], B)
    a_blk = fill(bf, ["d2x_smooth", "d2y_smooth"], B)
    rmask = np.zeros((N, R), np.float32); rmask[rf["n"], rf["slot"]] = 1.0
    bmask = np.zeros((N, B), np.float32); bmask[bf["n"], bf["slot"]] = 1.0
    rush_slot_id = np.zeros((N, R), int); rush_slot_id[rf["n"], rf["slot"]] = rf["nflId_pr"].map(enc_rush)
    blk_slot_id = np.zeros((N, B), int); blk_slot_id[bf["n"], bf["slot"]] = bf["nflId"].map(enc_blk)
    d_rush = np.hypot(x_rush[..., 0] - x_qb[:, None, 0], x_rush[..., 1] - x_qb[:, None, 1]) * rmask
    d_blk = np.hypot(x_blk[..., 0] - x_qb[:, None, 0], x_blk[..., 1] - x_qb[:, None, 1]) * bmask

    theta = None
    if assignment is not None:                                 # HMM weights theta[n, b_slot, r_slot]
        a = assignment[assignment["nflId_pr"] != -1][["gameId", "playId", "frameId",
                                                       "nflId_pr", "nflId", "assignment_probs"]]
        a = a.merge(frames, on=["gameId", "playId", "frameId"]) \
             .merge(rf[["n", "nflId_pr", "slot"]].rename(columns={"slot": "r_slot"}), on=["n", "nflId_pr"]) \
             .merge(bf[["n", "nflId", "slot"]].rename(columns={"slot": "b_slot"}), on=["n", "nflId"])
        theta = np.zeros((N, B, R), np.float32)
        theta[a["n"].to_numpy(), a["b_slot"].to_numpy(), a["r_slot"].to_numpy()] = a["assignment_probs"].to_numpy()
    play_key = frames["gameId"].astype(str) + "_" + frames["playId"].astype(str)
    play_row = pd.factorize(play_key)[0].astype(int)
    play_start = np.concatenate([[0], np.where(np.diff(play_row) != 0)[0] + 1, [N]])
    data = {
        "x_qb": x_qb, "v_qb": v_qb, "a_qb": a_qb,
        "x_rush": x_rush, "v_rush": v_rush, "a_rush": a_rush, "d_rush": d_rush,
        "rmask": rmask, "rush_slot_id": rush_slot_id,
        "x_blk": x_blk, "v_blk": v_blk, "a_blk": a_blk, "d_blk": d_blk,
        "bmask": bmask, "blk_slot_id": blk_slot_id,
        "play_row": play_row, "play_start": play_start, "frame_id": frames["frameId"].to_numpy(int),
        "gameId": frames["gameId"].to_numpy(), "playId": frames["playId"].to_numpy(),
        "N_rushers": len(enc_rush), "N_blockers": len(enc_blk), "Rmax": R, "Bmax": B,
    }
    if theta is not None:
        data["theta"] = theta
    return data, {"rusher": enc_rush, "blocker": enc_blk}


def build_pocket_design_from_files(weeks=range(1), assignment_path=None):
    sample = pd.concat([pd.read_csv(f"data/sample_data_week_{i}.csv") for i in weeks], ignore_index=True)
    assignment = pd.read_csv(assignment_path) if assignment_path else None
    return build_pocket_design(sample, pd.read_csv("data/plays.csv"), pd.read_csv("data/games.csv"),
                               pd.read_csv("data/players.csv"), assignment=assignment)


FIELD_LEN, FIELD_WID = 120.0, 53.3
_END_EVENTS = ["pass_forward", "qb_sack", "run", "handoff", "fumble", "qb_strip_sack"]


def build_alltwentytwo_design(tracking, pff, plays):
    """Per-(gameId,playId,frameId) ALL-22 design for full-field/downfield space control & openness.
    Offense = receivers (PFF 'Pass Route'); defense = coverage (PFF 'Coverage'); plus the QB and the
    ball. Kinematics derived (no Kalman) from NGS speed/dir/accel; coords NORMALIZED so the offense
    always attacks +x; windowed snap->release. Feeds src/pitch_control.py (field_grid + radius
    override). Returns ragged padded arrays + per-frame LOS, ball position, and event."""
    roles = pff[["gameId", "playId", "nflId", "pff_role"]]
    ball = tracking[tracking["team"] == "football"][["gameId", "playId", "frameId", "x", "y", "playDirection"]].copy()
    trk = tracking[tracking["team"] != "football"].copy()
    trk = trk.dropna(subset=["nflId"]); trk["nflId"] = trk["nflId"].astype(int)
    trk = trk.merge(roles, on=["gameId", "playId", "nflId"], how="left")
    trk = trk.merge(plays[["gameId", "playId", "absoluteYardlineNumber"]], on=["gameId", "playId"], how="left")

    left = (trk["playDirection"] == "left").to_numpy()                  # normalize: offense attacks +x
    trk["xn"] = np.where(left, FIELD_LEN - trk["x"], trk["x"])
    trk["yn"] = np.where(left, FIELD_WID - trk["y"], trk["y"])
    dirn = np.where(left, (trk["dir"] + 180.0) % 360.0, trk["dir"]); rad = np.deg2rad(dirn)
    s, acc = trk["s"].to_numpy(), trk["a"].to_numpy()
    trk["dx"] = np.cos(rad) * s; trk["dy"] = np.sin(rad) * s
    trk["d2x"] = np.cos(rad) * acc; trk["d2y"] = np.sin(rad) * acc
    trk["los"] = np.where(left, FIELD_LEN - trk["absoluteYardlineNumber"], trk["absoluteYardlineNumber"])
    bleft = (ball["playDirection"] == "left").to_numpy()
    ball["xn"] = np.where(bleft, FIELD_LEN - ball["x"], ball["x"])
    ball["yn"] = np.where(bleft, FIELD_WID - ball["y"], ball["y"])

    ev = trk[["gameId", "playId", "frameId", "event"]].dropna(subset=["event"]).drop_duplicates()
    snap = ev[ev["event"] == "ball_snap"].groupby(["gameId", "playId"])["frameId"].min().rename("snap")
    end = (ev[ev["event"].isin(_END_EVENTS)].groupby(["gameId", "playId"])["frameId"].min().rename("end"))
    win = pd.concat([snap, end], axis=1).reset_index()
    win["end"] = win["end"].fillna(1e9)                                  # no release event -> keep to play end
    trk = trk.merge(win.dropna(subset=["snap"]), on=["gameId", "playId"])
    trk = trk[(trk["frameId"] >= trk["snap"]) & (trk["frameId"] <= trk["end"])]

    qb = trk[trk["pff_role"] == "Pass"].drop_duplicates(["gameId", "playId", "frameId"])
    frames = (qb[["gameId", "playId", "frameId", "los"]].sort_values(["gameId", "playId", "frameId"])
              .reset_index(drop=True))
    frames["n"] = np.arange(len(frames)); N = len(frames)
    key = ["gameId", "playId", "frameId"]
    qb = qb.merge(frames[key + ["n"]], on=key)
    x_qb = qb.sort_values("n")[["xn", "yn"]].to_numpy(np.float32)
    v_qb = qb.sort_values("n")[["dx", "dy"]].to_numpy(np.float32)
    a_qb = qb.sort_values("n")[["d2x", "d2y"]].to_numpy(np.float32)

    def team(role):
        d = trk[trk["pff_role"] == role].merge(frames[key + ["n"]], on=key)
        d = d.sort_values(["n", "nflId"]); d["slot"] = d.groupby("n").cumcount()
        return d
    rec, cov = team("Pass Route"), team("Coverage")
    K, D = int(rec["slot"].max() + 1), int(cov["slot"].max() + 1)
    enc_rec, enc_cov = _encode(rec["nflId"]), _encode(cov["nflId"])

    def fill(df, cols, Kk):
        arr = np.zeros((N, Kk, len(cols)), np.float32)
        ni, si = df["n"].to_numpy(), df["slot"].to_numpy()
        for j, c in enumerate(cols):
            arr[ni, si, j] = df[c].to_numpy()
        return arr
    pos, vel, acc_c = ["xn", "yn"], ["dx", "dy"], ["d2x", "d2y"]
    x_rec, v_rec, a_rec = fill(rec, pos, K), fill(rec, vel, K), fill(rec, acc_c, K)
    x_cov, v_cov, a_cov = fill(cov, pos, D), fill(cov, vel, D), fill(cov, acc_c, D)
    rec_mask = np.zeros((N, K), np.float32); rec_mask[rec["n"], rec["slot"]] = 1.0
    cov_mask = np.zeros((N, D), np.float32); cov_mask[cov["n"], cov["slot"]] = 1.0
    rec_slot_id = np.zeros((N, K), int); rec_slot_id[rec["n"], rec["slot"]] = rec["nflId"].map(enc_rec)
    cov_slot_id = np.zeros((N, D), int); cov_slot_id[cov["n"], cov["slot"]] = cov["nflId"].map(enc_cov)

    bj = frames.merge(ball[key + ["xn", "yn"]], on=key, how="left")
    x_ball = bj[["xn", "yn"]].to_numpy(np.float32)

    play_key = frames["gameId"].astype(str) + "_" + frames["playId"].astype(str)
    play_row = pd.factorize(play_key)[0].astype(int)
    play_start = np.concatenate([[0], np.where(np.diff(play_row) != 0)[0] + 1, [N]])
    data = {
        "x_qb": x_qb, "v_qb": v_qb, "a_qb": a_qb, "x_ball": x_ball,
        "x_rec": x_rec, "v_rec": v_rec, "a_rec": a_rec, "rec_mask": rec_mask, "rec_slot_id": rec_slot_id,
        "x_cov": x_cov, "v_cov": v_cov, "a_cov": a_cov, "cov_mask": cov_mask, "cov_slot_id": cov_slot_id,
        "los_x": frames["los"].to_numpy(np.float32), "play_row": play_row, "play_start": play_start,
        "frame_id": frames["frameId"].to_numpy(int), "gameId": frames["gameId"].to_numpy(),
        "playId": frames["playId"].to_numpy(), "K": K, "D": D,
        "N_rec": len(enc_rec), "N_cov": len(enc_cov),
    }
    return data, {"receiver": enc_rec, "coverage": enc_cov}


def build_alltwentytwo_design_from_files(weeks=range(1, 2)):
    tracking = pd.concat([pd.read_csv(f"data/week{i}.csv") for i in weeks], ignore_index=True)
    return build_alltwentytwo_design(tracking, pd.read_csv("data/pffScoutingData.csv"),
                                     pd.read_csv("data/plays.csv"))


def build_qb_force_design_from_files(weeks=range(1), assignment_path="assignment_data_phase25.csv",
                                     openness_path=None):
    sample = pd.concat([pd.read_csv(f"data/sample_data_week_{i}.csv") for i in weeks],
                       ignore_index=True)
    openness = pd.read_csv(openness_path) if openness_path else None
    return build_qb_force_design(sample, pd.read_csv(assignment_path), pd.read_csv("data/plays.csv"),
                                 pd.read_csv("data/games.csv"), pd.read_csv("data/players.csv"),
                                 openness=openness)


if __name__ == "__main__":
    import pickle

    data, enc = build_play_design_from_files()
    pickle.dump(data, open("play_design_data.pkl", "wb"))
    pickle.dump(enc, open("play_encodings.pkl", "wb"))
    print("wrote play_design_data.pkl", {k: getattr(v, "shape", v) for k, v in data.items()
                                         if not str(k).startswith("N_")})
