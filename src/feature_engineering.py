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
            pos = [pl["officialPosition"].get(i, "UNK") for i in ids]
            pos_levels = sorted(set(pos))[1:]  # drop first as reference
        h = np.array([pl["height_in"].get(i, np.nan) for i in ids], float)
        w = np.array([pl["weight"].get(i, np.nan) for i in ids], float)
        h = np.nan_to_num((h - np.nanmean(h)) / (np.nanstd(h) + 1e-9))
        w = np.nan_to_num((w - np.nanmean(w)) / (np.nanstd(w) + 1e-9))
        cols = []
        if positions:
            for lev in pos_levels:
                cols.append(np.array([1.0 if pl["officialPosition"].get(i, "UNK") == lev else 0.0
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
        levels = sorted({pos_get(i, "UNK") for i in ids})[1:]  # drop reference level
        for lev in levels:
            cols.append(np.array([1.0 if pos_get(i, "UNK") == lev else 0.0 for i in ids]))
    cols += [h, w]
    return np.column_stack(cols)


def build_continuous_design(sample_data, assignment_data, plays, games, players):
    """long (play, rusher, frame) data dict: forecast next-frame strain from current (AR(1))"""
    sit = _situation_table(plays, games)
    rf, _ = _strain_per_rusher_frame(sample_data)
    n0 = len(rf)
    # response = MAX strain across ALL rushers at the NEXT frame (the play's peak
    # pressure at t+1); predictor strain_current = the individual rusher's OWN strain at t.
    maxnext = (rf.groupby(["gameId", "playId", "frameId"])["strain"].max()
               .rename("outcome").reset_index())
    maxnext["frameId"] = maxnext["frameId"] - 1   # shift so it joins onto frame t (its t+1 max)
    sit_valid = sit.reset_index().dropna(
        subset=["score_diff", "time_remaining", "yard_line"])[["gameId", "playId"]]
    rf = (rf.rename(columns={"strain": "strain_current"})
          .merge(maxnext, on=["gameId", "playId", "frameId"], how="left")
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
        "assignment": assignment, "blocker_ids": blocker_ids,
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


def build_continuous_design_from_files(weeks=range(8)):
    sample = pd.concat([pd.read_csv(f"data/sample_data_week_{i}.csv") for i in weeks],
                       ignore_index=True)
    return build_continuous_design(sample, pd.read_csv("assignment_data.csv"),
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


if __name__ == "__main__":
    import pickle

    data, enc = build_play_design_from_files()
    pickle.dump(data, open("play_design_data.pkl", "wb"))
    pickle.dump(enc, open("play_encodings.pkl", "wb"))
    print("wrote play_design_data.pkl", {k: getattr(v, "shape", v) for k, v in data.items()
                                         if not str(k).startswith("N_")})
