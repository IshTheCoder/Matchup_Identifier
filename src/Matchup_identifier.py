#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

df_pff = pd.read_csv("data/pffScoutingData.csv")
players = pd.read_csv("data/players.csv")
weeks_used = 8
all_files = ["data/week" + str(i) + ".csv" for i in range(1, weeks_used + 1)]

df_ngs = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)


for i, f in enumerate(all_files):
    df_ngs = pd.read_csv(f)
    merged_df = df_ngs.merge(
        df_pff[["gameId", "playId", "nflId", "pff_role", "pff_positionLinedUp"]],
        left_on=["gameId", "playId", "nflId"],
        right_on=["gameId", "playId", "nflId"],
    )

    merged_df = merged_df[
        (merged_df["pff_role"] == "Pass Block")
        | (merged_df["pff_role"] == "Pass Rush")
        | (merged_df["pff_role"] == "Pass")
    ]
    merged_df = pd.merge(players[["nflId", "officialPosition"]], merged_df)

    df_qb = merged_df[merged_df["pff_role"] == "Pass"][
        ["gameId", "playId", "frameId", "time", "nflId", "x", "y", "event"]
    ]
    df_qb_snap = df_qb[df_qb.event == "ball_snap"][["gameId", "playId", "time"]]
    df_qb_snap.rename(mapper={"time": "snap_time"}, axis=1, inplace=True)
    df_qb = pd.merge(df_qb_snap, df_qb)
    df_qb = df_qb[df_qb.snap_time <= df_qb.time]

    df_pr = merged_df[merged_df["pff_role"] == "Pass Rush"][
        ["gameId", "playId", "frameId", "time", "nflId", "x", "y", "officialPosition"]
    ]
    df_pb = merged_df[merged_df["pff_role"] == "Pass Block"][
        ["gameId", "playId", "frameId", "time", "nflId", "x", "y", "officialPosition"]
    ]

    df_x = df_pr.merge(
        df_qb,
        left_on=["gameId", "playId", "frameId", "time"],
        right_on=["gameId", "playId", "frameId", "time"],
        suffixes=("_pr", "_qb"),
    )

    df_final = df_x.merge(
        df_pb,
        left_on=["gameId", "playId", "frameId", "time"],
        right_on=["gameId", "playId", "frameId", "time"],
        suffixes=("", "_pb"),
    )

    df_final["possession_id"] = df_final.groupby(["gameId", "playId"]).ngroup()

    df_final.sort_values(["nflId", "nflId_pr", "time"]).to_csv(
        f"data/sample_data_week_{i}.csv", index=False
    )
