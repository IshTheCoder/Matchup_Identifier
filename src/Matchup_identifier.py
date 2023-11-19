#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pykalman as pk
from scipy.linalg import toeplitz

df_pff = pd.read_csv("data/pffScoutingData.csv")
players = pd.read_csv("data/players.csv")
weeks_used = 8
all_files = ["data/week" + str(i) + ".csv" for i in range(1, weeks_used + 1)]


delta = 0.1
column = np.array([1, 0, 0])
first_line = np.array([1, delta, delta**2 * 0.5])
transition_matrices = toeplitz(column, first_line)
observation_matrices = np.eye(3)
observation_covariance = np.eye(3)
Q = np.array([[(delta**3) / 6, 0.5 * delta**2, delta]])
transition_covariance = Q.T.dot(Q)


def smooth_data_y(data, transition_covariance, observation_covariance):
    obs = data.sort_values(by="frameId", ascending=True)[["y", "dy", "d2y"]].to_numpy()
    kf = pk.KalmanFilter(
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrices,
        observation_covariance=observation_covariance,
        observation_matrices=observation_matrices,
        initial_state_covariance=np.eye(3) * 0.001,
        initial_state_mean=obs[0],
    )
    return kf.smooth(obs)[0]


def smooth_data_x(data, transition_covariance, observation_covariance):
    obs = data.sort_values(by="frameId", ascending=True)[["x", "dx", "d2x"]].to_numpy()
    kf = pk.KalmanFilter(
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrices,
        observation_covariance=observation_covariance,
        observation_matrices=observation_matrices,
        initial_state_covariance=np.eye(3) * 0.001,
        initial_state_mean=obs[0],
    )
    return kf.smooth(obs)[0]


for i, f in enumerate(all_files):
    df_ngs = pd.read_csv(f)

    df_ngs["dx"] = np.cos(df_ngs["dir"]) * df_ngs["s"]
    df_ngs["dy"] = np.sin(df_ngs["dir"]) * df_ngs["s"]
    df_ngs["d2y"] = np.sin(df_ngs["dir"]) * df_ngs["a"]
    df_ngs["d2x"] = np.cos(df_ngs["dir"]) * df_ngs["a"]

    smoothed_data_x = (
        df_ngs.groupby(["nflId", "gameId", "playId"], group_keys=True)
        .apply(
            lambda x: smooth_data_x(x, transition_covariance, observation_covariance)
        )
        .reset_index()
        .explode(0)
    )
    smoothed_data_y = (
        df_ngs.groupby(["nflId", "gameId", "playId"], group_keys=True)
        .apply(
            lambda x: smooth_data_y(x, transition_covariance, observation_covariance)
        )
        .reset_index()
        .explode(0)
    )

    smoothed_data_x.rename(columns={0: "smooth"}, inplace=True)
    smoothed_data_y.rename(columns={0: "smooth"}, inplace=True)

    smoothed_x_columns = smoothed_data_x["smooth"].apply(pd.Series)
    smoothed_y_columns = smoothed_data_y["smooth"].apply(pd.Series)

    smoothed_x_columns.rename(
        columns={0: "x_smooth", 1: "dx_smooth", 2: "d2x_smooth"}, inplace=True
    )
    smoothed_y_columns.rename(
        columns={0: "y_smooth", 1: "dy_smooth", 2: "d2y_smooth"}, inplace=True
    )

    smoothed_data = pd.concat([smoothed_x_columns, smoothed_y_columns], axis=1)
    smoothed_data = pd.concat(
        [smoothed_data, smoothed_data_x[["playId", "nflId", "gameId"]]], axis=1
    )
    smoothed_data["frameId"] = (
        smoothed_data.groupby(["nflId", "playId", "gameId"]).cumcount() + 1
    )

    smoothed_data = smoothed_data.merge(df_ngs)

    merged_df = smoothed_data.merge(
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
        [
            "gameId",
            "playId",
            "frameId",
            "time",
            "nflId",
            "x_smooth",
            "dx_smooth",
            "d2x_smooth",
            "y_smooth",
            "dy_smooth",
            "d2y_smooth",
            "event",
        ]
    ]
    df_qb_snap = df_qb[df_qb.event == "ball_snap"][["gameId", "playId", "time"]]
    df_qb_snap.rename(mapper={"time": "snap_time"}, axis=1, inplace=True)
    df_qb = pd.merge(df_qb_snap, df_qb)
    df_qb = df_qb[df_qb.snap_time <= df_qb.time]

    df_pr = merged_df[merged_df["pff_role"] == "Pass Rush"][
        [
            "gameId",
            "playId",
            "frameId",
            "time",
            "nflId",
            "x_smooth",
            "dx_smooth",
            "d2x_smooth",
            "y_smooth",
            "dy_smooth",
            "d2y_smooth",
            "officialPosition",
        ]
    ]
    df_pb = merged_df[merged_df["pff_role"] == "Pass Block"][
        [
            "gameId",
            "playId",
            "frameId",
            "time",
            "nflId",
            "x_smooth",
            "dx_smooth",
            "d2x_smooth",
            "y_smooth",
            "dy_smooth",
            "d2y_smooth",
            "officialPosition",
        ]
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
