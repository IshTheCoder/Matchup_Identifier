#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

df_pff = pd.read_csv("pffScoutingData.csv")
weeks_used = 1
all_files = ["week" + str(i) + ".csv" for i in range(1, weeks_used + 1)]

df_ngs = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

print(df_pff["pff_role"].unique())


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

merged_df

df_qb = merged_df[merged_df["pff_role"] == "Pass"][
    ["gameId", "playId", "frameId", "time", "nflId", "x", "y"]
]
df_qb

df_pr = merged_df[merged_df["pff_role"] == "Pass Rush"][
    ["gameId", "playId", "frameId", "time", "nflId", "x", "y"]
]
df_pb = merged_df[merged_df["pff_role"] == "Pass Block"][
    ["gameId", "playId", "frameId", "time", "nflId", "x", "y"]
]

df_x = df_pr.merge(
    df_qb,
    left_on=["gameId", "playId", "frameId", "time"],
    right_on=["gameId", "playId", "frameId", "time"],
    suffixes=("_pr", "_qb"),
)


df_test = df_x
# df_x[(df_x['gameId'] ==2021090900)&(df_x['playId'] ==97)&(df_x['frameId'] <=10)]


df_final = df_test.merge(
    df_pb,
    left_on=["gameId", "playId", "frameId", "time"],
    right_on=["gameId", "playId", "frameId", "time"],
    suffixes=("", "_pb"),
)

X = df_final[["x_pr", "y_pr", "x_qb", "y_qb"]].values


Y = df_final[["x", "y"]].values

print(X.shape, np.shape(Y))
