#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pykalman as pk
from scipy.linalg import toeplitz
from scipy.stats import chi2

df_pff = pd.read_csv("data/pffScoutingData.csv")
players = pd.read_csv("data/players.csv")
weeks_used = 8
all_files = ["data/week" + str(i) + ".csv" for i in range(1, weeks_used + 1)]


delta = 0.1  # frame interval (NFL tracking is 10 Hz)

# constant-acceleration state transition for the state [position, velocity, accel]
column = np.array([1, 0, 0])
first_line = np.array([1, delta, delta**2 * 0.5])
transition_matrices = toeplitz(column, first_line)

# --- process noise -----------------------------------------------------------
# Full-rank discrete white-noise-acceleration (jerk) covariance. The previous
# Q.T @ Q (Q a single 1x3 row) was rank-1 / singular, which is degenerate and can
# destabilize the gain. The DWNA form below is the standard full-rank process
# covariance for a [pos, vel, acc] state; PROCESS_SPECTRAL_DENSITY (q, yd^2/s^5)
# is the one knob to tune via diagnose_smoothing.
PROCESS_SPECTRAL_DENSITY = 1.0
_d = delta
transition_covariance = PROCESS_SPECTRAL_DENSITY * np.array(
    [
        [_d**5 / 20, _d**4 / 8, _d**3 / 6],
        [_d**4 / 8, _d**3 / 3, _d**2 / 2],
        [_d**3 / 6, _d**2 / 2, _d],
    ]
)

# --- observation model -------------------------------------------------------
# "position" (default): observe only the position coordinate and let the filter
#   infer velocity/acceleration. The dx/dy/d2x/d2y channels are deterministic
#   functions of the same s/a/dir tracking signal, so feeding them as independent
#   observations (the old eye(3) model) double-counts the measurement and is also
#   unit-incoherent (variance 1 for yd, yd/s and yd/s^2 alike).
# "full": observe position, velocity AND acceleration, but with a unit-coherent
#   diagonal covariance. Use only if the derived kinematics are trusted as carrying
#   independent information.
OBSERVATION_MODE = "position"

if OBSERVATION_MODE == "position":
    observation_matrices = np.array([[1.0, 0.0, 0.0]])
    # ~(0.03 yd)^2. Tuned via diagnose_smoothing: with q=1 this gives mean NIS ~1.1
    # (well calibrated) and ~0.013 yd position RMSE over a sample of moving plays.
    # Note: the standardized-innovation autocorrelation stays high (~0.9) across the
    # well-calibrated region, indicating a single white-jerk process model does not
    # fully capture smooth 10 Hz player motion -- a model-structure limitation, not a
    # value to be tuned away. Re-run diagnose_smoothing if changing q or the data.
    observation_covariance = np.array([[1e-3]])
elif OBSERVATION_MODE == "full":
    observation_matrices = np.eye(3)
    # position ~0.1 yd, velocity ~0.5 yd/s, acceleration ~1.0 yd/s^2
    observation_covariance = np.diag([0.01, 0.25, 1.0])
else:
    raise ValueError(f"unknown OBSERVATION_MODE {OBSERVATION_MODE!r}")

# position tight, velocity/acceleration loose (one frame barely constrains them);
# the old eye(3)*0.001 was over-confident on vel/accel.
initial_state_covariance = np.diag([0.01, 1.0, 4.0])


def _build_kf(obs_full):
    """KalmanFilter configured from the module-level parameters.

    obs_full is the full [pos, vel, accel] array (sorted by frame); the velocity /
    acceleration columns are only used to seed the initial state mean, never as
    observations unless OBSERVATION_MODE == "full".
    """
    return pk.KalmanFilter(
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrices,
        observation_covariance=observation_covariance,
        observation_matrices=observation_matrices,
        initial_state_covariance=initial_state_covariance,
        initial_state_mean=obs_full[0],
    )


def _observations(obs_full):
    """select the observed channels for the current OBSERVATION_MODE"""
    if OBSERVATION_MODE == "position":
        return obs_full[:, 0:1]
    return obs_full


def _smooth_state(data, columns):
    """RTS-smoothed [pos, vel, accel] state for one (player, play) trajectory"""
    obs_full = data.sort_values(by="frameId", ascending=True)[columns].to_numpy()
    kf = _build_kf(obs_full)
    return kf.smooth(_observations(obs_full))[0]


def smooth_data_y(data, transition_covariance, observation_covariance):
    return _smooth_state(data, ["y", "dy", "d2y"])


def smooth_data_x(data, transition_covariance, observation_covariance):
    return _smooth_state(data, ["x", "dx", "d2x"])


def diagnose_smoothing(data, columns):
    """innovation-based diagnostics for the Kalman configuration on one trajectory

    Returns a dict with:
      - mean_nis / nis_expected / frac_nis_in_band : normalized-innovation-squared
        consistency. A well-tuned filter has mean NIS ~ #observed dims and ~95% of
        samples inside the chi^2 95% band.
      - innov_autocorr : lag-1..5 autocorrelation of the standardized position
        innovation; near zero => white => noise levels well chosen.
      - rmse_smoothed_vs_raw_position : over-smoothing check (small but non-zero).
      - rmse_smoothed_vs_derived_velocity : sanity vs the s/dir-derived velocity.
    """
    obs_full = data.sort_values(by="frameId", ascending=True)[columns].to_numpy()
    kf = _build_kf(obs_full)
    obs = _observations(obs_full)
    filt_means, filt_covs = kf.filter(obs)
    smooth_means = kf.smooth(obs)[0]

    F = transition_matrices
    H = observation_matrices
    R = observation_covariance
    Q = transition_covariance
    n = obs.shape[0]
    m = H.shape[0]

    nis = np.zeros(n - 1)
    innov_pos = np.zeros(n - 1)
    for t in range(1, n):
        x_pred = F @ filt_means[t - 1]
        p_pred = F @ filt_covs[t - 1] @ F.T + Q
        s = H @ p_pred @ H.T + R
        innovation = obs[t] - H @ x_pred
        nis[t - 1] = float(innovation @ np.linalg.solve(s, innovation))
        innov_pos[t - 1] = innovation[0] / np.sqrt(s[0, 0])

    def autocorr(x, lag):
        x = x - x.mean()
        denom = x @ x
        return float((x[:-lag] @ x[lag:]) / denom) if denom > 0 else 0.0

    lo, hi = float(chi2.ppf(0.025, m)), float(chi2.ppf(0.975, m))
    return {
        "observation_mode": OBSERVATION_MODE,
        "n_frames": int(n),
        "mean_nis": float(nis.mean()),
        "nis_expected": int(m),
        "nis_95_band": (lo, hi),
        "frac_nis_in_band": float(((nis >= lo) & (nis <= hi)).mean()),
        "innov_autocorr": {lag: autocorr(innov_pos, lag) for lag in range(1, 6)},
        "rmse_smoothed_vs_raw_position": float(
            np.sqrt(np.mean((smooth_means[:, 0] - obs_full[:, 0]) ** 2))
        ),
        "rmse_smoothed_vs_derived_velocity": float(
            np.sqrt(np.mean((smooth_means[:, 1] - obs_full[:, 1]) ** 2))
        ),
    }


def run_pipeline():
    """smooth every week's tracking data and write data/sample_data_week_{i}.csv"""
    for i, f in enumerate(all_files):
        df_ngs = pd.read_csv(f)

        df_ngs["dx"] = np.cos(np.deg2rad(df_ngs["dir"])) * df_ngs["s"]
        df_ngs["dy"] = np.sin(np.deg2rad(df_ngs["dir"])) * df_ngs["s"]
        df_ngs["d2y"] = np.sin(np.deg2rad(df_ngs["dir"])) * df_ngs["a"]
        df_ngs["d2x"] = np.cos(np.deg2rad(df_ngs["dir"])) * df_ngs["a"]

        print(f"smoothing week {i}")

        smoothed_data_x = (
            df_ngs.groupby(["nflId", "gameId", "playId"], group_keys=True)
            .apply(
                lambda x: smooth_data_x(
                    x, transition_covariance, observation_covariance
                )
            )
            .reset_index()
            .explode(0)
        )
        smoothed_data_y = (
            df_ngs.groupby(["nflId", "gameId", "playId"], group_keys=True)
            .apply(
                lambda x: smooth_data_y(
                    x, transition_covariance, observation_covariance
                )
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
                "dir",
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
                "dir",
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
        print(f"finished week {i}")


if __name__ == "__main__":
    run_pipeline()
