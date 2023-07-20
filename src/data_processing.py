from typing import Tuple

import numpy as np
import pandas as pd


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
        .apply(lambda x: x[["x_smooth", "y_smooth"]].head(1).values.flatten())
        .reset_index()
        .groupby("nflId")
        .apply(lambda x: np.array(x.sort_values("time")[0].values.tolist()))
        .reset_index()[0]
        .values.tolist(),
        axis=-1,
    ).swapaxes(-2, -1)

    O = np.stack(
        data.groupby(["nflId_pr", "time"])
        .apply(lambda x: x[["x_smooth_pr", "y_smooth_pr"]].head(1).values.flatten())
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
    I = np.hstack([I.flatten(), I.flatten()])
    X = np.column_stack([O.flatten(), B.flatten()])
    y = D.flatten()[:, np.newaxis]
    return X, y, I
