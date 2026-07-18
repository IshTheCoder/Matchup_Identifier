"""Parquet export of MCMC model outputs to model_outputs/ for cross-language (R) access.

Posterior samples stay pickled for Python reuse; this additionally writes typed, tidy parquet that R
reads via the `arrow` package. For each fitted per-player effect we write two artifacts:
  model_outputs/{name}_summary.parquet : one row per player (nflId, name, pos, mean, lo, hi, [snaps])
  model_outputs/{name}_draws.parquet   : tidy posterior draws  (nflId, draw, value)
and arbitrary derived tables via export_table().
"""
import os
import numpy as np
import pandas as pd

MODEL_DIR = "model_outputs"


def _ensure():
    os.makedirs(MODEL_DIR, exist_ok=True)


def reconstruct_draws(samples, design, kind):
    """Per-entity effect draws (D, N) for kind in {'rusher','blocker','quarterback'}.

    Standard attribute-centered form  weight @ covariates.T + sigma * z, or a direct posterior site
    '{kind}_effect' when the model samples the effect directly (e.g. the centered dose model's
    'blocker_effect')."""
    direct = f"{kind}_effect"
    if direct in samples:
        return np.asarray(samples[direct])
    w = np.asarray(samples[f"{kind}_weight"])
    z = np.asarray(samples[f"z_{kind}"])
    s = np.asarray(samples[f"sigma_{kind}"])
    cov = np.asarray(design[f"{kind}_covariates"])
    return w @ cov.T + s[:, None] * z


def export_effect(name, draws, ids, players, snaps=None):
    """Write {name}_summary.parquet + {name}_draws.parquet for a per-entity effect.
    draws: (D, N) posterior draws; ids: the nflId for each of the N columns; players: players.csv
    indexed by nflId. Returns the summary DataFrame."""
    _ensure()
    D, N = draws.shape
    ids = np.asarray(ids)
    summ = pd.DataFrame({
        "nflId": ids,
        "name": [players["displayName"].get(i, "?") for i in ids],
        "pos": [players["officialPosition"].get(i, "?") for i in ids],
        "mean": draws.mean(0),
        "lo": np.percentile(draws, 2.5, axis=0),
        "hi": np.percentile(draws, 97.5, axis=0),
    })
    if snaps is not None:
        summ["snaps"] = np.asarray(snaps)
    summ.to_parquet(f"{MODEL_DIR}/{name}_summary.parquet", index=False)
    tidy = pd.DataFrame({
        "nflId": np.repeat(ids, D),
        "draw": np.tile(np.arange(D), N),
        "value": draws.T.reshape(-1),
    })
    tidy.to_parquet(f"{MODEL_DIR}/{name}_draws.parquet", index=False)
    return summ


def export_table(name, df):
    """Write any derived DataFrame to model_outputs/{name}.parquet."""
    _ensure()
    df.to_parquet(f"{MODEL_DIR}/{name}.parquet", index=False)
