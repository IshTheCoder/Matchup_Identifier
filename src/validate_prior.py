"""Validate the matchup prior: does it improve snap-frame assignment vs uniform, and
does its influence fade over the play?

Fits a position with prior="two_stage", then for each possession runs the per-blocker
forward-backward under (a) uniform initial distribution and (b) the learned conditional
logit, and compares the snap-frame (t=0) argmax assignment to PFF's hand-labeled
pff_nflIdBlockedPlayer (held out from fitting). Also reports the per-frame divergence
between the two to confirm the prior fades.

Run:  python src/validate_prior.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import baum_welch_jax as bwj
from data_processing import possession_to_voxel, possession_to_prior_features

ORIENT, CONC = "vonmises", 1.0
_fb = jax.jit(bwj.forward_backward_single)


def _snap_assign(emis_log, log_pi, rho, k, t):
    """smoothed posterior gamma (t,k) for one blocker"""
    gamma, _, _ = _fb(emis_log, log_pi, bwj.build_log_transition(rho, k), jnp.ones(t))
    return np.asarray(gamma)


def validate(position="T", fit_path="data/sample_data_week_0.csv",
             eval_path="data/sample_data_week_0.csv", n_iter=12):
    plays = pd.read_csv("data/plays.csv")
    pff = pd.read_csv("data/pffScoutingData.csv")

    # --- fit the prior on the fit week ---
    res = bwj.fit_by_position(csv_paths=[fit_path], positions=(position,), n_iter=n_iter,
                              orientation=ORIENT, conc=CONC, prior="two_stage",
                              out_path="/tmp/_fit_prior.csv")[0]
    tau = jnp.asarray(res["tau"]); sigma = float(res["sigma"]); rho = float(res["rho"])
    beta = jnp.asarray(res["beta"]); mean = np.asarray(res["feat_mean"]); std = np.asarray(res["feat_std"])

    # --- evaluate snap-frame assignment vs PFF labels ---
    data = pd.read_csv(eval_path)
    pos_data = data[data["officialPosition_pb"] == position]
    blocked = pff.set_index(["gameId", "playId", "nflId"])["pff_nflIdBlockedPlayer"]

    n = 0
    hit_prior_alone = hit_uniform_em = hit_prior_em = hit_nearest = 0
    kl_by_frame = {}  # frame -> [KL(prior||uniform) per blocker]
    for _, poss in pos_data.groupby("possession_id"):
        B, O, D = possession_to_voxel(poss)
        k, j, t = O.shape[1], D.shape[1], O.shape[0]
        if k == 1:
            continue
        X, bids, rids = possession_to_prior_features(poss, plays, pff)
        Xs = (X - mean) / std  # (j,k,F) standardized like training
        gid, pid = poss["gameId"].iloc[0], poss["playId"].iloc[0]
        Oj, Bj = jnp.asarray(O), jnp.asarray(B)
        for bi, bid in enumerate(bids):
            lbl = blocked.get((gid, pid, bid), np.nan)
            if not (lbl in rids):  # PFF label must point to a known rusher
                continue
            truth = int(np.where(rids == lbl)[0][0])
            emis = bwj.emission_logprob(tau, sigma, Oj, Bj,
                                        jnp.asarray(D[:, bi, :])[:, None, :], ORIENT, CONC)[:, 0, :]
            s = Xs[bi] @ np.asarray(beta)          # (k,) prior scores
            log_pi_prior = jnp.asarray(s - np.logaddexp.reduce(s))
            log_pi_unif = jnp.log(jnp.ones(k) / k)
            g_prior = _snap_assign(emis, log_pi_prior, rho, k, t)
            g_unif = _snap_assign(emis, log_pi_unif, rho, k, t)
            n += 1
            hit_prior_alone += int(np.argmax(np.asarray(s)) == truth)
            hit_uniform_em += int(np.argmax(g_unif[0]) == truth)
            hit_prior_em += int(np.argmax(g_prior[0]) == truth)
            hit_nearest += int(np.argmin(X[bi, :, 3]) == truth)  # feature 3 = dist
            for fr in range(t):
                p, q = g_prior[fr] + 1e-12, g_unif[fr] + 1e-12
                kl_by_frame.setdefault(fr, []).append(float(np.sum(p * np.log(p / q))))

    print(f"\n=== {position}: snap-frame top-1 accuracy vs PFF blocked-player (n={n}) ===")
    print(f"  nearest-rusher baseline : {hit_nearest/n:.3f}")
    print(f"  prior alone (argmax pi) : {hit_prior_alone/n:.3f}")
    print(f"  uniform prior + emission: {hit_uniform_em/n:.3f}")
    print(f"  learned prior + emission: {hit_prior_em/n:.3f}")
    print("\n=== fade: mean KL(prior||uniform) of gamma_t by frame ===")
    for fr in [0, 2, 5, 10, 20, 30]:
        if fr in kl_by_frame:
            print(f"  frame {fr:>2}: {np.mean(kl_by_frame[fr]):.4f}")


if __name__ == "__main__":
    for pos in ("T", "RB"):
        validate(position=pos)
