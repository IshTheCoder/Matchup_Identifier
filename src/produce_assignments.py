"""Produce frame-by-frame blocker->rusher assignment probabilities from the fitted
HMM parameters, in the long `assignment_data` format consumed by feature_engineering.

For each possession and each blocker, computes the smoothed posterior
gamma[t, k] = P(blocker guards rusher k at frame t) via forward-backward, using the
fitted (tau, sigma, rho) for that blocker's officialPosition_pb, the von Mises
orientation emission, and the conditional-logit matchup prior as the initial
distribution (log_pi = log_softmax_k(standardize(X) . beta)) when the fitted params
carry a beta; otherwise a uniform initial distribution. Emits one row per
(frame, blocker nflId, rusher nflId_pr) with that probability in `assignment_probs`.

Output columns: gameId, playId, frameId, nflId_pr, nflId, assignment_probs

Possessions are bucketed by k and padded to a fixed (T_max, J_max) per bucket so only
a handful of jit shapes compile per week.

Run:  python src/produce_assignments.py          # all 8 weeks -> assignment_data.csv
"""

import os
from collections import defaultdict

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import baum_welch_jax as bwj
from data_processing import possession_to_voxel, possession_to_prior_features

POSITIONS = ("T", "C", "RB", "TE", "WR", "FB", "G")
ORIENT, CONC = "vonmises", 1.0  # emission orientation (matches the fitted params / paper)
_DUMMY = (np.array([0.5, 0.5]), 1.0, 0.9, None, None, None)  # for padded/unknown blockers

# Phase 2: disengaged ("null") state. When on, each blocker gets a (k+1)-th hidden state
# whose emission is a flat positional background C_B plus a uniform orientation; the null
# state is emitted with nflId_pr == NULL_RUSHER. NULL_INIT is its snap-frame prior prob.
NULL_STATE = True
STRUCTURED_NULL = True  # Phase 2.5: asymmetric transition {rho_stay, p_fail (enter), rho_null (stay)}
C_B = -5.0          # background log-density (week-0 sweep elbow: ~7.6% mean null mass)
NULL_INIT = 0.05    # initial (snap) probability mass on the disengaged state
NULL_RUSHER = -1    # sentinel nflId_pr for the null state in assignment_data.csv
_PFAIL_DEFAULT = 0.01   # fallback engaged->null hazard for unseen blockers
_RHONULL_DEFAULT = 0.99  # fallback null persistence


def _parse_vec(s):
    """parse a numpy-array repr cell (handles line-wrapped reprs); -> 1-D float array"""
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=float)
    return np.fromstring(str(s).replace("\n", " ").strip().strip("[]"), sep=" ")


def load_fitted_params(path="fitted_params_jax.csv"):
    """position -> (tau, sigma, rho, beta|None, feat_mean|None, feat_std|None)"""
    df = pd.read_csv(path)
    has_beta = "beta" in df.columns
    out = {}
    for _, row in df.iterrows():
        beta = mean = std = None
        if has_beta and isinstance(row.get("beta"), str) and row["beta"].strip() not in ("", "nan", "None"):
            beta = _parse_vec(row["beta"])
            mean = _parse_vec(row["feat_mean"])
            std = _parse_vec(row["feat_std"])
        out[row["position"]] = (
            _parse_vec(row["tau"]), float(row["sigma"]), float(row["rho"]), beta, mean, std
        )
    return out


def load_rho_by_player(path="rho_by_player.csv"):
    """nflId -> per-player stickiness rho_j (hierarchical fit); {} if the file is absent.

    When present, produce_assignments builds each blocker's transition from its own rho_j,
    falling back to the position-level rho (the Beta mean a/(a+b)) for unseen blockers.
    """
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {int(r["nflId"]): float(r["rho"]) for _, r in df.iterrows()}


def load_pfail_by_player(path="rho_by_player.csv"):
    """nflId -> per-player engaged->null hazard p_fail (Phase 2.5); {} if absent / no column"""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "p_fail" not in df.columns:
        return {}
    return {int(r["nflId"]): float(r["p_fail"]) for _, r in df.iterrows()}


def load_rho_null_by_position(path="fitted_params_jax.csv"):
    """position -> per-position null persistence rho_null (Phase 2.5); {} if absent / no column"""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "rho_null" not in df.columns:
        return {}
    return {row["position"]: float(row["rho_null"]) for _, row in df.iterrows()}


def _blocker_gamma(tau, sigma, rho, p_fail, rho_null, O, B, D_b, t_mask, log_pi, c_b):
    """smoothed posterior gamma (t, k[+1]) for one blocker (D_b (t,1,C), log_pi (k[+1],))"""
    log_emis = bwj.emission_logprob(
        tau, sigma, O, B, D_b, ORIENT, CONC, null_state=NULL_STATE, c_b=c_b)[:, 0, :]
    K = O.shape[1]
    if NULL_STATE and STRUCTURED_NULL:  # Phase 2.5: asymmetric {rho, p_fail, rho_null}
        log_T = bwj.build_log_transition_null_batched(rho, p_fail, rho_null, K)
    else:
        log_T = bwj.build_log_transition(rho, K + (1 if NULL_STATE else 0))
    gamma, _, _ = bwj.forward_backward_single(log_emis, log_pi, log_T, t_mask)
    return gamma


# vmap over blockers (inner) and possessions (outer); O/B/t_mask/c_b shared per possession
_gamma_batch = jax.jit(
    jax.vmap(
        jax.vmap(_blocker_gamma, in_axes=(0, 0, 0, 0, 0, None, None, 0, None, 0, None)),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None),
    )
)


def _extract(poss, plays_df, pff_df):
    """possession -> (B, O, D, frame_ids, rusher_ids, blocker_ids, positions, game, play, X)"""
    B, O, D = possession_to_voxel(poss)  # B (t,2), O (t,k,3), D (t,j,3)
    rusher_ids = np.sort(poss["nflId_pr"].unique())
    blocker_ids = np.sort(poss["nflId"].unique())
    frame_ids = (
        poss.drop_duplicates("time")[["time", "frameId"]]
        .sort_values("time")["frameId"].to_numpy()
    )
    pos_of = poss.drop_duplicates("nflId").set_index("nflId")["officialPosition_pb"].to_dict()
    positions = [pos_of[b] for b in blocker_ids]
    X, _, _ = possession_to_prior_features(poss, plays_df, pff_df)  # (j, k, F)
    return (B, O, D, frame_ids, rusher_ids, blocker_ids, positions,
            poss["gameId"].iloc[0], poss["playId"].iloc[0], X)


def _emit_rows(rec, gamma):
    """vectorized long rows for one possession; gamma is (Jmax, Tmax, k)"""
    (_, _, _, frame_ids, rusher_ids, blocker_ids, positions, game, play, _, fitted) = rec
    ids = np.append(rusher_ids, NULL_RUSHER) if NULL_STATE else np.asarray(rusher_ids)
    t, k = len(frame_ids), len(ids)  # k = #rushers (+1 for the null state)
    cols = {c: [] for c in ("gameId", "playId", "frameId", "nflId_pr", "nflId", "assignment_probs")}
    for bi, (bid, posn) in enumerate(zip(blocker_ids, positions)):
        if posn not in fitted:
            continue
        g = np.asarray(gamma[bi, :t, :])
        cols["gameId"].append(np.full(t * k, game))
        cols["playId"].append(np.full(t * k, play))
        cols["frameId"].append(np.repeat(frame_ids, k))
        cols["nflId_pr"].append(np.tile(ids, t))
        cols["nflId"].append(np.full(t * k, bid))
        cols["assignment_probs"].append(g.reshape(-1))
    if not cols["gameId"]:
        return None
    return {c: np.concatenate(v) for c, v in cols.items()}


def _prior_log_pi(X_bi, par, k):
    """initial log-distribution for one blocker: softmax(standardize(X).beta) or uniform.

    With NULL_STATE, returns (k+1,): rushers share (1-NULL_INIT) of the mass via the
    conditional logit, and the disengaged state gets NULL_INIT.
    """
    _, _, _, beta, mean, std = par
    if beta is None:
        base = np.full(k, -np.log(k))
    else:
        s = ((X_bi - mean) / std) @ beta  # (k,)
        base = s - np.logaddexp.reduce(s)
    if not NULL_STATE:
        return base
    out = np.empty(k + 1)
    out[:k] = np.log1p(-NULL_INIT) + base
    out[k] = np.log(NULL_INIT)
    return out


def _process_week(data, fitted, plays_df, pff_df, rho_player=None, c_b=C_B,
                  pfail_player=None, rho_null_by_pos=None):
    """all assignment rows for one week's smoothed data, as a single DataFrame

    rho_player (optional): nflId -> per-player stickiness; when given, each blocker's
    transition uses its own rho_j (fallback to the position rho for unseen blockers).
    c_b: background log-density for the disengaged (null) state when NULL_STATE.
    pfail_player / rho_null_by_pos (Phase 2.5): nflId->p_fail and position->rho_null for the
    structured null transition (fallback to defaults for unseen blockers/positions).
    """
    rho_player = rho_player or {}
    pfail_player = pfail_player or {}
    rho_null_by_pos = rho_null_by_pos or {}
    recs = [_extract(poss, plays_df, pff_df) for _, poss in data.groupby("possession_id")]
    out_chunks = []
    ns = 1 if NULL_STATE else 0  # extra (null) state appended to each chain

    # k==1 possessions: blocker trivially guards the only rusher (prob 1; null prob 0)
    for r in recs:
        rusher_ids, blocker_ids, frame_ids = r[4], r[5], r[3]
        if len(rusher_ids) != 1:
            continue
        gamma = np.ones((len(blocker_ids), len(frame_ids), 1))
        if NULL_STATE:  # append a zero-probability null column to keep the schema
            gamma = np.concatenate([gamma, np.zeros_like(gamma)], axis=-1)
        rows = _emit_rows(r + (fitted,), gamma)
        if rows:
            out_chunks.append(pd.DataFrame(rows))

    # k>=2: bucket by k, pad to (T_max, J_max), run one batched forward-backward
    buckets = defaultdict(list)
    for r in recs:
        if r[1].shape[1] >= 2:
            buckets[r[1].shape[1]].append(r)

    for k, group in buckets.items():
        P = len(group)
        T_max = max(r[1].shape[0] for r in group)
        J_max = max(len(r[5]) for r in group)

        O_a = np.zeros((P, T_max, k, 3))
        B_a = np.zeros((P, T_max, 2))
        Db = np.zeros((P, J_max, T_max, 1, 3))
        t_mask = np.zeros((P, T_max))
        taus = np.tile(_DUMMY[0], (P, J_max, 1))
        sigmas = np.full((P, J_max), _DUMMY[1])
        rhos = np.full((P, J_max), _DUMMY[2])
        pfails = np.full((P, J_max), _PFAIL_DEFAULT)
        rho_nulls = np.full((P, J_max), _RHONULL_DEFAULT)
        log_pi = np.tile(np.full(k + ns, -np.log(k + ns)), (P, J_max, 1))  # default uniform (k[+null])

        for p, r in enumerate(group):
            B, O, D, frame_ids, rusher_ids, blocker_ids, positions, game, play, X = r
            t, j = O.shape[0], D.shape[1]
            O_a[p, :t] = O
            B_a[p, :t] = B[:, :2]
            t_mask[p, :t] = 1.0
            for bi in range(j):
                Db[p, bi, :t, 0, :] = D[:, bi, :]
                par = fitted.get(positions[bi], _DUMMY)
                taus[p, bi] = par[0]
                sigmas[p, bi] = par[1]
                rhos[p, bi] = rho_player.get(int(blocker_ids[bi]), par[2])
                pfails[p, bi] = pfail_player.get(int(blocker_ids[bi]), _PFAIL_DEFAULT)
                rho_nulls[p, bi] = rho_null_by_pos.get(positions[bi], _RHONULL_DEFAULT)
                log_pi[p, bi] = _prior_log_pi(X[bi], par, k)

        gammas = np.asarray(
            _gamma_batch(
                jnp.asarray(taus), jnp.asarray(sigmas), jnp.asarray(rhos),
                jnp.asarray(pfails), jnp.asarray(rho_nulls),
                jnp.asarray(O_a), jnp.asarray(B_a), jnp.asarray(Db),
                jnp.asarray(t_mask), jnp.asarray(log_pi), jnp.asarray(float(c_b)),
            )
        )  # (P, J_max, T_max, k[+1])

        for p, r in enumerate(group):
            rows = _emit_rows(r + (fitted,), gammas[p])
            if rows:
                out_chunks.append(pd.DataFrame(rows))

    if not out_chunks:
        return None
    return pd.concat(out_chunks, ignore_index=True)


def produce_assignments(
    csv_paths=None,
    fitted_path="fitted_params_jax.csv",
    out_path="assignment_data.csv",
    rho_by_player_path="rho_by_player.csv",
    c_b=C_B,
):
    """write frame-by-frame assignment_data.csv from every possession in every week"""
    if csv_paths is None:
        csv_paths = [f"data/sample_data_week_{i}.csv" for i in range(8)]
    elif isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    fitted = load_fitted_params(fitted_path)
    rho_player = load_rho_by_player(rho_by_player_path)
    pfail_player = load_pfail_by_player(rho_by_player_path)
    rho_null_by_pos = load_rho_null_by_position(fitted_path)
    structured = bool(NULL_STATE and STRUCTURED_NULL and pfail_player and rho_null_by_pos)
    has_prior = any(v[3] is not None for v in fitted.values())
    print(f"loaded fitted params for {sorted(fitted)}; matchup prior: "
          f"{'on' if has_prior else 'off (uniform)'}; orientation={ORIENT}; "
          f"per-player rho: {'on (' + str(len(rho_player)) + ' players)' if rho_player else 'off (position rho)'}; "
          f"null state: {'off' if not NULL_STATE else ('structured (c_b=' + str(c_b) + ', p_fail+rho_null fit)' if structured else 'symmetric (c_b=' + str(c_b) + ')')}")
    plays_df = pd.read_csv("data/plays.csv")
    pff_df = pd.read_csv("data/pffScoutingData.csv")

    cols = ["gameId", "playId", "frameId", "nflId_pr", "nflId", "assignment_probs"]
    if os.path.exists(out_path):  # we append per-week below; start from a clean file
        os.remove(out_path)
    wrote_header = False
    total = 0
    for path in csv_paths:
        print(f"processing {path}", flush=True)
        data = pd.read_csv(path)
        week_df = _process_week(data, fitted, plays_df, pff_df, rho_player, c_b,
                                pfail_player, rho_null_by_pos)
        if week_df is None:
            continue
        week_df[cols].to_csv(out_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        total += len(week_df)
        print(f"  wrote {len(week_df)} rows (cumulative {total})", flush=True)
        jax.clear_caches()  # bound the XLA compile cache across weeks

    print(f"done: {total} rows -> {out_path}", flush=True)
    return total


if __name__ == "__main__":
    produce_assignments()
