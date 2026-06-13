"""JAX port of the defensive-matchup Baum-Welch EM (closed-form M-steps).

Mirrors the numpy reference in baum_welch.py exactly (same model, same closed-form
updates, same emission-normalization / unnormalized-log-likelihood split), but
vectorized with jax: log-space forward-backward via lax.scan, vmap over defenders
and possessions, and closed-form M-steps expressed as einsum accumulators (no giant
design matrix materialized).

Batching strategy: possessions are RAGGED (frames t, blockers j, rushers k all vary).
We group possessions by k (the HMM state dimension; few distinct values), and within a
group pad t->T_max and j->J_max with boolean masks. k is never masked, so the
transition (1-rho)/(k-1) and the state dimension stay exact and no -inf masking is
needed -- the only masks are multiplicative (gamma/xi -> 0) and where-based identity
carries for padded frames.

The numpy module remains the reference oracle; test_em_recovery_jax.py asserts tight
numerical equivalence.
"""

from collections import defaultdict
from functools import partial
from typing import Dict, List, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import digamma, i0e, logsumexp
from jax.scipy.stats import beta, norm


class Params(NamedTuple):
    tau: jnp.ndarray  # (2,) simplex weights (rusher, qb)
    sigma: jnp.ndarray  # scalar position variance
    rho: jnp.ndarray  # scalar stickiness (legacy; in hierarchical mode = Beta mean a/(a+b))
    beta: jnp.ndarray = None  # (F,) conditional-logit prior weights; None in "free" mode
    rho_player: jnp.ndarray = None  # (n_players,) per-player stickiness (hierarchical mode only)
    ab: jnp.ndarray = None  # (2,) Beta(a,b) hyperparameters (hierarchical mode only)


class KGroup(NamedTuple):
    """A batch of possessions that all share the same number of rushers k.

    Arrays are right-padded over frames (T_max) and blockers (J_max); the masks mark
    valid entries. k and orientation usage are recoverable from the array shapes.
    """

    O: jnp.ndarray  # (P, T_max, k, C) rusher pos (+ dir if C==3)
    B: jnp.ndarray  # (P, T_max, 2)   qb pos
    D: jnp.ndarray  # (P, T_max, J_max, C) blocker pos (+ dir if C==3)
    t_mask: jnp.ndarray  # (P, T_max) valid frames
    j_mask: jnp.ndarray  # (P, J_max) valid blockers
    log_pi: jnp.ndarray  # (P, J_max, k) per-defender log prior (used in "free" mode)
    X: jnp.ndarray  # (P, J_max, k, F) pre-snap pair features (used in "logit" mode)
    blocker_id: jnp.ndarray = None  # (P, J_max) position-level contiguous player index; -1 pad (hierarchical mode)


# --------------------------------------------------------------------------- #
# emission
# --------------------------------------------------------------------------- #
def _orientation_current(O: jnp.ndarray) -> jnp.ndarray:
    """legacy orientation term (reproduces compute_emission in baum_welch.py): (...,T,k)

    Uses ONLY the rusher orientation (O dir), a method-of-moments Beta with a
    hard-coded 0.85 variance, evaluated at the deterministic complement 1 - Z_rusher.
    Kept for backward compatibility / reproducing the current fitted params; note it
    never reads the observed blocker orientation.
    """
    cos = jnp.cos(jnp.deg2rad(O[..., -1]))
    raw_rusher = 0.5 * (cos + 1.0)
    raw_blocker = 0.5 * (-cos + 1.0)
    r = jnp.where(raw_rusher == 0, 0.001, jnp.where(raw_rusher == 1, 0.999, raw_rusher))
    b = jnp.where(raw_blocker == 0, 0.001, jnp.where(raw_blocker == 1, 0.999, raw_blocker))
    var = 0.85 * r * (1.0 - r)
    alpha = jnp.power(r, 2) * ((1.0 - r) / var - 1.0 / r)
    bparam = alpha * (1.0 / r - 1.0)
    return beta.logpdf(b, alpha, bparam)  # (..., T, k)


def _orientation_paper(O: jnp.ndarray, D: jnp.ndarray, alpha: float = 2.0,
                       eps: float = 1e-3) -> jnp.ndarray:
    """paper-faithful orientation term: (...,T,J,k)

    Scores the OBSERVED blocker orientation Z_tj = (cos(eta_blocker)+1)/2 under
    Beta(alpha, alpha*Z_tk/(1-Z_tk)) whose mean is 1 - Z_tk (Z_tk from the rusher
    orientation). alpha is fixed (=2 in the paper). This rewards a blocker facing
    opposite its assigned rusher. (paper.tex eqs 201-206 / 527.)
    """
    Zj = jnp.clip(0.5 * (jnp.cos(jnp.deg2rad(D[..., -1])) + 1.0), eps, 1.0 - eps)  # (...,T,J)
    Zk = jnp.clip(0.5 * (jnp.cos(jnp.deg2rad(O[..., -1])) + 1.0), eps, 1.0 - eps)  # (...,T,k)
    beta_param = alpha * Zk / (1.0 - Zk)  # (..., T, k)
    return beta.logpdf(Zj[..., :, None], alpha, beta_param[..., None, :])  # (...,T,J,k)


def _orientation_vonmises(O: jnp.ndarray, D: jnp.ndarray, kappa: float) -> jnp.ndarray:
    """von Mises (circular normal) orientation term: (...,T,J,k)

    Models the blocker's facing angle as von Mises with mean opposite its assigned
    rusher (mu = eta_rusher + pi) and concentration kappa:
        log density = -kappa * cos(eta_blocker - eta_rusher) - log(2*pi*I0(kappa)).
    The across-rusher spread is kappa*spread_k(cos) <= 2*kappa -- bounded and linear in
    kappa, so kappa cleanly controls how much orientation matters (no Beta-style blowup
    near axis-aligned rushers). This is the natural distribution for angular data.
    """
    ang_r = jnp.deg2rad(O[..., -1])  # (..., T, k) rusher facing
    ang_b = jnp.deg2rad(D[..., -1])  # (..., T, J) blocker facing
    delta = ang_b[..., :, None] - ang_r[..., None, :]  # (..., T, J, k)
    log_norm = jnp.log(2.0 * jnp.pi) + jnp.log(i0e(kappa)) + kappa  # log(2*pi*I0(kappa))
    return -kappa * jnp.cos(delta) - log_norm


def emission_logprob(
    tau: jnp.ndarray,
    sigma: jnp.ndarray,
    O: jnp.ndarray,
    B: jnp.ndarray,
    D: jnp.ndarray,
    orientation: str = "current",
    conc: float = 2.0,
    null_state: bool = False,
    c_b: float = -6.0,
) -> jnp.ndarray:
    """unnormalized log emission densities, batched over any leading dims -> (...,T,J,k)

    Always includes the isotropic 2-D Gaussian on position centered on
    tau_r*rusher + tau_q*qb. The orientation term is selectable:
      "none"     : position only.
      "current"  : legacy Beta (rusher-only, MoM); reproduces baum_welch.py.
      "paper"    : paper-faithful Beta on the observed blocker orientation; conc = alpha.
      "vonmises" : von Mises on the blocker facing angle; conc = kappa (concentration).
    `conc` is the concentration hyperparameter for the paper/vonmises terms. If O lacks
    a 3rd (dir) column, falls back to position-only regardless of mode.

    When null_state, a (k+1)-th "disengaged" state is appended with a flat positional
    background log-density c_b and a uniform orientation (von Mises kappa=0, log-density
    -log(2*pi)); it captures frames where the blocker engages no rusher. -> (...,T,J,k+1)
    """
    O2 = O[..., 0:2]
    B2 = B[..., 0:2]
    D2 = D[..., 0:2]
    mu = tau[0] * O2 + tau[1] * B2[..., None, :]  # (..., T, k, 2)
    std = jnp.sqrt(sigma)
    lp_xy = norm.logpdf(D2[..., :, None, :], loc=mu[..., None, :, :], scale=std)
    log_loc = lp_xy.sum(-1)  # (..., T, J, k)

    has_orient = not (orientation == "none" or O.shape[-1] < 3)
    if not has_orient:
        em = log_loc
    elif orientation == "current":
        em = log_loc + _orientation_current(O)[..., None, :]  # (...,T,k)->(...,T,1,k)
    elif orientation == "paper":
        em = log_loc + _orientation_paper(O, D, conc)  # (...,T,J,k)
    elif orientation == "vonmises":
        em = log_loc + _orientation_vonmises(O, D, conc)  # (...,T,J,k)
    else:
        raise ValueError(f"unknown orientation mode {orientation!r}")

    if null_state:
        # flat positional background + uniform (kappa=0) orientation, broadcast to (...,T,J,1)
        null_lp = c_b + (-jnp.log(2.0 * jnp.pi) if has_orient else 0.0)
        null_col = jnp.full(em.shape[:-1] + (1,), null_lp, dtype=em.dtype)
        em = jnp.concatenate([em, null_col], axis=-1)
    return em


# --------------------------------------------------------------------------- #
# transition
# --------------------------------------------------------------------------- #
def build_log_transition(rho: jnp.ndarray, k: int) -> jnp.ndarray:
    """log of the (k, k) transition: rho on the diagonal, (1-rho)/(k-1) off"""
    off = (1.0 - rho) / (k - 1)
    T = jnp.full((k, k), off)
    T = T.at[jnp.diag_indices(k)].set(rho)
    return jnp.log(T)


def build_log_transition_batched(rho: jnp.ndarray, k: int) -> jnp.ndarray:
    """vectorized build_log_transition: rho (...,) -> (..., k, k) log-transition

    Each leading entry gets its own stickiness: rho on the diagonal, (1-rho)/(k-1) off.
    Used by the hierarchical per-blocker rho path (rho indexed per (possession, blocker)).
    """
    off = (1.0 - rho) / (k - 1)  # (...)
    T = jnp.broadcast_to(off[..., None, None], rho.shape + (k, k))
    eye = jnp.eye(k, dtype=bool)
    T = jnp.where(eye, rho[..., None, None], T)
    return jnp.log(T)


def build_log_transition_null_batched(rho, p_fail, rho_null, K):
    """structured (K+1, K+1) log-transition with a disengaged/null state (index K).

    Column b = from-state, row a = to-state (columns sum to 1). For an engaged origin
    i<K: stay rho (diagonal), switch to another rusher (1-rho-p_fail)/(K-1), fail to null
    p_fail. For the null origin: stay rho_null, recover to each rusher (1-rho_null)/K.
    rho, p_fail are (...,) per (possession, blocker); rho_null is a scalar (per position).
    -> (..., K+1, K+1).
    """
    shape = jnp.asarray(rho).shape
    n = K + 1
    sw = (1.0 - rho - p_fail) / (K - 1)  # (...) engaged->other rusher
    eyeK = jnp.eye(K)
    block = sw[..., None, None] * (1.0 - eyeK) + rho[..., None, None] * eyeK  # (...,K,K)
    T = jnp.zeros(shape + (n, n))
    T = T.at[..., :K, :K].set(block)
    T = T.at[..., K, :K].set(jnp.broadcast_to(p_fail[..., None], shape + (K,)))  # engaged->null
    rec = (1.0 - rho_null) / K
    T = T.at[..., :K, K].set(rec)          # null->rusher (broadcast scalar)
    T = T.at[..., K, K].set(rho_null)      # null->null
    return jnp.log(jnp.clip(T, 1e-12, 1.0))


# --------------------------------------------------------------------------- #
# E-step: log-space forward-backward for one (t, k) sequence
# --------------------------------------------------------------------------- #
def forward_backward_single(log_emis_unn, log_pi, log_T, t_mask):
    """returns (gamma (T,k), xi (T-1,k,k), data_loglik scalar) for one defender

    gamma/xi are computed from per-timestep k-normalized emissions (matches the numpy
    E-step); the data log-likelihood is computed on the UNnormalized emissions via the
    forward scaling constants (matches sequence_log_likelihood). Padded frames (t_mask
    == 0) act as identity transitions and contribute nothing.
    """
    T, k = log_emis_unn.shape
    log_emis_n = log_emis_unn - logsumexp(log_emis_unn, axis=-1, keepdims=True)

    # ---- forward (normalized emissions) ----  t=0 always valid (right padding)
    a0 = log_pi + log_emis_n[0]
    a0 = a0 - logsumexp(a0)

    def fwd_step(carry, inp):
        log_e, valid = inp
        pred = logsumexp(log_T + carry[None, :], axis=1)  # log(T @ alpha)
        a_unn = jnp.where(valid, pred + log_e, carry)
        a = a_unn - logsumexp(a_unn)
        return a, a

    _, a_rest = jax.lax.scan(fwd_step, a0, (log_emis_n[1:], t_mask[1:]))
    log_alpha = jnp.concatenate([a0[None, :], a_rest], axis=0)  # (T, k)

    # ---- backward (normalized emissions) ----  base case beta(last)=1 -> log 0
    bT = jnp.zeros(k)

    def bwd_step(carry, inp):
        log_e_next, valid_next = inp  # emission/mask at t+1
        prod = log_e_next + carry
        b_unn = jnp.where(valid_next, logsumexp(log_T + prod[None, :], axis=1), carry)
        b = b_unn - logsumexp(b_unn)
        return b, b

    _, b_rev = jax.lax.scan(
        bwd_step, bT, (log_emis_n[1:], t_mask[1:]), reverse=True
    )
    log_beta = jnp.concatenate([b_rev, bT[None, :]], axis=0)  # (T, k)

    # ---- gamma ----
    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - logsumexp(log_gamma, axis=-1, keepdims=True)
    gamma = jnp.exp(log_gamma) * t_mask[:, None]

    # ---- xi ----  xi[i,a,b] = T[a,b] * (beta*pdf)_{i+1}[a] * alpha_i[b], joint-normed
    term_a = log_beta[1:] + log_emis_n[1:]  # (T-1, k) index a (state at t+1)
    term_b = log_alpha[:-1]  # (T-1, k) index b (state at t)
    log_xi = log_T[None, :, :] + term_a[:, :, None] + term_b[:, None, :]
    log_xi = log_xi - logsumexp(log_xi, axis=(1, 2), keepdims=True)
    xi_mask = t_mask[:-1] * t_mask[1:]
    xi = jnp.exp(log_xi) * xi_mask[:, None, None]

    # ---- data log-likelihood (UNnormalized forward) ----
    la0 = log_pi + log_emis_unn[0]
    c0 = logsumexp(la0)
    la0n = la0 - c0

    def ll_step(carry, inp):
        log_e, valid = inp
        pred = logsumexp(log_T + carry[None, :], axis=1)
        a_unn = jnp.where(valid, pred + log_e, carry)
        c = logsumexp(a_unn)  # == 0 when invalid (carry is normalized)
        return a_unn - c, c

    _, cs = jax.lax.scan(ll_step, la0n, (log_emis_unn[1:], t_mask[1:]))
    loglik = c0 + cs.sum()

    return gamma, xi, loglik


def _fb_over_defenders(log_emis_unn_p, log_pi_p, log_T, t_mask_p):
    """vmap forward_backward_single over the J axis of one possession"""
    fn = lambda le, lp: forward_backward_single(le, lp, log_T, t_mask_p)
    # le: (T, k) from axis 1 of (T, J, k); lp: (k,) from axis 0 of (J, k)
    return jax.vmap(fn, in_axes=(1, 0))(log_emis_unn_p, log_pi_p)


def forward_backward_group(log_emis_unn, log_pi, log_T, t_mask):
    """E-step over a padded k-group.

    log_emis_unn (P,T,J,k), log_pi (P,J,k), t_mask (P,T) ->
    gamma (P,T,J,k), xi (P,J,T-1,k,k), loglik (P,J)
    """
    fn = lambda le, lp, tm: _fb_over_defenders(le, lp, log_T, tm)
    gamma_j, xi_j, ll_j = jax.vmap(fn, in_axes=(0, 0, 0))(log_emis_unn, log_pi, t_mask)
    gamma = jnp.transpose(gamma_j, (0, 2, 1, 3))  # (P,J,T,k) -> (P,T,J,k)
    return gamma, xi_j, ll_j


def _fb_over_defenders_perT(log_emis_unn_p, log_pi_p, log_T_p, t_mask_p):
    """vmap forward_backward_single over J, with a PER-BLOCKER transition log_T_p (J,k,k)"""
    fn = lambda le, lp, lT: forward_backward_single(le, lp, lT, t_mask_p)
    return jax.vmap(fn, in_axes=(1, 0, 0))(log_emis_unn_p, log_pi_p, log_T_p)


def forward_backward_group_perT(log_emis_unn, log_pi, log_T, t_mask):
    """E-step over a padded k-group with a per-(possession, blocker) transition.

    log_emis_unn (P,T,J,k), log_pi (P,J,k), log_T (P,J,k,k), t_mask (P,T) ->
    gamma (P,T,J,k), xi (P,J,T-1,k,k), loglik (P,J). Mirrors forward_backward_group
    but each blocker carries its own (k,k) transition (hierarchical per-player rho).
    """
    fn = lambda le, lp, lT, tm: _fb_over_defenders_perT(le, lp, lT, tm)
    gamma_j, xi_j, ll_j = jax.vmap(fn, in_axes=(0, 0, 0, 0))(log_emis_unn, log_pi, log_T, t_mask)
    gamma = jnp.transpose(gamma_j, (0, 2, 1, 3))  # (P,J,T,k) -> (P,T,J,k)
    return gamma, xi_j, ll_j


# --------------------------------------------------------------------------- #
# M-steps
# --------------------------------------------------------------------------- #
def betabinom_eb(S, W, a0=1.0, b0=1.0, n_steps=100):
    """empirical-Bayes Beta(a,b) for per-player stay/switch counts via Minka's fixed point

    S, W: (n_players,) expected retained / switched transition counts (soft, fractional).
    Treats each player's (S_j, W_j) as Beta-Binomial(a,b) and maximizes the marginal over
    (a,b) by the multiplicative fixed point (works with fractional counts via digamma).
    Players with no transitions (S_j=W_j=0) contribute zero and are harmless. Returns (a,b).
    """
    n = S + W
    a, b = a0, b0
    for _ in range(n_steps):
        ab = a + b
        denom = jnp.sum(digamma(n + ab) - digamma(ab))
        num_a = jnp.sum(digamma(S + a) - digamma(a))
        num_b = jnp.sum(digamma(W + b) - digamma(b))
        a = a * num_a / denom
        b = b * num_b / denom
    return a, b


def rho_posterior_mean(S, W, a, b):
    """Beta-Binomial posterior mean stickiness per player: (S_j + a)/(S_j + W_j + a + b)"""
    return (S + a) / (S + W + a + b)


def dirichlet_eb(counts, alpha0=None, n_steps=100):
    """empirical-Bayes Dirichlet(alpha) for per-player category counts via Minka's fixed point

    counts: (n_players, C) expected (soft, fractional) counts per category. Treats each
    player's row as Multinomial with a shared Dirichlet(alpha) prior and maximizes the
    marginal (Dirichlet-Multinomial) over alpha. Returns alpha (C,). The 3-category
    generalization of betabinom_eb (C=2 reproduces it). Posterior mean per player:
    (counts_jk + alpha_k) / (tot_j + sum_k alpha_k).
    """
    C = counts.shape[1]
    tot = counts.sum(axis=1)  # (n_players,)
    alpha = jnp.ones(C) if alpha0 is None else jnp.asarray(alpha0, dtype=float)
    for _ in range(n_steps):
        a0 = alpha.sum()
        denom = jnp.sum(digamma(tot + a0) - digamma(a0))
        num = jnp.sum(digamma(counts + alpha[None, :]) - digamma(alpha[None, :]), axis=0)  # (C,)
        alpha = alpha * num / denom
    return alpha


def dirichlet_posterior_mean(counts, alpha):
    """per-player posterior category probs: (counts_jk + alpha_k)/(tot_j + sum alpha)"""
    return (counts + alpha[None, :]) / (counts.sum(axis=1, keepdims=True) + alpha.sum())


def simplex_project(XtWX, XtWy):
    """constrained GLS: tau_gls projected onto {tau : 1' tau = 1} (== maximize_tau)"""
    tau_gls = jnp.linalg.solve(XtWX, XtWy)
    ones = jnp.ones(2)
    A = jnp.linalg.solve(XtWX, ones)  # (X'WX)^-1 1
    denom = ones @ A
    correction = (1.0 - ones @ tau_gls) / denom
    return tau_gls + A * correction


# --------------------------------------------------------------------------- #
# conditional-logit matchup prior: log_pi[p,j,k] = log_softmax_k(X[p,j,k,:] . beta)
# --------------------------------------------------------------------------- #
def log_pi_from_features(X, beta):
    """log_softmax over k of (X @ beta); X (P,J,k,F), beta (F,) -> (P,J,k)"""
    s = jnp.einsum("pjkf,f->pjk", X, beta)
    return s - logsumexp(s, axis=-1, keepdims=True)


@jax.jit
def _logit_grad_fisher(X, gamma0, j_mask, beta):
    """soft-label conditional-logit gradient (F,) and Fisher info (F,F) for one group

    Q(beta) = sum_{p,j valid} sum_k gamma0[p,j,k] log_softmax_k(X.beta); the snap-frame
    responsibilities gamma0 are the soft targets. grad = sum (gamma0 - pi) x; the Fisher
    information (= -Hessian, positive definite) gives the Newton step beta += FI^-1 grad.
    """
    s = jnp.einsum("pjkf,f->pjk", X, beta)
    pi = jax.nn.softmax(s, axis=-1)  # (P,J,k)
    resid = (gamma0 - pi) * j_mask[:, :, None]
    grad = jnp.einsum("pjk,pjkf->f", resid, X)
    xbar = jnp.einsum("pjk,pjkf->pjf", pi, X)  # (P,J,F)
    term1 = jnp.einsum("pjk,pjkf,pjkg->pjfg", pi, X, X)  # (P,J,F,F)
    fisher_each = term1 - jnp.einsum("pjf,pjg->pjfg", xbar, xbar)
    fisher = jnp.einsum("pj,pjfg->fg", j_mask, fisher_each)
    return grad, fisher


def fit_beta(group_data, beta0, n_newton=10, ridge=1e-3, tol=1e-6):
    """Newton-fit the conditional-logit beta to snap responsibilities, pooled over groups

    group_data: list of (X, gamma0, j_mask) per k-group. Warm-started from beta0.
    """
    beta = beta0
    F = beta.shape[0]
    eye = jnp.eye(F)
    for _ in range(n_newton):
        grad = jnp.zeros(F)
        fisher = jnp.zeros((F, F))
        for X, gamma0, j_mask in group_data:
            g, fi = _logit_grad_fisher(X, gamma0, j_mask, beta)
            grad = grad + g
            fisher = fisher + fi
        step = jnp.linalg.solve(fisher + ridge * eye, grad)
        beta = beta + step
        if float(jnp.max(jnp.abs(step))) < tol:
            break
    return beta


@partial(jax.jit, static_argnames=("orientation", "conc", "prior_mode", "hier_rho"))
def group_estep(params, O, B, D, t_mask, j_mask, log_pi, X, blocker_id=None,
                orientation="current", conc=2.0, prior_mode="free", hier_rho=False):
    """E-step + M-step accumulators for one k-group (k, T, J derived from shapes)

    prior_mode: "free" uses the passed log_pi (re-estimated to gamma[:,0] in em_step);
    "logit" computes log_pi = log_softmax_k(X . beta) from the matchup-prior features.
    hier_rho: when True, the transition is per-blocker (rho indexed by blocker_id into
    params.rho_player) and the returned rho stats are per-(possession, blocker) counts
    `diag_pj, switch_pj`; when False, a single shared rho and pooled scalar counts.
    """
    k = O.shape[2]
    if prior_mode == "logit":
        log_pi = log_pi_from_features(X, params.beta)

    log_emis = emission_logprob(params.tau, params.sigma, O, B, D, orientation, conc)  # (P,T,J,k)
    if hier_rho:
        rho_pj = params.rho_player[jnp.clip(blocker_id, 0, None)]  # (P,J); pad idx clipped, masked below
        log_T = build_log_transition_batched(rho_pj, k)  # (P,J,k,k)
        gamma, xi_j, ll_j = forward_backward_group_perT(log_emis, log_pi, log_T, t_mask)
    else:
        log_T = build_log_transition(params.rho, k)
        gamma, xi_j, ll_j = forward_backward_group(log_emis, log_pi, log_T, t_mask)

    # snap-frame responsibilities (t=0 always valid) -> "free"-mode prior update and
    # "logit"-mode soft targets for the beta M-step
    gamma0 = gamma[:, 0, :, :]  # (P, J, k)

    # zero out padded defenders for the accumulators
    g = gamma * j_mask[:, None, :, None]  # (P,T,J,k)
    xi = xi_j * j_mask[:, :, None, None, None]  # (P,J,T-1,k,k)
    loglik = jnp.sum(ll_j * j_mask)

    O2 = O[..., 0:2]
    B2 = B[..., 0:2]
    D2 = D[..., 0:2]

    # tau accumulators: rows are [rusher_coord, qb_coord], response blocker_coord
    S_OO = jnp.einsum("ptjk,ptkc,ptkc->", g, O2, O2)
    S_OB = jnp.einsum("ptjk,ptkc,ptc->", g, O2, B2)
    S_BB = jnp.einsum("ptjk,ptc,ptc->", g, B2, B2)
    y_O = jnp.einsum("ptjk,ptkc,ptjc->", g, O2, D2)
    y_B = jnp.einsum("ptjk,ptc,ptjc->", g, B2, D2)
    XtWX = jnp.stack([jnp.stack([S_OO, S_OB]), jnp.stack([S_OB, S_BB])])
    XtWy = jnp.stack([y_O, y_B])

    # sigma sufficient statistic: y'Wy. The full weighted SSE is recombined with
    # XtWX/XtWy at the UPDATED tau in em_step (numpy's maximize_sigma uses tau_new,
    # not the E-step's old tau): SSE = y'Wy - 2 tau.XtWy + tau'XtWX tau. The
    # responsibility mass (factor 2 from x,y) is the denominator.
    yty = jnp.einsum("ptjk,ptjc,ptjc->", g, D2, D2)
    sden = 2.0 * g.sum()

    # rho accumulators: stay = sum of xi diagonal, switch = off-diagonal.
    if hier_rho:
        # per-(possession, blocker) counts for the Beta-Binomial per-player M-step
        diag_pj = jnp.einsum("pjtaa->pj", xi)  # (P,J) stay mass per blocker
        total_pj = jnp.einsum("pjtab->pj", xi)  # (P,J) all transitions per blocker
        switch_pj = total_pj - diag_pj
        return XtWX, XtWy, yty, sden, diag_pj, switch_pj, loglik, gamma0

    # raw (MLE) pools every frame-transition; play-normalized weights each play 1/t_i
    # (and switches by 1/(k-1)), per paper.tex eq 557.
    diag_p = jnp.einsum("pjtaa->p", xi)  # (P,) stay mass per possession
    total_p = jnp.einsum("pjtab->p", xi)  # (P,) all transition mass per possession
    switch_p = total_p - diag_p
    stay = diag_p.sum()
    switch = switch_p.sum()
    t_i = jnp.maximum(t_mask.sum(axis=1), 1.0)  # frames per possession
    stay_pn = (diag_p / t_i).sum()
    switch_pn = (switch_p / (t_i * (k - 1))).sum()

    return XtWX, XtWy, yty, sden, stay, switch, stay_pn, switch_pn, loglik, gamma0


def em_step(params: Params, groups: Dict[int, KGroup], orientation="current",
            rho_mode="mle", conc=2.0, prior_mode="free", beta_ridge=1e-3, n_newton=10):
    """one EM iteration over all k-groups -> (new_params, new_groups, total_loglik)

    orientation: emission orientation mode ("none"|"current"|"paper"|"vonmises").
    conc: concentration hyperparameter for paper (alpha) / vonmises (kappa).
    rho_mode: "mle" (pool every transition) or "play_normalized" (each play weighted 1/t_i).
    prior_mode: "free" re-estimates log_pi to the snap responsibilities per play (default,
      backward-compatible); "logit" fits the shared conditional-logit beta to those
      responsibilities (Newton, warm-started) and derives log_pi = softmax(X.beta).
    """
    hier = rho_mode == "hierarchical"
    XtWX = jnp.zeros((2, 2))
    XtWy = jnp.zeros(2)
    yty = 0.0
    sden = 0.0
    stay = switch = stay_pn = switch_pn = 0.0
    total_ll = 0.0
    new_groups = {}
    beta_data = []
    if hier:
        n_players = params.rho_player.shape[0]
        S_player = jnp.zeros(n_players)
        W_player = jnp.zeros(n_players)
    for k, G in groups.items():
        if hier:
            (dXtWX, dXtWy, dyty, dsden, diag_pj, switch_pj, dll, gamma0) = group_estep(
                params, G.O, G.B, G.D, G.t_mask, G.j_mask, G.log_pi, G.X, G.blocker_id,
                orientation, conc, prior_mode, hier_rho=True,
            )
            idx = jnp.clip(G.blocker_id, 0, None)  # pad -1 -> 0, zeroed by `valid`
            valid = G.blocker_id >= 0
            S_player = S_player.at[idx].add(jnp.where(valid, diag_pj, 0.0))
            W_player = W_player.at[idx].add(jnp.where(valid, switch_pj, 0.0))
        else:
            (dXtWX, dXtWy, dyty, dsden, dstay, dswitch, dstay_pn, dswitch_pn, dll,
             gamma0) = group_estep(
                params, G.O, G.B, G.D, G.t_mask, G.j_mask, G.log_pi, G.X, None,
                orientation, conc, prior_mode,
            )
            stay = stay + dstay
            switch = switch + dswitch
            stay_pn = stay_pn + dstay_pn
            switch_pn = switch_pn + dswitch_pn
        XtWX = XtWX + dXtWX
        XtWy = XtWy + dXtWy
        yty = yty + dyty
        sden = sden + dsden
        total_ll = total_ll + dll
        if prior_mode == "logit":
            beta_data.append((G.X, gamma0, G.j_mask))
            new_groups[k] = G  # log_pi is derived from X.beta, not stored
        else:
            new_groups[k] = G._replace(log_pi=jnp.log(gamma0))

    tau = simplex_project(XtWX, XtWy)
    # weighted SSE at the UPDATED tau, from sufficient stats (matches maximize_sigma)
    sigma = (yty - 2.0 * (tau @ XtWy) + tau @ (XtWX @ tau)) / sden

    rho_player = ab = None
    if hier:
        # Beta-Binomial empirical Bayes: refit (a,b) to the per-player counts, then the
        # posterior-mean per-player stickiness (shrinks low-snap players to a/(a+b)).
        a, b = betabinom_eb(S_player, W_player, a0=params.ab[0], b0=params.ab[1])
        rho_player = rho_posterior_mean(S_player, W_player, a, b)
        ab = jnp.array([a, b])
        rho = a / (a + b)
    elif rho_mode == "play_normalized":
        rho = (stay_pn / switch_pn) / (1.0 + stay_pn / switch_pn)
    else:
        rho = (stay / switch) / (1.0 + stay / switch)

    beta = params.beta
    if prior_mode == "logit":
        beta = fit_beta(beta_data, params.beta, n_newton=n_newton, ridge=beta_ridge)

    return Params(tau, sigma, rho, beta, rho_player, ab), new_groups, float(total_ll)


def run_em(params: Params, groups: Dict[int, KGroup], n_iter: int,
           orientation="current", rho_mode="mle", conc=2.0, prior_mode="free",
           beta_ridge=1e-3, n_newton=10):
    """run n_iter EM iterations; returns (params, groups, per-iteration loglik list)

    loglik[i] is the data log-likelihood at the params used for iteration i's update
    (computed before the update), matching the numpy reference loop. Note loglik is
    only comparable ACROSS runs that share the same orientation mode + conc (different
    emissions change the likelihood scale). prior_mode "free" (default) reproduces the
    original behavior; "logit" fits the conditional-logit matchup prior jointly.
    """
    lls = []
    for _ in range(n_iter):
        params, groups, ll = em_step(
            params, groups, orientation, rho_mode, conc, prior_mode, beta_ridge, n_newton
        )
        lls.append(ll)
    return params, groups, lls


# --------------------------------------------------------------------------- #
# data prep: list of (B, O, D) per possession -> padded, k-grouped jax arrays
# --------------------------------------------------------------------------- #
def group_and_pad(
    B_list: List[np.ndarray], O_list: List[np.ndarray], D_list: List[np.ndarray],
    X_list: List[np.ndarray] = None, bid_list: List[np.ndarray] = None,
) -> Dict[int, KGroup]:
    """bucket possessions by k (rushers), filter k==1, right-pad t and j with masks

    X_list (optional): per-possession (j, k, F) pre-snap prior features, aligned to the
    voxel axes. When omitted, a dummy F=1 feature tensor is stored (used only by the
    "logit" prior mode; "free" mode ignores it).
    bid_list (optional): per-possession (j,) contiguous player indices in voxel order,
    used by the hierarchical per-blocker rho path; padded blocker slots get -1.
    """
    use_X = X_list is not None
    use_bid = bid_list is not None
    buckets = defaultdict(list)
    for idx, (B, O, D) in enumerate(zip(B_list, O_list, D_list)):
        k = O.shape[1]
        if k == 1:  # transition (1-rho)/(k-1) is undefined; numpy main loop drops these
            continue
        Xp = np.asarray(X_list[idx]) if use_X else None
        bp = np.asarray(bid_list[idx]) if use_bid else None
        buckets[k].append((np.asarray(B), np.asarray(O), np.asarray(D), Xp, bp))

    groups = {}
    for k, items in buckets.items():
        P = len(items)
        T_max = max(B.shape[0] for B, _, _, _, _ in items)
        J_max = max(D.shape[1] for _, _, D, _, _ in items)
        C_o = items[0][1].shape[-1]
        C_d = items[0][2].shape[-1]
        F = items[0][3].shape[-1] if use_X else 1

        O_arr = np.zeros((P, T_max, k, C_o))
        B_arr = np.zeros((P, T_max, 2))
        D_arr = np.zeros((P, T_max, J_max, C_d))
        t_mask = np.zeros((P, T_max))
        j_mask = np.zeros((P, J_max))
        X_arr = np.zeros((P, J_max, k, F))
        bid_arr = np.full((P, J_max), -1, dtype=np.int32)
        for p, (B, O, D, Xp, bp) in enumerate(items):
            t = B.shape[0]
            j = D.shape[1]
            B_arr[p, :t] = B[:, :2]
            O_arr[p, :t] = O
            D_arr[p, :t, :j] = D
            t_mask[p, :t] = 1.0
            j_mask[p, :j] = 1.0
            if use_X:
                X_arr[p, :j] = Xp  # (j, k, F)
            if use_bid:
                bid_arr[p, :j] = bp
        log_pi = np.log(np.full((P, J_max, k), 1.0 / k))

        groups[k] = KGroup(
            jnp.asarray(O_arr),
            jnp.asarray(B_arr),
            jnp.asarray(D_arr),
            jnp.asarray(t_mask),
            jnp.asarray(j_mask),
            jnp.asarray(log_pi),
            jnp.asarray(X_arr),
            jnp.asarray(bid_arr),
        )
    return groups


def init_params(tau, sigma, rho, beta=None, rho_player=None, ab=None) -> Params:
    return Params(
        jnp.asarray(np.asarray(tau, dtype=float).reshape(-1)[:2]),
        jnp.asarray(float(sigma)),
        jnp.asarray(float(rho)),
        None if beta is None else jnp.asarray(np.asarray(beta, dtype=float).reshape(-1)),
        None if rho_player is None else jnp.asarray(np.asarray(rho_player, dtype=float).reshape(-1)),
        None if ab is None else jnp.asarray(np.asarray(ab, dtype=float).reshape(-1)),
    )


# --------------------------------------------------------------------------- #
# real-data driver (mirrors baum_welch.py __main__)
# --------------------------------------------------------------------------- #
def load_position_voxels(csv_paths=None, positions=("T", "C", "RB", "TE", "WR", "FB", "G"),
                         with_features=False):
    """pool (B, O, D[, X]) per blocker position across week file(s); load each once

    Returns {position: (B_list, O_list, D_list, X_list)} (X_list empty unless
    with_features). possession_id resets per week, so each file is grouped independently
    and its possessions appended. k==1 possessions are dropped. When with_features, the
    pre-snap prior features X (axis-aligned to the voxels) are built per possession.
    """
    import pandas as pd

    from data_processing import possession_to_voxel, possession_to_prior_features

    if csv_paths is None:
        csv_paths = [f"data/sample_data_week_{i}.csv" for i in range(8)]
    elif isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    plays_df = pff_df = None
    if with_features:
        plays_df = pd.read_csv("data/plays.csv")
        pff_df = pd.read_csv("data/pffScoutingData.csv")

    voxels = {p: ([], [], [], [], []) for p in positions}  # B, O, D, X, blocker_ids
    for path in csv_paths:
        print(f"loading {path}", flush=True)
        data = pd.read_csv(path)
        for position in positions:
            pos_data = data[data["officialPosition_pb"] == position]
            if pos_data.empty:
                continue
            B_list, O_list, D_list, X_list, BID_list = voxels[position]
            for _, poss in pos_data.groupby("possession_id"):
                B, O, D = possession_to_voxel(poss)
                if O.shape[1] == 1:
                    continue
                B_list.append(B)
                O_list.append(O)
                D_list.append(D)
                # blocker nflIds in voxel (j-axis) order == sorted unique nflId
                BID_list.append(np.sort(poss["nflId"].unique()))
                if with_features:
                    X, _, _ = possession_to_prior_features(poss, plays_df, pff_df)
                    X_list.append(X)
    return voxels


def standardize_features(X_list):
    """z-score each feature column over all (possession, j, k) rows; returns (X_std, mean, std)"""
    flat = np.concatenate([np.asarray(X).reshape(-1, X.shape[-1]) for X in X_list], axis=0)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    X_std = [(np.asarray(X) - mean) / std for X in X_list]
    return X_std, mean, std


def fit_by_position(
    csv_paths=None,
    positions=("T", "C", "RB", "TE", "WR", "FB", "G"),
    n_iter: int = 15,
    tau0=(0.8, 0.2),
    sigma0: float = 2.0,
    rho0: float = 0.95,
    orientation: str = "current",
    rho_mode: str = "mle",
    conc: float = 2.0,
    prior: str = "free",
    beta_ridge: float = 1e-2,
    out_path: str = "fitted_params_jax.csv",
    rho_by_player_path: str = "rho_by_player.csv",
):
    """fit tau/sigma/rho (+ matchup-prior beta) per blocker position from sample-data csv(s)

    prior: "free" (uniform-equivalent, default), "logit" (joint-EM conditional-logit
    matchup prior), or "two_stage" (free fit -> fit beta to the snap responsibilities ->
    joint logit fit warm-started from that beta).
    rho_mode: "mle" / "play_normalized" (single pooled rho per position) or
    "hierarchical" (per-player rho_j with a Beta-Binomial empirical-Bayes prior; writes
    the per-player stickiness to rho_by_player_path and the per-position Beta (a,b) to
    the rho_ab column of out_path).
    """
    import pandas as pd

    with_features = prior in ("logit", "two_stage")
    voxels = load_position_voxels(csv_paths, positions, with_features=with_features)

    hier = rho_mode == "hierarchical"
    param_list = []
    player_rows = []
    for position in positions:
        B_list, O_list, D_list, X_list, BID_list = voxels[position]
        n_poss = len(B_list)
        print(f"fitting {position} on {n_poss} pooled possessions "
              f"(orientation={orientation}, conc={conc}, rho={rho_mode}, prior={prior})",
              flush=True)
        if not B_list:
            print(f"  no usable possessions for {position}; skipping")
            continue

        # hierarchical: per-position contiguous player index + initial rho_player/(a,b)
        bid_contig = player_ids = None
        rp_kwargs = {}
        if hier:
            player_ids = np.unique(np.concatenate([np.asarray(b) for b in BID_list]))
            id2idx = {int(pid): i for i, pid in enumerate(player_ids)}
            bid_contig = [np.array([id2idx[int(p)] for p in b], dtype=np.int32) for b in BID_list]
            C0 = 10.0  # initial Beta concentration around rho0
            rp_kwargs = dict(rho_player=np.full(len(player_ids), rho0),
                             ab=np.array([rho0 * C0, (1.0 - rho0) * C0]))

        beta_out = feat_mean = feat_std = None
        if with_features:
            X_std, feat_mean, feat_std = standardize_features(X_list)
            F = X_std[0].shape[-1]
            groups = group_and_pad(B_list, O_list, D_list, X_std, bid_contig)
            if prior == "two_stage":
                # stage 1: free fit -> log_pi holds the snap responsibilities
                p_free, g_free, _ = run_em(
                    init_params(tau0, sigma0, rho0, beta=np.zeros(F), **rp_kwargs), groups,
                    n_iter, orientation, rho_mode, conc, prior_mode="free")
                beta0 = fit_beta(
                    [(G.X, jnp.exp(G.log_pi), G.j_mask) for G in g_free.values()],
                    jnp.zeros(F), n_newton=30, ridge=beta_ridge)
                params = init_params(tau0, sigma0, rho0, beta=np.asarray(beta0), **rp_kwargs)
            else:
                params = init_params(tau0, sigma0, rho0, beta=np.zeros(F), **rp_kwargs)
            params, groups, lls = run_em(
                params, groups, n_iter, orientation, rho_mode, conc,
                prior_mode="logit", beta_ridge=beta_ridge)
            beta_out = np.asarray(params.beta)
        else:
            groups = group_and_pad(B_list, O_list, D_list, None, bid_contig)
            params = init_params(tau0, sigma0, rho0, **rp_kwargs)
            params, groups, lls = run_em(
                params, groups, n_iter, orientation, rho_mode, conc)

        ab_out = None
        if hier:
            ab_out = (float(params.ab[0]), float(params.ab[1]))
            # one extra E-step pass to recover per-player transition counts for the sidecar
            prior_mode = "logit" if with_features else "free"
            S = np.zeros(len(player_ids))
            W = np.zeros(len(player_ids))
            for kk, G in groups.items():
                out = group_estep(params, G.O, G.B, G.D, G.t_mask, G.j_mask, G.log_pi,
                                  G.X, G.blocker_id, orientation, conc, prior_mode,
                                  hier_rho=True)
                bid = np.asarray(G.blocker_id)
                m = bid >= 0
                np.add.at(S, bid[m], np.asarray(out[4])[m])
                np.add.at(W, bid[m], np.asarray(out[5])[m])
            player_rows.append(pd.DataFrame({
                "nflId": player_ids, "position": position,
                "rho": np.asarray(params.rho_player), "n_transitions": S + W,
            }))

        data_dict = {
            "tau": np.asarray(params.tau),
            "sigma": float(params.sigma),
            "rho": float(params.rho),
            "position": position,
            "n_possessions": n_poss,
            "final_loglik": lls[-1],
            "beta": beta_out,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "rho_ab": None if ab_out is None else np.array(ab_out),
        }
        print(data_dict)
        param_list.append(data_dict)
        jax.clear_caches()  # bound the XLA compile cache to one position's shapes

    pd.DataFrame(param_list).to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    if hier and player_rows:
        pd.concat(player_rows, ignore_index=True).to_csv(rho_by_player_path, index=False)
        print(f"wrote {rho_by_player_path} ({sum(len(r) for r in player_rows)} players)")
    return param_list


def _parse_vec(s):
    """parse a numpy-array-repr CSV cell -> 1-D float array"""
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=float)
    return np.fromstring(str(s).replace("\n", " ").strip().strip("[]"), sep=" ")


def _log_pi_null_from_features(X, beta, null_init):
    """(P,J,K,F) standardized X + beta -> (P,J,K+1) log initial dist with a null slot.

    Rushers share log_softmax(X.beta) scaled by (1-null_init); the null state gets null_init.
    """
    base = log_pi_from_features(X, beta)  # (P,J,K) log-softmax over rushers
    rush = jnp.log1p(-null_init) + base
    nullc = jnp.full(base.shape[:-1] + (1,), jnp.log(null_init))
    return jnp.concatenate([rush, nullc], axis=-1)


def fit_null_transition_by_position(
    csv_paths=None,
    positions=("T", "C", "RB", "TE", "WR", "FB", "G"),
    fitted_path="fitted_params_jax_phase1.csv",
    rho_by_player_path="rho_by_player_phase1.csv",
    c_b=-5.0, conc=1.0, orientation="vonmises", null_init=0.05, n_iter=15,
    out_path="fitted_params_jax_phase25.csv",
    out_player_path="rho_by_player_phase25.csv",
):
    """Phase 2.5: fit the structured null transition {rho_stay, p_fail (per-player), rho_null
    (per-position)} by coordinate ascent, HOLDING the Phase-1 emission (tau, sigma, beta) and
    its feature standardization FIXED. Each iteration runs the (K+1)-state forward-backward
    with the null emission, accumulates directional xi counts, and closed-form-updates the
    transition: Dirichlet-EB over {same, switch, fail} -> per-player (rho, p_fail); pooled
    null->null fraction -> per-position rho_null. Writes Phase-2.5 params to new paths."""
    import pandas as pd

    fitted = pd.read_csv(fitted_path).set_index("position")
    rdf = pd.read_csv(rho_by_player_path)
    rho_init = dict(zip(rdf["nflId"].astype(int), rdf["rho"].astype(float)))
    voxels = load_position_voxels(csv_paths, positions, with_features=True)

    param_rows, player_rows = [], []
    for position in positions:
        B_list, O_list, D_list, X_list, BID_list = voxels[position]
        if not B_list:
            print(f"  no possessions for {position}; skipping", flush=True)
            continue
        row = fitted.loc[position]
        tau = jnp.asarray(_parse_vec(row["tau"]))
        sigma = float(row["sigma"])
        beta = jnp.asarray(_parse_vec(row["beta"]))
        fmean, fstd = _parse_vec(row["feat_mean"]), _parse_vec(row["feat_std"])
        rho0 = float(row["rho"])

        X_std = [(np.asarray(x) - fmean) / fstd for x in X_list]
        player_ids = np.unique(np.concatenate([np.asarray(b) for b in BID_list]))
        id2idx = {int(p): i for i, p in enumerate(player_ids)}
        bid_contig = [np.array([id2idx[int(p)] for p in b], np.int32) for b in BID_list]
        groups = group_and_pad(B_list, O_list, D_list, X_std, bid_contig)
        n = len(player_ids)

        rho_player = jnp.asarray(np.clip(
            [rho_init.get(int(p), rho0) for p in player_ids], 1e-3, 1 - 1e-3))
        pfail_player = jnp.full(n, 0.01)
        rho_null = 0.99
        alpha = jnp.array([rho0 * 20.0, (1 - rho0) * 18.0, (1 - rho0) * 2.0])  # (same, switch, fail)
        counts = jnp.zeros((n, 3))
        for it in range(n_iter):
            A = jnp.zeros(n); Bc = jnp.zeros(n); F = jnp.zeros(n)
            Nstay = Nrec = ll = 0.0
            for K, G in groups.items():
                idx = jnp.clip(G.blocker_id, 0, None)
                rho_pj, pf_pj = rho_player[idx], pfail_player[idx]
                log_emis = emission_logprob(tau, sigma, G.O, G.B, G.D, orientation, conc,
                                            null_state=True, c_b=c_b)  # (P,T,J,K+1)
                log_pi = _log_pi_null_from_features(G.X, beta, null_init)  # (P,J,K+1)
                log_T = build_log_transition_null_batched(rho_pj, pf_pj, rho_null, K)
                gamma, xi, ll_j = forward_backward_group_perT(log_emis, log_pi, log_T, G.t_mask)
                xi = xi * G.j_mask[:, :, None, None, None]  # (P,J,T-1,K+1,K+1) a=t+1,b=t
                A_pj = jnp.einsum("pjtii->pj", xi[:, :, :, :K, :K])
                leave_pj = jnp.einsum("pjtab->pj", xi[:, :, :, :, :K])
                F_pj = jnp.einsum("pjtb->pj", xi[:, :, :, K, :K])
                B_pj = leave_pj - A_pj - F_pj
                Nstay = Nstay + jnp.sum(xi[:, :, :, K, K])
                Nrec = Nrec + jnp.sum(xi[:, :, :, :K, K])
                valid = G.blocker_id >= 0
                A = A.at[idx].add(jnp.where(valid, A_pj, 0.0))
                Bc = Bc.at[idx].add(jnp.where(valid, B_pj, 0.0))
                F = F.at[idx].add(jnp.where(valid, F_pj, 0.0))
                ll = ll + jnp.sum(ll_j * G.j_mask)
            counts = jnp.clip(jnp.stack([A, Bc, F], axis=1), 0.0, None)
            alpha = dirichlet_eb(counts, alpha0=alpha, n_steps=50)
            pm = dirichlet_posterior_mean(counts, alpha)  # (n,3)
            rho_player, pfail_player = pm[:, 0], pm[:, 2]
            rho_null = float((Nstay + 1.0) / (Nstay + Nrec + 2.0))
        print(f"{position}: ll={float(ll):.0f} alpha={np.round(np.asarray(alpha),2)} "
              f"rho_null={rho_null:.4f} mean p_fail={float(pfail_player.mean()):.4f} "
              f"mean rho={float(rho_player.mean()):.4f}", flush=True)
        ntr = np.asarray(counts.sum(1))
        for i, p in enumerate(player_ids):
            player_rows.append({"nflId": int(p), "position": position,
                                "rho": float(rho_player[i]), "p_fail": float(pfail_player[i]),
                                "n_transitions": float(ntr[i])})
        d = row.to_dict(); d["position"] = position
        d["rho_null"] = rho_null; d["dir_alpha"] = np.asarray(alpha); d["c_b"] = c_b
        param_rows.append(d)
        jax.clear_caches()

    pd.DataFrame(param_rows).to_csv(out_path, index=False)
    pd.DataFrame(player_rows).to_csv(out_player_path, index=False)
    print(f"wrote {out_path} and {out_player_path} ({len(player_rows)} players)", flush=True)
    return param_rows


if __name__ == "__main__":
    fit_by_position()
