"""Validation for the JAX Baum-Welch port (baum_welch_jax.py).

Two layers:
  * component equivalence (1e-6) vs the numpy reference -- localizes any bug to a
    single function (emission, forward-backward, simplex projection, M-step sums);
  * integration gates -- parameter recovery on synthetic data, monotone log-
    likelihood, and tight end-to-end equivalence to the numpy EM on identical data.

Reuses simulate_possession from test_em_recovery (the numpy harness).

Runs without pytest:  python tests/test_em_recovery_jax.py
"""

import os
import sys

# resolve the src/ modules (and the sibling test_em_recovery) regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # avoid the broken CUDA-plugin probe noise

import jax

jax.config.update("jax_enable_x64", True)  # match numpy float64 for tight equivalence

import jax.numpy as jnp
import numpy as np

import baum_welch as bw
import baum_welch_jax as bwj
from data_processing import voxels_to_design_response
from test_em_recovery import _check, _run_em, simulate_possession


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_possessions(seed, n_poss=20, k=4, j=4, t=60, with_orientation=False,
                      tau=(0.7, 0.3), sigma=0.5, rho=0.9):
    rng = np.random.default_rng(seed)
    B_list, O_list, D_list = [], [], []
    for _ in range(n_poss):
        B, O, D, _ = simulate_possession(
            tau, sigma, rho, k, j, t, rng, with_orientation=with_orientation
        )
        B_list.append(B)
        O_list.append(O)
        D_list.append(D)
    return B_list, O_list, D_list


def _run_em_jax(B_list, O_list, D_list, n_iter, tau0, sigma0, rho0):
    groups = bwj.group_and_pad(B_list, O_list, D_list)
    params = bwj.init_params(tau0, sigma0, rho0)
    params, groups, lls = bwj.run_em(params, groups, n_iter)
    return (params.tau, params.sigma, params.rho), lls


# --------------------------------------------------------------------------- #
# component equivalence
# --------------------------------------------------------------------------- #
def test_emission_equivalence():
    print("test_emission_equivalence (vs log(compute_emission))")
    for orient in (False, True):
        rng = np.random.default_rng(11)
        B, O, D, _ = simulate_possession((0.7, 0.3), 0.5, 0.9, 4, 4, 30, rng,
                                         with_orientation=orient)
        tau = np.array([[0.6], [0.4]])
        sigma = 0.7
        # numpy: per defender, (t, k); stack to (t, j, k)
        np_emis = np.stack(
            [bw.compute_emission(tau, sigma, D[:, m, :], O, B) for m in range(D.shape[1])],
            axis=1,
        )
        jx_log = bwj.emission_logprob(
            jnp.asarray(tau.reshape(-1)), jnp.asarray(sigma),
            jnp.asarray(O[None]), jnp.asarray(B[None]), jnp.asarray(D[None]),
        )[0]
        _check(f"emission matches (orientation={orient})",
               np.allclose(np.asarray(jx_log), np.log(np_emis), atol=1e-6),
               f"max abs diff {np.abs(np.asarray(jx_log) - np.log(np_emis)).max():.2e}")


def test_forward_backward_equivalence():
    print("test_forward_backward_equivalence (gamma & xi vs numpy)")
    rng = np.random.default_rng(12)
    B, O, D, _ = simulate_possession((0.7, 0.3), 0.5, 0.9, 4, 4, 40, rng)
    tau = np.array([[0.6], [0.4]])
    sigma = 0.7
    rho = 0.85
    k = O.shape[1]
    m = 0  # one defender

    # numpy reference
    emis = bw.compute_emission(tau, sigma, D[:, m, :], O, B)
    emis_n = emis / emis.sum(axis=1, keepdims=True)
    T = bw.build_transition_matrix(rho, k)
    prior = np.ones(k) / k
    fwd = bw.forward_procedure(emis_n, prior, T, emis.shape[0])
    bwd = bw.backward_procedure(emis_n, T, emis.shape[0])
    gamma_np = bw.calculate_lambda(fwd, bwd)
    xi_np = bw.calculate_eta(bwd, emis_n, fwd, T)
    ll_np = bw.sequence_log_likelihood(emis, prior, T)

    # jax
    log_emis = jnp.log(jnp.asarray(emis))
    gamma_jx, xi_jx, ll_jx = bwj.forward_backward_single(
        log_emis, jnp.log(jnp.asarray(prior)), bwj.build_log_transition(rho, k),
        jnp.ones(emis.shape[0]),
    )
    _check("gamma matches", np.allclose(np.asarray(gamma_jx), gamma_np, atol=1e-6),
           f"max diff {np.abs(np.asarray(gamma_jx) - gamma_np).max():.2e}")
    _check("xi matches", np.allclose(np.asarray(xi_jx), xi_np, atol=1e-6),
           f"max diff {np.abs(np.asarray(xi_jx) - xi_np).max():.2e}")
    _check("data loglik matches", abs(float(ll_jx) - ll_np) < 1e-6,
           f"jax {float(ll_jx):.6f} vs np {ll_np:.6f}")
    # marginal consistency (the xi index-convention gate)
    _check("xi.sum(axis=1) == gamma[:-1]",
           np.allclose(np.asarray(xi_jx).sum(axis=1), np.asarray(gamma_jx)[:-1], atol=1e-8))
    _check("xi.sum(axis=2) == gamma[1:]",
           np.allclose(np.asarray(xi_jx).sum(axis=2), np.asarray(gamma_jx)[1:], atol=1e-8))


def test_simplex_project_equivalence():
    print("test_simplex_project_equivalence (vs maximize_tau)")
    rng = np.random.default_rng(13)
    n = 200
    X = rng.normal(size=(n, 2))
    w = rng.uniform(0.1, 1.0, size=n)
    y = (X @ np.array([[0.7], [0.3]]) + rng.normal(0, 0.3, size=(n, 1)))
    tau_np = bw.maximize_tau(X, w, y).ravel()
    XtWX = (X.T * w) @ X
    XtWy = (X.T * w) @ y
    tau_jx = np.asarray(bwj.simplex_project(jnp.asarray(XtWX), jnp.asarray(XtWy.ravel())))
    _check("tau matches maximize_tau", np.allclose(tau_jx, tau_np, atol=1e-8),
           f"jax {tau_jx} vs np {tau_np}")


def test_mstep_accumulator_equivalence():
    print("test_mstep_accumulator_equivalence (tau & sigma einsums vs design matrix)")
    rng = np.random.default_rng(14)
    # one possession, j != k to exercise alignment
    t, j, k = 8, 3, 4
    B, O, D, _ = simulate_possession((0.7, 0.3), 0.5, 0.9, k, j, t, rng)
    I = rng.uniform(size=(t, j, k))  # arbitrary responsibilities (t, j, k)
    tau = np.array([[0.65], [0.35]])

    # numpy path: build design matrix and run the closed forms
    X, y, W = voxels_to_design_response(D, O, B, I)
    XtWX_np = (X.T * W) @ X
    XtWy_np = (X.T * W) @ y
    tau_np = bw.maximize_tau(X, W, y).ravel()
    sigma_np = float(np.squeeze(bw.maximize_sigma(tau, W, y, X)))

    # jax path: einsum accumulators with gamma = I
    O2, B2, D2 = O[:, :, 0:2], B[:, 0:2], D[:, :, 0:2]
    g = jnp.asarray(I)[None]  # (1,t,j,k)
    Oj = jnp.asarray(O2)[None]
    Bj = jnp.asarray(B2)[None]
    Dj = jnp.asarray(D2)[None]
    S_OO = jnp.einsum("ptjk,ptkc,ptkc->", g, Oj, Oj)
    S_OB = jnp.einsum("ptjk,ptkc,ptc->", g, Oj, Bj)
    S_BB = jnp.einsum("ptjk,ptc,ptc->", g, Bj, Bj)
    y_O = jnp.einsum("ptjk,ptkc,ptjc->", g, Oj, Dj)
    y_B = jnp.einsum("ptjk,ptc,ptjc->", g, Bj, Dj)
    XtWX_jx = np.array([[float(S_OO), float(S_OB)], [float(S_OB), float(S_BB)]])
    XtWy_jx = np.array([float(y_O), float(y_B)])

    _check("X'WX matches", np.allclose(XtWX_jx, XtWX_np, atol=1e-6),
           f"max diff {np.abs(XtWX_jx - XtWX_np).max():.2e}")
    _check("X'Wy matches", np.allclose(XtWy_jx, XtWy_np.ravel(), atol=1e-6))

    tau_jx = np.asarray(bwj.simplex_project(jnp.asarray(XtWX_jx), jnp.asarray(XtWy_jx)))
    _check("tau from accumulators matches", np.allclose(tau_jx, tau_np, atol=1e-6))

    # sigma einsum vs numpy
    mu = tau[0, 0] * O2 + tau[1, 0] * B2[:, None, :]  # (t,k,2)
    resid = D2[:, :, None, :] - mu[:, None, :, :]  # (t,j,k,2)
    snum = float(jnp.einsum("tjk,tjkc->", jnp.asarray(I), jnp.asarray(resid) ** 2))
    sden = 2.0 * float(I.sum())
    _check("sigma matches maximize_sigma", abs(snum / sden - sigma_np) < 1e-6,
           f"jax {snum/sden:.6f} vs np {sigma_np:.6f}")


# --------------------------------------------------------------------------- #
# integration
# --------------------------------------------------------------------------- #
def test_jax_recovers_parameters():
    print("test_jax_recovers_parameters")
    B_list, O_list, D_list = _make_possessions(seed=0)
    (tau, sigma, rho), lls = _run_em_jax(
        B_list, O_list, D_list, n_iter=40, tau0=(0.5, 0.5), sigma0=1.0, rho0=0.8
    )
    tau = np.asarray(tau)
    print(f"    recovered tau={tau}, sigma={float(sigma):.4f}, rho={float(rho):.4f}")
    _check("tau_rusher within 0.05", abs(tau[0] - 0.7) < 0.05, f"got {tau[0]:.4f}")
    _check("tau on simplex", abs(tau.sum() - 1.0) < 1e-8)
    _check("sigma within 15%", abs(float(sigma) - 0.5) / 0.5 < 0.15, f"got {float(sigma):.4f}")
    _check("rho within 0.03", abs(float(rho) - 0.9) < 0.03, f"got {float(rho):.4f}")
    viol = [(a, b) for a, b in zip(lls, lls[1:]) if b < a - 1e-6 * abs(a) - 1e-6]
    _check("loglik monotone", not viol, f"{len(viol)} violations")


def test_jax_recovers_with_orientation():
    print("test_jax_recovers_with_orientation")
    B_list, O_list, D_list = _make_possessions(seed=7, with_orientation=True)
    (tau, sigma, rho), lls = _run_em_jax(
        B_list, O_list, D_list, n_iter=40, tau0=(0.5, 0.5), sigma0=1.0, rho0=0.8
    )
    tau = np.asarray(tau)
    print(f"    recovered tau={tau}, sigma={float(sigma):.4f}, rho={float(rho):.4f}")
    _check("tau_rusher within 0.07", abs(tau[0] - 0.7) < 0.07, f"got {tau[0]:.4f}")
    _check("tau on simplex", abs(tau.sum() - 1.0) < 1e-8)
    _check("rho within 0.05", abs(float(rho) - 0.9) < 0.05, f"got {float(rho):.4f}")
    viol = [(a, b) for a, b in zip(lls, lls[1:]) if b < a - 1e-6 * abs(a) - 1e-6]
    _check("loglik monotone", not viol, f"{len(viol)} violations")


def test_jax_matches_numpy():
    print("test_jax_matches_numpy (end-to-end equivalence on identical data)")
    for orient in (False, True):
        B_list, O_list, D_list = _make_possessions(seed=3, n_poss=12, t=40,
                                                   with_orientation=orient)
        n_iter = 15
        tau_np, sigma_np, rho_np, lls_np = _run_em(
            B_list, O_list, D_list, k=4, j=4, n_iter=n_iter,
            tau0=(0.5, 0.5), sigma0=1.0, rho0=0.8,
        )
        (tau_jx, sigma_jx, rho_jx), lls_jx = _run_em_jax(
            B_list, O_list, D_list, n_iter=n_iter, tau0=(0.5, 0.5), sigma0=1.0, rho0=0.8
        )
        tau_jx = np.asarray(tau_jx)
        tag = f"(orientation={orient})"
        _check(f"tau matches numpy {tag}", np.allclose(tau_jx, tau_np.ravel(), atol=1e-6),
               f"jax {tau_jx} vs np {tau_np.ravel()}")
        _check(f"sigma matches numpy {tag}", abs(float(sigma_jx) - sigma_np) < 1e-6,
               f"jax {float(sigma_jx):.8f} vs np {sigma_np:.8f}")
        _check(f"rho matches numpy {tag}", abs(float(rho_jx) - rho_np) < 1e-6,
               f"jax {float(rho_jx):.8f} vs np {rho_np:.8f}")
        _check(f"loglik trace matches numpy {tag}",
               np.allclose(np.array(lls_jx), np.array(lls_np), atol=1e-5),
               f"max diff {np.abs(np.array(lls_jx) - np.array(lls_np)).max():.2e}")


def test_fit_beta_recovers():
    print("test_fit_beta_recovers (conditional-logit M-step)")
    rng = np.random.default_rng(20)
    P, J, k, F = 400, 4, 5, 6
    X = rng.normal(size=(P, J, k, F))  # features vary across k -> beta identified
    beta_true = rng.normal(size=F)
    s = np.einsum("pjkf,f->pjk", X, beta_true)
    pi = np.exp(s - s.max(-1, keepdims=True))
    pi /= pi.sum(-1, keepdims=True)  # exact soft targets from beta_true
    beta_hat = np.asarray(
        bwj.fit_beta(
            [(jnp.asarray(X), jnp.asarray(pi), jnp.ones((P, J)))],
            jnp.zeros(F), n_newton=30, ridge=1e-8,
        )
    )
    _check("beta recovered from soft labels", np.allclose(beta_hat, beta_true, atol=1e-3),
           f"max abs diff {np.abs(beta_hat - beta_true).max():.2e}")
    # log_pi_from_features normalizes over k
    lp = np.asarray(bwj.log_pi_from_features(jnp.asarray(X), jnp.asarray(beta_true)))
    _check("log_pi normalized over k", np.allclose(np.exp(lp).sum(-1), 1.0, atol=1e-10))


def test_logit_prior_em_integration():
    print("test_logit_prior_em_integration (logit prior runs, doesn't break recovery)")
    B_list, O_list, D_list = _make_possessions(seed=0)
    rng = np.random.default_rng(5)
    # arbitrary per-possession (j, k, F) features that vary across k
    X_list = [rng.normal(size=(D.shape[1], O.shape[1], 4)) for O, D in zip(O_list, D_list)]
    groups = bwj.group_and_pad(B_list, O_list, D_list, X_list)
    params = bwj.init_params((0.5, 0.5), 1.0, 0.8, beta=np.zeros(4))
    params, _, lls = bwj.run_em(params, groups, n_iter=30, prior_mode="logit")
    tau = np.asarray(params.tau)
    print(f"    tau={tau}, sigma={float(params.sigma):.4f}, rho={float(params.rho):.4f}, "
          f"|beta|={np.linalg.norm(np.asarray(params.beta)):.3f}")
    # location/transition params still recovered with the prior active
    _check("tau_rusher within 0.06", abs(tau[0] - 0.7) < 0.06, f"got {tau[0]:.4f}")
    _check("rho within 0.04", abs(float(params.rho) - 0.9) < 0.04, f"got {float(params.rho):.4f}")
    _check("beta moved off zero", np.linalg.norm(np.asarray(params.beta)) > 1e-3)
    _check("loglik finite & monotone",
           np.all(np.isfinite(lls)) and not [
               (a, b) for a, b in zip(lls, lls[1:]) if b < a - 1e-6 * abs(a) - 1e-6])


ALL_TESTS = [
    test_emission_equivalence,
    test_forward_backward_equivalence,
    test_simplex_project_equivalence,
    test_mstep_accumulator_equivalence,
    test_fit_beta_recovers,
    test_logit_prior_em_integration,
    test_jax_recovers_parameters,
    test_jax_recovers_with_orientation,
    test_jax_matches_numpy,
]


if __name__ == "__main__":
    failures = 0
    for test in ALL_TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  !! {exc}")
        print()
    if failures:
        print(f"{failures} test(s) FAILED")
        raise SystemExit(1)
    print("all tests PASSED")
