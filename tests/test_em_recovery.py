"""Synthetic ground-truth validation for the defensive-matchup EM.

Simulates possessions from the exact generative model (known tau, sigma, rho and
known blocker->rusher assignment Markov chains), then checks that the fixed EM
recovers the parameters and that the incomplete-data log-likelihood increases
monotonically. Also contains focused unit tests for the individual M-step / E-step
fixes.

Runs without pytest:  python tests/test_em_recovery.py
(or, if pytest is installed:  python -m pytest tests/test_em_recovery.py -v)
"""

import os
import sys

# resolve the src/ modules regardless of the current working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from baum_welch import (
    build_transition_matrix,
    calculate_lambda,
    calculate_eta,
    data_log_likelihood,
    expectation_maximization,
    forward_procedure,
    backward_procedure,
    maximize_rho,
    maximize_tau,
)
from data_processing import voxels_to_design_response


# --------------------------------------------------------------------------- #
# generative simulator
# --------------------------------------------------------------------------- #
def simulate_possession(tau, sigma, rho, k, j, t, rng, with_orientation=False):
    """Simulate one possession from the generative model.

    Args:
        tau: (tau_rusher, tau_qb) convex weights (sum to 1)
        sigma: position variance (NOT std)
        rho: stickiness (stay probability)
        k: number of rushers, j: number of blockers, t: number of frames
        rng: numpy Generator
        with_orientation: if True, append a `dir` column to O and D so the Beta
            orientation emission term is active

    Returns:
        B (t, 2), O (t, k, .), D (t, j, .), Z (t, j) ground-truth assignments
    """
    tau_o, tau_b = float(tau[0]), float(tau[1])
    std = np.sqrt(sigma)

    B = np.cumsum(rng.normal(0.0, 0.3, size=(t, 2)), axis=0) + np.array([30.0, 26.0])

    O_xy = np.zeros((t, k, 2))
    for m in range(k):
        start = np.array([24.0 + 1.5 * m, 18.0 + 4.0 * m])
        O_xy[:, m, :] = np.cumsum(rng.normal(0.0, 0.4, size=(t, 2)), axis=0) + start

    T = build_transition_matrix(rho, k)

    D_xy = np.zeros((t, j, 2))
    Z = np.zeros((t, j), dtype=int)
    for d in range(j):
        z = int(rng.integers(0, k))
        for ti in range(t):
            Z[ti, d] = z
            mean = tau_o * O_xy[ti, z, :] + tau_b * B[ti, :]
            D_xy[ti, d, :] = mean + rng.normal(0.0, std, size=2)
            z = int(rng.choice(k, p=T[z, :]))

    if not with_orientation:
        return B, O_xy, D_xy, Z

    # orientation: rusher dir random in [0, 360); blocker faces opposite-ish so the
    # cos() emission is informative. (Only used to exercise the Beta term.)
    O_dir = rng.uniform(0.0, 360.0, size=(t, k, 1))
    O = np.concatenate([O_xy, O_dir], axis=-1)
    D_dir = np.zeros((t, j, 1))
    for d in range(j):
        for ti in range(t):
            D_dir[ti, d, 0] = (O_dir[ti, Z[ti, d], 0] + 180.0) % 360.0
    D = np.concatenate([D_xy, D_dir], axis=-1)
    return B, O, D, Z


def _run_em(B_list, O_list, D_list, k, j, n_iter, tau0, sigma0, rho0):
    """Run the EM loop, returning (tau, sigma, rho, log_likelihoods)."""
    tau, sigma, rho = np.array(tau0, dtype=float).reshape((2, 1)), float(sigma0), float(rho0)
    ssd = [np.ones((j, k)) / k for _ in range(len(B_list))]
    lls = []
    for _ in range(n_iter):
        lls.append(data_log_likelihood(tau, sigma, rho, ssd, D_list, O_list, B_list))
        tau, sigma, rho, ssd = expectation_maximization(
            tau,
            sigma_init=sigma,
            rho_init=rho,
            forward_result=ssd,
            B=B_list,
            D=D_list,
            O=O_list,
        )
    return tau, sigma, rho, lls


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{(' - ' + detail) if detail else ''}")
    if not cond:
        raise AssertionError(f"{name} {detail}")


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_em_recovers_parameters():
    print("test_em_recovers_parameters (orientation disabled)")
    rng = np.random.default_rng(0)
    true_tau, true_sigma, true_rho = (0.7, 0.3), 0.5, 0.9
    k, j, t, n_poss = 4, 4, 60, 20

    B_list, O_list, D_list = [], [], []
    for _ in range(n_poss):
        B, O, D, _ = simulate_possession(true_tau, true_sigma, true_rho, k, j, t, rng)
        B_list.append(B)
        O_list.append(O)
        D_list.append(D)

    tau, sigma, rho, lls = _run_em(
        B_list, O_list, D_list, k, j, n_iter=40, tau0=(0.5, 0.5), sigma0=1.0, rho0=0.8
    )

    print(f"    recovered tau={tau.ravel()}, sigma={sigma:.4f}, rho={rho:.4f}")
    _check("tau_rusher within 0.05", abs(tau[0, 0] - 0.7) < 0.05, f"got {tau[0,0]:.4f}")
    _check("tau_qb within 0.05", abs(tau[1, 0] - 0.3) < 0.05, f"got {tau[1,0]:.4f}")
    _check("tau on simplex", abs(tau.sum() - 1.0) < 1e-8)
    _check("sigma within 15%", abs(sigma - true_sigma) / true_sigma < 0.15, f"got {sigma:.4f}")
    _check("rho within 0.03", abs(rho - true_rho) < 0.03, f"got {rho:.4f}")

    violations = [(a, b) for a, b in zip(lls, lls[1:]) if b < a - 1e-6 * abs(a) - 1e-6]
    _check("log-likelihood monotone non-decreasing", not violations,
           f"{len(violations)} violations" if violations else "")


def test_em_recovers_with_orientation():
    print("test_em_recovers_with_orientation (Beta term active)")
    rng = np.random.default_rng(7)
    true_tau, true_sigma, true_rho = (0.7, 0.3), 0.5, 0.9
    k, j, t, n_poss = 4, 4, 60, 20

    B_list, O_list, D_list = [], [], []
    for _ in range(n_poss):
        B, O, D, _ = simulate_possession(
            true_tau, true_sigma, true_rho, k, j, t, rng, with_orientation=True
        )
        B_list.append(B)
        O_list.append(O)
        D_list.append(D)

    tau, sigma, rho, lls = _run_em(
        B_list, O_list, D_list, k, j, n_iter=40, tau0=(0.5, 0.5), sigma0=1.0, rho0=0.8
    )
    print(f"    recovered tau={tau.ravel()}, sigma={sigma:.4f}, rho={rho:.4f}")
    # orientation adds an emission factor independent of tau/sigma/rho, so the
    # location parameters should still be recovered.
    _check("tau_rusher within 0.07", abs(tau[0, 0] - 0.7) < 0.07, f"got {tau[0,0]:.4f}")
    _check("tau on simplex", abs(tau.sum() - 1.0) < 1e-8)
    _check("rho within 0.05", abs(rho - true_rho) < 0.05, f"got {rho:.4f}")
    violations = [(a, b) for a, b in zip(lls, lls[1:]) if b < a - 1e-6 * abs(a) - 1e-6]
    _check("log-likelihood monotone non-decreasing", not violations,
           f"{len(violations)} violations" if violations else "")


def test_maximize_tau_simplex():
    print("test_maximize_tau_simplex")
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(0.0, 1.0, size=(n, 2))
    true_tau = np.array([[0.7], [0.3]])

    # exact (noiseless) data -> GLS recovers tau exactly and it already sums to 1
    D = X @ true_tau
    tau = maximize_tau(X, np.ones(n), D)
    _check("recovers exact tau", np.allclose(tau, true_tau, atol=1e-8), f"got {tau.ravel()}")
    _check("sum to 1 (exact)", abs(tau.sum() - 1.0) < 1e-10)

    # noisy data + non-uniform weights -> still projected onto the simplex
    Dn = D + rng.normal(0.0, 0.5, size=(n, 1))
    w = rng.uniform(0.1, 1.0, size=n)
    tau_n = maximize_tau(X, w, Dn)
    _check("sum to 1 (noisy, weighted)", abs(tau_n.sum() - 1.0) < 1e-10, f"sum={float(tau_n.sum()):.12f}")


def test_calculate_eta_marginal_consistency():
    print("test_calculate_eta_marginal_consistency")
    rng = np.random.default_rng(2)
    t, k, rho = 30, 4, 0.85
    emission = rng.uniform(0.1, 1.0, size=(t, k))
    emission_norm = emission / emission.sum(axis=1, keepdims=True)
    prior = np.ones(k) / k
    T = build_transition_matrix(rho, k)

    fwd = forward_procedure(emission_norm, prior, T, t)
    bwd = backward_procedure(emission_norm, T, t)
    gamma = calculate_lambda(fwd, bwd)
    eta = calculate_eta(bwd, emission_norm, fwd, T)

    _check("each eta slice sums to 1", np.allclose(eta.sum(axis=(1, 2)), 1.0))
    # summing over the t+1 axis (axis 1) recovers gamma at time t
    _check("eta marginal == gamma[:-1]", np.allclose(eta.sum(axis=1), gamma[:-1], atol=1e-8))
    # summing over the t axis (axis 2) recovers gamma at time t+1
    _check("eta marginal == gamma[1:]", np.allclose(eta.sum(axis=2), gamma[1:], atol=1e-8))


def test_maximize_rho_counts():
    print("test_maximize_rho_counts")
    k = 4
    # known transition-count matrix: heavy diagonal -> high rho
    c = np.array(
        [
            [90.0, 2.0, 1.0, 1.0],
            [1.0, 88.0, 3.0, 2.0],
            [2.0, 1.0, 91.0, 1.0],
            [1.0, 2.0, 1.0, 89.0],
        ]
    )
    A = [c[np.newaxis, np.newaxis, :, :]]  # (1, 1, k, k)
    rho = maximize_rho(A)
    stay = np.trace(c)
    switch = c.sum() - stay
    expected = stay / (stay + switch)
    _check("rho == stay/(stay+switch)", abs(rho - expected) < 1e-12, f"got {rho:.6f} vs {expected:.6f}")


def test_design_response_weight_alignment():
    print("test_design_response_weight_alignment (j != k)")
    rng = np.random.default_rng(3)
    t, j, k = 5, 2, 3  # deliberately j != k
    D = rng.normal(size=(t, j, 2))
    O = rng.normal(size=(t, k, 2))
    B = rng.normal(size=(t, 2))
    I = rng.uniform(size=(t, j, k))  # responsibilities, (t, j, k)

    X, y, W = voxels_to_design_response(D, O, B, I)

    # rebuild the expected (X, y, weight) for every (t, k, j, coord) row in C-order
    # (coord fastest) and confirm each row matches.
    n = t * k * j * 2
    _check("row count", X.shape[0] == n and W.shape[0] == n, f"X={X.shape}, W={W.shape}")

    ok = True
    r = 0
    for tt in range(t):
        for kk in range(k):
            for jj in range(j):
                for coord in range(2):
                    ok &= np.isclose(X[r, 0], O[tt, kk, coord])      # rusher coord
                    ok &= np.isclose(X[r, 1], B[tt, coord])          # qb coord
                    ok &= np.isclose(y[r, 0], D[tt, jj, coord])      # blocker coord
                    ok &= np.isclose(W[r], I[tt, jj, kk])            # responsibility
                    r += 1
    _check("every row (X, y, weight) aligned", ok)


ALL_TESTS = [
    test_maximize_tau_simplex,
    test_calculate_eta_marginal_consistency,
    test_maximize_rho_counts,
    test_design_response_weight_alignment,
    test_em_recovers_parameters,
    test_em_recovers_with_orientation,
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
