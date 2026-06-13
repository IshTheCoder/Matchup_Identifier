### functions for determining e-m algorithm for hidden markov model
from typing import List, Tuple

import numpy as np
from scipy.linalg import solve
from scipy.stats import norm, beta

from data_processing import voxels_to_design_response


def calculate_lambda(
    forward_result: np.ndarray, backward_result: np.ndarray
) -> np.ndarray:
    """_summary_

    Args:
        forward_result (np.ndarray): forward step result (t x k) matrix
        backward_result (np.ndarray): backward result matrix (t x k ) matrix

    Returns:
        Tuple[np.ndarray,np.ndarray]: tuple of state distributions of each timestep, and likelihood

    """

    lam = forward_result * backward_result  ### normalize row
    likelihood = lam.sum(axis=1, keepdims=True)  ### need this for later
    lam_normalized = lam / likelihood
    return lam_normalized


def calculate_eta(
    backward_result: np.ndarray,
    pdf: np.ndarray,
    forward_result: np.ndarray,
    transition_matrix: np.ndarray,
) -> np.ndarray:
    """calculate estimated transition prob

    Args:
        backward_result (np.ndarray): backward algorithm result
        pdf (np.ndarray): P(y | x)
        forward_result (np.ndarray): forward algorithm result
        transition_matrix (np.ndarray): current estimate of transition matrix

    Returns:
        np.ndarray: eta which is ( (t-1) x k x k) matrix
    """

    t, k = pdf.shape
    eta = np.zeros((t - 1, k, k))
    for i in range(t - 1):
        ### joint pairwise posterior xi_t(a, b) = P(x_t = a, x_{t+1} = b | y),
        ### proportional to alpha_t(a) * T(a -> b) * P(y_{t+1} | b) * beta_{t+1}(b).
        ### normalize each slice by a SINGLE scalar (joint over both axes) so that
        ### sum_b xi_t(a, b) == gamma_t(a); row-normalizing would instead give the
        ### conditional transition prob and break the rho M-step expected counts.
        num = transition_matrix * np.outer(
            backward_result[i + 1, :] * pdf[i + 1, :],
            forward_result[i, :],
        )
        eta[i, :, :] = num / num.sum()
    return eta


def compute_emission(
    tau_hat: np.ndarray,
    sigma_hat: float,
    D: np.ndarray,
    O: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """unnormalized HMM emission densities for a single defender

    The [t, m] entry is the density of the defender's observation at time t under
    the hypothesis that it is guarding rusher m: an isotropic 2-D Gaussian on
    position centered on the convex combination tau_o * rusher + tau_b * qb, times
    (when rusher orientation is available, i.e. O has a 3rd column) a Beta density
    on the blocker orientation.

    Args:
        tau_hat (np.ndarray): (2, 1) simplex vector (rusher weight, qb weight)
        sigma_hat (float): position variance
        D (np.ndarray): (t, >=2) one defender's positions (x, y[, dir])
        O (np.ndarray): (t, k, >=2) rusher positions (x, y[, dir])
        B (np.ndarray): (t, 2) quarterback positions (x, y)

    Returns:
        np.ndarray: (t, k) unnormalized emission densities
    """
    _, k, o_dim = O.shape
    D_rep = np.repeat(D[:, np.newaxis, :], k, axis=-2)
    B_rep = np.repeat(B[:, np.newaxis, :], k, axis=-2)

    O_B_stacked = np.stack(
        (O[:, :, 0:2], B_rep[:, :, 0:2]), axis=-1
    )  ## (t, k, 2, 2): for each rusher, columns are [rusher_xy, qb_xy]
    expected_centroid = np.squeeze(
        np.matmul(O_B_stacked, tau_hat), axis=-1
    )  ## (t, k, 2) convex combination of rusher and qb
    pdf_location = norm(loc=expected_centroid, scale=np.sqrt(sigma_hat)).pdf(
        D_rep[:, :, 0:2]
    )
    pdf_location = np.prod(
        pdf_location, -1
    )  ## x, y independent -> multiply their densities

    if o_dim < 3:
        ### no orientation channel (e.g. orientation term disabled in tests)
        return pdf_location

    rusher_orientation = np.cos(np.deg2rad(O[:, :, -1]))
    blocker_orientation = -1 * rusher_orientation

    rusher_orientation_scaled = 0.5 * (rusher_orientation + 1)
    blocker_orientation_scaled = 0.5 * (blocker_orientation + 1)

    rusher_orientation_scaled[rusher_orientation_scaled == 0] = 0.001
    rusher_orientation_scaled[rusher_orientation_scaled == 1] = 0.999

    blocker_orientation_scaled[blocker_orientation_scaled == 0] = 0.001
    blocker_orientation_scaled[blocker_orientation_scaled == 1] = 0.999

    beta_variance = 0.85 * (rusher_orientation_scaled) * (1 - rusher_orientation_scaled)
    alpha_param = np.power(rusher_orientation_scaled, 2) * (
        (1 - rusher_orientation_scaled) / beta_variance - 1 / rusher_orientation_scaled
    )
    beta_param = alpha_param * (1 / rusher_orientation_scaled - 1)

    pdf_orientation = beta(alpha_param, beta_param).pdf(blocker_orientation_scaled)

    return pdf_location * pdf_orientation


def sequence_log_likelihood(
    emission: np.ndarray,
    initial_state_distribution: np.ndarray,
    transition_matrix: np.ndarray,
) -> float:
    """incomplete-data log-likelihood log P(D_{1:t}) for one defender

    Runs the scaled forward recursion on the UNnormalized emission densities and
    accumulates the log of each per-step normalizer, which sums to the exact data
    log-likelihood (the hidden assignment chain marginalized out). EM is guaranteed
    to increase this quantity, so it is the monotonicity signal for the M-steps.

    Args:
        emission (np.ndarray): (t, k) unnormalized emission densities
        initial_state_distribution (np.ndarray): length-k prior over states
        transition_matrix (np.ndarray): (k, k) transition matrix

    Returns:
        float: log P(D_{1:t})
    """
    t, _ = emission.shape
    alpha = initial_state_distribution.flatten() * emission[0, :]
    c = alpha.sum()
    log_lik = np.log(c)
    alpha = alpha / c
    for i in range(1, t):
        alpha = np.matmul(transition_matrix, alpha) * emission[i, :]
        c = alpha.sum()
        log_lik += np.log(c)
        alpha = alpha / c
    return float(log_lik)


def build_transition_matrix(rho_hat: float, k: int) -> np.ndarray:
    """(k, k) transition matrix with rho on the diagonal and (1-rho)/(k-1) off"""
    transition_matrix = np.zeros((k, k))
    np.fill_diagonal(transition_matrix, rho_hat)
    transition_matrix[transition_matrix == 0] = (1 - rho_hat) / (k - 1)
    return transition_matrix


def expectation_matchup_step(
    tau_hat: np.ndarray,
    sigma_hat: float,
    rho_hat: float,
    forward_result: np.ndarray,
    D: np.ndarray,
    O: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """returns a  t x k matrix indicating probability that at timestep t, a player is assigned (guarding) player k
    all from this wikipedia site https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
    Args:
        tau_hat (np.ndarray): (2,1) simplex vector
        sigma_hat (float): variance estimate of player position
        rho_hat (float): transition probability from one player assignment to another
        forward_result (np.ndarray): k length vector indicating initial distribution over state for latent state
        D (np.ndarray): t x 3 dimensional vector of offensive player positions and orientation (x,y,o)
        O (np.ndarray): t x k x 3 dimensional tensor of defensive player positions and orientation (x,y,o)
        B (np.ndarray): t x 2 dimensional vector of ball carrier positions (x,y)
        iterations (int): when to stop algorithm

    Returns:
        np.ndarray: t x k x k matrix indicating probability that at timestep t, a player is assigned (guarding) player k
    """

    t, k, _ = O.shape  ### get dimensions

    pdf_emission = compute_emission(tau_hat, sigma_hat, D, O, B)
    ### normalize across the k candidate rushers at each timestep. This per-timestep
    ### rescaling cancels in the gamma/xi posteriors, so it does not affect the
    ### E-step; the absolute likelihood scale is recovered separately in
    ### sequence_log_likelihood using the UNnormalized emission.
    pdf_emission = pdf_emission / pdf_emission.sum(axis=1, keepdims=True)

    transition_matrix = build_transition_matrix(rho_hat, k)

    forward_result = forward_procedure(
        pdf_emission, forward_result, transition_matrix, t
    )
    backward_result = backward_procedure(pdf_emission, transition_matrix, t)

    lam_normalized = calculate_lambda(forward_result, backward_result)

    eta = calculate_eta(
        backward_result, pdf_emission, forward_result, transition_matrix
    )

    return lam_normalized, eta


def forward_procedure(
    pdf_location_difference: np.ndarray,
    initial_state_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    max_timestep: int = 0,
) -> np.ndarray:
    """implementation of forward algorithm

    Args:
        pdf_location_difference (np.ndarray): t x k matrix of probabilities according to normal distribution P(yt | xt = k)
        initial_state_distribution (np.ndarray): (k x 1) vector (will be used recursively)
        transition_matrix (np.ndarray): k x k transition matrix P(xt = k | x(t-1) = k) etc
        max_timestep (int, optional): length of sequence we are using Defaults to 0.

    Returns:
        np.ndarray: (max_timestep x k) vector of estimated forward values
    """
    k, _ = transition_matrix.shape
    out_array = np.zeros((max_timestep, k))
    t = 0  ## base case
    ### initial_state_distribution is the length-k prior pi over states; flatten so
    ### a (k,) or (k, 1) argument both behave as the full prior (previously [0, :]
    ### picked only its first component).
    out_array[t, :] = initial_state_distribution.flatten() * pdf_location_difference[t, :]
    out_array[t, :] /= out_array[t, :].sum()
    t += 1
    while t < max_timestep:
        out_array[t, :] = (
            np.matmul(transition_matrix, out_array[t - 1, :])
            * pdf_location_difference[t, :]
        )
        out_array[t, :] /= out_array[t, :].sum()
        t += 1
    return out_array


def backward_procedure(
    pdf_location_difference: np.ndarray,
    transition_matrix: np.ndarray,
    max_timestep: int = 0,
) -> np.ndarray:
    """implementation of backward algorithm

    Args:
        pdf_location_difference (np.ndarray): t x k matrix of probabilities according to normal distribution P(yt | xt = k)

        transition_matrix (np.ndarray): k x k transition matrix P(xt = k | x(t-1) = k) etc
        max_timestep (int, optional): length of sequence we are using Defaults to 0.

    Returns:
        np.ndarray: (max_timestep x k) vector of estimated forward values
    """
    k, _ = transition_matrix.shape
    out_array = np.zeros((max_timestep, k))
    t = max_timestep - 1  ## base case
    out_array[t, :] = 1  ### initial state probs are 1
    t -= 1
    while t >= 0:
        product = out_array[t + 1, :] * pdf_location_difference[t + 1, :]
        out_array[t, :] = np.matmul(transition_matrix, product)
        out_array[t, :] /= out_array[t, :].sum()
        t -= 1
    return out_array


def maximize_tau(X: np.ndarray, Sigma: np.ndarray, D: np.ndarray) -> np.ndarray:
    """produces estimate of tau vector using generalized least squares

    Args:
        X (np.ndarray): design matrix
        Sigma (np.ndarray): weight matrix (diagonalized vector (n ,1))
        D (np.ndarray): observation matrix

    Returns:
        np.ndarray: simplex vector
    """

    _, d = X.shape
    X_T_W = X.T * (Sigma)  ### X' W, with W = diag(Sigma)
    X_T_W_X = np.matmul(X_T_W, X)
    tau_gls = solve(X_T_W_X, np.matmul(X_T_W, D))  # unconstrained gls estimator
    unit_vector = np.ones((d, 1))
    ### constrained gls projection onto the simplex {tau : 1' tau = 1}
    ### tau_hat = tau_gls + (X'WX)^-1 1 * (1 - 1' tau_gls) / (1' (X'WX)^-1 1)
    A = solve(X_T_W_X, unit_vector)  # (X'WX)^-1 1
    denom = float(unit_vector.T @ A)  # 1' (X'WX)^-1 1 (scalar)
    correction = (1.0 - float(unit_vector.T @ tau_gls)) / denom
    tau_hat = tau_gls + A * correction
    return tau_hat


def maximize_sigma(
    tau_hat: np.ndarray, Sigma: np.ndarray, D: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """finds the variance value

    Args:
        tau_hat (np.ndarray): estimate of tau
        Sigma (np.ndarray): diagonal weight matrix represented as (n,1) array
        D (np.ndarray): observation matrix
        X (np.ndarray): design matrix

    Returns:
        np.ndarray: scalar of sigma
    """

    ### isotropic variance MLE: weighted SSE divided by the responsibility mass.
    ### Each design row carries weight Sigma (the assignment responsibility), and
    ### for a fixed (t, j) the responsibilities sum to 1 across the k candidate
    ### rushers, so Sigma.sum() is the effective per-coordinate sample size. The
    ### previous normalizer X.shape[0] (= 2*t*j*k raw rows) biased sigma low by ~k.
    residual = D - np.matmul(X, tau_hat)
    return np.matmul(residual.T * (Sigma), residual) / Sigma.sum()


def maximize_rho(A: List[np.ndarray]) -> float:
    """finds best rho param

    Args:
        A (List[np.ndarray]): list of transitions by possession

    Returns:
        float: our best estimate of rho
    """

    ### Closed-form M-step for the stickiness rho. The transition model is
    ### P(stay) = rho, P(switch to a specific other state) = (1 - rho) / (k - 1).
    ### Maximizing the expected complete-data transition log-likelihood
    ###   S log rho + W log(1 - rho) - W log(k - 1)
    ### (S = expected stays, W = expected switches) gives rho = S / (S + W),
    ### i.e. Q = rho / (1 - rho) = S / W with NO (k - 1) factor.
    stay = 0.0
    switch = 0.0
    for element in A:
        c = np.sum(element, axis=(0, 1))  ### (k, k) expected transition counts
        stay += np.trace(c)
        switch += c.sum() - np.trace(c)
    Q_hat = stay / switch
    return Q_hat / (1 + Q_hat)


def expectation_matchup_possession(
    tau_hat: np.ndarray,
    sigma_hat: float,
    rho_hat: float,
    forward_result: np.ndarray,
    D: np.ndarray,
    O: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """returns a  t x k matrix indicating probability that at timestep t, a player is assigned (guarding) player k
    all from this wikipedia site https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
    Args:
        tau_hat (np.ndarray): (2,1) simplex vector
        sigma_hat (float): variance estimate of player position
        rho_hat (float): transition probability from one player assignment to another
        forward_result (np.ndarray): j x k matrix indicatng starting state distribution
        D (np.ndarray): t x j x 2 dimensional vector of offensive player positions (x,y)
        O (np.ndarray): t x k x 2 dimensional tensor of defensive player positions (x,y)
        B (np.ndarray): t x 2 dimensional vector of ball carrier positions (x,y)

    Returns:
        np.ndarray: t x j x k x k matrix indicating probability that at timestep t, a player is assigned (guarding) player k
    """
    A_list = []
    I_list = []
    for i in range(D.shape[1]):
        I, A = expectation_matchup_step(
            tau_hat,
            sigma_hat,
            rho_hat,
            forward_result[i, :][:, np.newaxis],
            D[:, i, :],
            O,
            B,
        )

        I_list.append(I)
        A_list.append(A)
    A_stack = np.stack(A_list, 1)
    I_stack = np.stack(I_list, -2)
    return I_stack, A_stack


def expectation_maximization_possession(
    tau_init: np.ndarray,
    sigma_init: float,
    rho_init: float,
    forward_result: np.ndarray,
    D: np.ndarray,
    O: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """computes the e-m algorithm for defender assignments

    Args:
        tau_init (np.ndarray): initial value for parameter vector
        sigma_init (float): initial value for sigma^2
        rho_init (float): initial value for rho
        forward_result (np.ndarray): j x k matrix indicating distribution over starting states
        D (np.ndarray): t x j x 2 dimensional vector of offensive player positions (x,y)
        O (np.ndarray): t x k x 2 dimensional tensor of defensive player positions (x,y)
        B (np.ndarray): t x 2 dimensional vector of ball carrier positions (x,y)

    Returns:
        Tuple[np.ndarray,float,float]: estimated values of initial variables after one time step
    """

    I, A = expectation_matchup_possession(
        tau_init, sigma_init, rho_init, forward_result, D, O, B
    )  ### expectation step

    ### rho update
    rho_new = maximize_rho([A])

    ### tau update

    X, y, Sigma = voxels_to_design_response(D, O, B, I)
    tau_new = maximize_tau(X, Sigma, y)

    ### sigma update
    sigma_new = np.squeeze(maximize_sigma(tau_new, Sigma, y, X)).item()

    return tau_new, sigma_new, rho_new, I[0, :, :]


def expectation_maximization(
    tau_init: np.ndarray,
    sigma_init: float,
    rho_init: float,
    forward_result: List[np.ndarray],
    D: List[np.ndarray],
    O: List[np.ndarray],
    B: List[np.ndarray],
) -> Tuple[np.ndarray, float, float]:
    """computes the e-m algorithm for defender assignments

    Args:
        tau_init (np.ndarray): initial value for parameter vector
        sigma_init (float): initial value for sigma^2
        rho_init (float): initial value for rho
        forward_result List(np.ndarray): j x k matrix indicating distribution over starting states n of them
        D List(np.ndarray): t x j x 2 dimensional vector of offensive player positions (x,y) n of them
        O List(np.ndarray): t x k x 2 dimensional tensor of defensive player positions (x,y) n of them
        B List(np.ndarray): t x 2 dimensional vector of ball carrier positions (x,y) n of them

    Returns:
        Tuple[np.ndarray,float,float]: estimated values of initial variables after one time step

    """
    n = len(forward_result)
    forward_result_list = []
    A_list = []
    X_list = []
    y_list = []
    Sigma_list = []
    for i in range(n):
        I, A = expectation_matchup_possession(
            tau_init, sigma_init, rho_init, forward_result[i], D[i], O[i], B[i]
        )  ### expectation step

        X, y, Sigma = voxels_to_design_response(D[i], O[i], B[i], I)
        Sigma_list.append(Sigma)
        y_list.append(y)
        X_list.append(X)
        forward_result_list.append(I[0, :, :])
        A_list.append(A)

    X_new = np.vstack(X_list)
    y_new = np.vstack(y_list)
    Sigma_new = np.concatenate(Sigma_list)

    ### incomplete-data log-likelihood at the CURRENT parameters; EM must increase
    ### this across iterations (use it as the monotonicity / correctness signal).
    log_likelihood = data_log_likelihood(
        tau_init, sigma_init, rho_init, forward_result, D, O, B
    )
    print(f"Data log-likelihood: {log_likelihood}")

    ### rho update
    rho_new = maximize_rho(A_list)

    ### tau update
    tau_new = maximize_tau(X_new, Sigma_new, y_new)

    ### sigma update
    sigma_new = np.squeeze(maximize_sigma(tau_new, Sigma_new, y_new, X_new)).item()

    return tau_new, sigma_new, rho_new, forward_result_list


def data_log_likelihood(
    tau: np.ndarray,
    sigma: float,
    rho: float,
    forward_result: List[np.ndarray],
    D: List[np.ndarray],
    O: List[np.ndarray],
    B: List[np.ndarray],
) -> float:
    """total incomplete-data log-likelihood across all possessions and defenders

    Sums log P(D_{1:t}) (hidden assignment chain marginalized out, via the forward
    recursion on unnormalized emissions) over every defender in every possession.
    EM is guaranteed to increase this between iterations, so a non-decreasing
    sequence of these values is the correctness check on the M-step updates.

    Args:
        tau (np.ndarray): (2, 1) simplex vector
        sigma (float): position variance
        rho (float): stickiness
        forward_result (List[np.ndarray]): per-possession (j, k) prior over states
        D (List[np.ndarray]): per-possession (t, j, .) blocker positions
        O (List[np.ndarray]): per-possession (t, k, .) rusher positions
        B (List[np.ndarray]): per-possession (t, 2) qb positions

    Returns:
        float: total data log-likelihood
    """
    total = 0.0
    for i in range(len(D)):
        k = O[i].shape[1]
        transition_matrix = build_transition_matrix(rho, k)
        prior = np.asarray(forward_result[i])
        for m in range(D[i].shape[1]):
            emission = compute_emission(tau, sigma, D[i][:, m, :], O[i], B[i])
            total += sequence_log_likelihood(emission, prior[m, :], transition_matrix)
    return total


if __name__ == "__main__":
    import pandas as pd

    from data_processing import possession_to_voxel

    ### by position
    param_list = []
    data = pd.read_csv("data/sample_data_week_0.csv")
    data = data[data["possession_id"] != 670]
    positions = ["T", "C", "RB", "TE", "WR", "FB", "G"]
    for position in positions:
        print(f"fitting data for {position}")
        pos_data = data[(data["officialPosition_pb"] == position)]

        voxel_data = []
        for _, poss in pos_data.groupby("possession_id"):
            voxel_data.append(possession_to_voxel(poss))

        n = len(voxel_data)

        B_list = [voxel[0] for voxel in voxel_data if voxel[1].shape[1] != 1]
        O_list = [voxel[1] for voxel in voxel_data if voxel[1].shape[1] != 1]
        D_list = [voxel[2] for voxel in voxel_data if voxel[1].shape[1] != 1]
        k_list = [O.shape[1] for O in O_list]
        j_list = [D.shape[1] for D in D_list]
        t_list = [B.shape[0] for B in B_list]

        tau_hat = np.array([0.8, 0.2]).reshape((2, 1))
        rho_hat = 0.95
        sigma_hat = 2
        i = 0

        starting_state_distribution = [
            np.ones((j, k)) / k for j, k in zip(j_list, k_list)
        ]

        while i <= 15:
            (
                tau_hat,
                sigma_hat,
                rho_hat,
                starting_state_distribution,
            ) = expectation_maximization(
                tau_hat,
                forward_result=starting_state_distribution,
                rho_init=rho_hat,
                sigma_init=sigma_hat,
                B=B_list,
                D=D_list,
                O=O_list,
            )
            i += 1
        data_dict = {
            "tau": tau_hat,
            "sigma": sigma_hat,
            "rho": rho_hat,
            "position": position,
        }
        print(data_dict)
        param_list.append(data_dict)
        print("param estimation completed")
    pd.DataFrame(param_list).to_csv("fitted_params.csv", index=False)
