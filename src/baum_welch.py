### functions for determining e-m algorithm for hidden markov model
from typing import List, Tuple

import numpy as np
from scipy.linalg import solve
from scipy.stats import norm

from data_processing import voxels_to_design_response


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
        D (np.ndarray): t x 2 dimensional vector of offensive player positions (x,y)
        O (np.ndarray): t x k x 2 dimensional tensor of defensive player positions (x,y)
        B (np.ndarray): t x 2 dimensional vector of ball carrier positions (x,y)
        iterations (int): when to stop algorithm

    Returns:
        np.ndarray: t x k matrix indicating probability that at timestep t, a player is assigned (guarding) player k
    """

    t, k, _ = O.shape  ### get dimensions
    D = np.repeat(
        D[:, np.newaxis, :], k, axis=-2
    )  ### replicate individual lineman k times
    B = np.repeat(
        B[:, np.newaxis, :], k, axis=-2
    )  ## extend ballcarrier k times B has same shape as O

    O_B_stacked = np.stack(
        (O, B), axis=-1
    )  ## new matrix with 4 dimensions (t x 2 x k x 2)

    expected_centroid = np.squeeze(
        np.matmul(O_B_stacked, tau_hat)
    )  ## gets convex combination of expected offensive lineman centroid for each possible defender
    pdf_location_difference = norm(loc=expected_centroid, scale=np.sqrt(sigma_hat)).pdf(
        D
    )
    pdf_location_difference = np.prod(
        pdf_location_difference, -1
    )  ## since x,y independent normal we multily their densities
    pdf_location_difference = np.vstack([pdf_location_difference, np.ones((k,))])

    transition_matrix = np.zeros((k, k))  ### state transition matrix
    np.fill_diagonal(transition_matrix, rho_hat)
    transition_matrix[transition_matrix == 0] = (1 - rho_hat) / (k - 1)

    forward_result = forward_procedure(
        pdf_location_difference, forward_result, transition_matrix, t
    )
    backward_result = backward_procedure(pdf_location_difference, transition_matrix, t)
    lam = forward_result * backward_result  ### normalize row
    likelihood = lam.sum(axis=1, keepdims=True)  ### need this for later
    lam_normalized = lam / likelihood

    eta = np.matmul(
        transition_matrix,
        (
            forward_result[0:-1, :]
            * backward_result[1:, :]
            * pdf_location_difference[1:, :]
        ).T,
    ).T
    eta /= eta.sum(axis=1, keepdims=True)

    return lam_normalized, eta, np.log(likelihood)


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
        np.ndarray: (max_timestep +1 x k) vector of estimated forward values
    """
    k, _ = transition_matrix.shape
    out_array = np.zeros((max_timestep + 1, k))
    t = 0  ## base case
    out_array[t, :] = initial_state_distribution[t, :] * pdf_location_difference[t, :]
    out_array[t, :] /= out_array[t, :].sum()
    t += 1
    while t < max_timestep + 1:
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
        np.ndarray: (max_timestep + 1 x k) vector of estimated forward values
    """
    k, _ = transition_matrix.shape
    out_array = np.zeros((max_timestep + 1, k))
    t = max_timestep  ## base case
    out_array[t, :] = 1  ### initial state probs are 1
    t -= 1
    while t >= 0:
        out_array[t, :] = np.matmul(
            transition_matrix, out_array[t + 1, :] * pdf_location_difference[t + 1, :]
        )
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
    X_T_Sigma_inv = X.T * (Sigma)
    X_T_Sigma_inv_X = np.matmul(X_T_Sigma_inv, X)
    tau_gls = solve(X_T_Sigma_inv_X, np.matmul(X_T_Sigma_inv, D))  # gls estimator
    unit_vector = np.ones((d, 1))
    A = solve(X_T_Sigma_inv_X, unit_vector)
    B = 1 / np.squeeze(unit_vector.T.dot(X_T_Sigma_inv_X).dot(unit_vector))
    tau_hat = tau_gls + (1 - tau_gls.dot(unit_vector.T)).dot(A * np.squeeze(B))
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

    n, _ = X.shape
    residual = D - np.matmul(X, tau_hat)
    return np.matmul(residual.T * Sigma, residual) / n


def maximize_rho_possession(A: np.ndarray) -> float:
    """returns estimate of rho

    Args:
        A (np.ndarray): (n*t x j x k) size tensor with first index representing each time step, j representing players and k representing pass rushers


    Returns:
        float: estimate of rho
    """

    ### note the first dimension of k is the time spent in your own state
    ### all other dimensions are transitions not to your own state

    _, _, k = A.shape
    numerator = np.sum(A[:, :, 0])
    denominator = np.sum(A) - numerator
    Q_hat = (1 / (k - 1)) * numerator / denominator

    rho = Q_hat / (1 + Q_hat)
    return rho


def maximize_rho(A: List[np.ndarray]) -> float:
    """finds best rho param

    Args:
        A (List[np.ndarray]): list of transitions by possession

    Returns:
        float: our best estimate of rho
    """

    numerator = 0
    total_sum = 0
    max_k = 0
    for element in A:
        numerator += element[:, :, 0].sum()
        total_sum += element.sum()
        k = element.shape[2]
        if k >= max_k:
            max_k = k

    Q_hat = (numerator / (total_sum - numerator)) * (1 / (max_k - 1))

    return Q_hat / (1 + Q_hat)


def expectation_matchup_possession(
    tau_hat: np.ndarray,
    sigma_hat: float,
    rho_hat: float,
    forward_result: np.ndarray,
    D: np.ndarray,
    O: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
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
        np.ndarray: t x j x k matrix indicating probability that at timestep t, a player is assigned (guarding) player k
    """
    A_list = []
    I_list = []
    log_lik = 0
    for i in range(D.shape[1]):
        I, A, likelihood = expectation_matchup_step(
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
        log_lik += likelihood
    A_stack = np.stack(A_list, -2)
    I_stack = np.stack(I_list, -2)
    return I_stack, A_stack, log_lik


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

    I, A, likelihood = expectation_matchup_possession(
        tau_init, sigma_init, rho_init, forward_result, D, O, B
    )  ### expectation step

    ### rho update
    rho_new = maximize_rho_possession(A)

    ### tau update

    X, y, Sigma = voxels_to_design_response(D, O, B, I)
    tau_new = maximize_tau(X, Sigma, y)

    ### sigma update
    sigma_new = np.squeeze(maximize_sigma(tau_new, Sigma, y, X))

    print(f"Likelihood: {likelihood}")

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
    likelihood_list = []
    X_list = []
    y_list = []
    Sigma_list = []
    for i in range(n):
        I, A, likelihood = expectation_matchup_possession(
            tau_init, sigma_init, rho_init, forward_result[i], D[i], O[i], B[i]
        )  ### expectation step
        X, y, Sigma = voxels_to_design_response(D[i], O[i], B[i], I)
        Sigma_list.append(Sigma)
        y_list.append(y)
        X_list.append(X)
        forward_result_list.append(I[0, :, :])
        A_list.append(A)
        likelihood_list.append(likelihood)

    X_new = np.vstack(X_list)
    y_new = np.vstack(y_list)
    Sigma_new = np.concatenate(Sigma_list)

    ### rho update
    rho_new = maximize_rho(A_list)

    ### tau update
    tau_new = maximize_tau(X_new, Sigma_new, y_new)

    ### sigma update
    sigma_new = np.squeeze(maximize_sigma(tau_new, Sigma_new, y_new, X_new))
    ## likelihood calc

    likelihood = np.concatenate(likelihood_list).mean()
    print(f"Likelihood: {likelihood}")

    return tau_new, sigma_new, rho_new, forward_result_list


if __name__ == "__main__":
    import pandas as pd

    from data_processing import possession_to_voxel

    data = pd.read_csv("data/sample_data.csv")

    voxel_data = []
    i = 0
    for _, poss in data.groupby("possession_id"):
        voxel_data.append(possession_to_voxel(poss))
        i += 1

        if i > 1000:
            break
    n = len(voxel_data)

    B_list = [voxel[0] for voxel in voxel_data]
    O_list = [voxel[1] for voxel in voxel_data]
    D_list = [voxel[2] for voxel in voxel_data]
    k_list = [O.shape[1] for O in O_list]
    j_list = [D.shape[1] for D in D_list]
    t_list = [B.shape[0] for B in B_list]

    tau_hat = np.array([0.8, 0.2]).reshape((2, 1))
    rho_hat = 0.95
    sigma_hat = 20
    i = 0

    starting_state_distribution = [np.ones((j, k)) / k for j, k in zip(j_list, k_list)]

    while i <= 3:
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
        print(tau_hat, sigma_hat, rho_hat)
