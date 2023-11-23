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
    partial_product = backward_result * pdf
    partial_product /= partial_product.sum(axis=1, keepdims=True)
    eta = np.zeros((t - 1, k, k))
    for i in range(t - 1):
        eta[i, :, :] = transition_matrix * np.outer(
            partial_product[i + 1, :], forward_result[i, :]
        )
        eta[i, :, :] /= eta[i, :, :].sum(axis=1, keepdims=True)
    return eta


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
    D = np.repeat(
        D[:, np.newaxis, :], k, axis=-2
    )  ### replicate individual lineman k times
    B = np.repeat(
        B[:, np.newaxis, :], k, axis=-2
    )  ## extend ballcarrier k times B has same shape as O

    O_B_stacked = np.stack(
        (O[:, :, 0:2], B[:, :, 0:2]), axis=-1
    )  ## new matrix with 4 dimensions (t x 2 x k x 2)

    expected_centroid = np.squeeze(
        np.matmul(O_B_stacked, tau_hat)
    )  ## gets convex combination of expected offensive lineman centroid for each possible defender
    pdf_location_difference = norm(loc=expected_centroid, scale=np.sqrt(sigma_hat)).pdf(
        D[:, :, 0:2]
    )
    pdf_location_difference = np.prod(
        pdf_location_difference, -1
    )  ## since x,y independent normal we multily their densities

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

    pdf_orientation_difference = beta(alpha_param, beta_param).pdf(
        blocker_orientation_scaled
    )

    pdf_emission = pdf_location_difference * pdf_orientation_difference
    pdf_emission /= pdf_emission.sum(axis=1, keepdims=True)

    transition_matrix = np.zeros((k, k))  ### state transition matrix
    np.fill_diagonal(transition_matrix, rho_hat)
    transition_matrix[transition_matrix == 0] = (1 - rho_hat) / (k - 1)

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
    out_array[t, :] = initial_state_distribution[t, :] * pdf_location_difference[t, :]
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
    return np.matmul(residual.T * (Sigma), residual) / n


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

    likelihood = 0
    likelihood_2 = 0
    for element in A:
        _, _, k, _ = element.shape
        likelihood += np.trace(np.sum(element, axis=(0, 1)))
        c = np.sum(element, axis=(0, 1))
        likelihood_2 += 1 / (k - 1) * (c.sum() - np.trace(c))
    Q_hat = likelihood / likelihood_2
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
    rho_new = maximize_rho_possession(A)

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

    lik = calculate_log_likelihood_regression(
        Sigma_new, tau_init, X_new, y_new, sigma_init
    )
    lik_2 = calculate_log_likelihood_probability(rho_init, A_list)

    print(f"Total Likelihood: {lik_2 - lik}")

    ### rho update
    rho_new = maximize_rho(A_list)

    ### tau update
    tau_new = maximize_tau(X_new, Sigma_new, y_new)

    ### sigma update
    sigma_new = np.squeeze(maximize_sigma(tau_new, Sigma_new, y_new, X_new)).item()

    return tau_new, sigma_new, rho_new, forward_result_list


def calculate_log_likelihood_regression(
    Sigma: np.ndarray, tau: np.ndarray, X: np.ndarray, D: np.ndarray, sigma: np.ndarray
) -> float:
    """

    Args:
        Sigma (np.ndarray): possition assignments (n x 1) for a possession
        tau (np.ndarray):  (2 x 1 ) coeff vector
        X (np.ndarray): n x 2 matrix
        D (np.ndarray): n x 1 matrix
        sigma (np.ndarray): variance float estimate
    """

    residual = np.square(np.matmul(X, tau) - D).T / sigma
    return np.sum(Sigma * residual)


def calculate_log_likelihood_probability(rho: float, A: List[np.ndarray]) -> float:
    """

    Args:
        rho (float): value of switch
        A (List[np.ndarray]): list of arrays to add likelihood to
    Returns:
        float: log likelihood
    """
    likelihood = 0
    for element in A:
        _, _, k, _ = element.shape
        likelihood += np.log(rho) * np.trace(np.sum(element, axis=(0, 1)))
        c = np.sum(element, axis=(0, 1))
        likelihood += np.log((1 - rho) / (k - 1)) * (c.sum() - np.trace(c))
    return likelihood


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
