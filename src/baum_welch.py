### functions for determining e-m algorithm for hidden markov model
import numpy as np
from scipy.stats import norm


def estimate_matchup(tau_hat: np.ndarray, sigma_hat: float, rho_hat: float, D: np.ndarray, O:np.ndarray, B: np.ndarray, iterations: int) -> np.ndarray:
    """ returns a  t x k matrix indicating probability that at timestep t, a player is assigned (guarding) player k
    all from this wikipedia site https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
    Args:
        tau_hat (np.ndarray): (2,1) simplex vector 
        sigma_hat (float): variance estimate of player position
        rho_hat (float): transition probability from one player assignment to another 
        D (np.ndarray): t x 2 dimensional vector of offensive player positions (x,y)
        O (np.ndarray): t x k x 2 dimensional tensor of defensive player positions (x,y)
        B (np.ndarray): t x 2 dimensional vector of ball carrier positions (x,y)
        iterations (int): when to stop algorithm

    Returns:
        np.ndarray: t x k matrix indicating probability that at timestep t, a player is assigned (guarding) player k
    """

    t, k, _ = O.shape ### get dimensions
    D = np.repeat(D[:, np.newaxis, :], k, axis = -2) ### replicate individual lineman k times
    B = np.repeat(B[:, np.newaxis, :], k, axis = -2) ## extend ballcarrier k times B has same shape as O

    O_B_stacked = np.stack((O,B), axis = -1) ## new matrix with 4 dimensions (t x 2 x k x 2)

    expected_centroid = np.squeeze(np.matmul(O_B_stacked, tau_hat)) ## gets convex combination of expected offensive lineman centroid for each possible defender

    location_difference = (D - expected_centroid) / np.sqrt(sigma_hat) ### gets Z-score difference
    pdf_location_difference = np.prod(norm.pdf(location_difference),-1) ## since x,y independent normal we multily their densities
    pdf_location_difference = np.vstack([pdf_location_difference, np.ones((k,))])

    transition_matrix = np.zeros((k,k))  ### state transition matrix
    np.fill_diagonal(transition_matrix, rho_hat)
    transition_matrix[transition_matrix == 0] = (1-rho_hat)/(k-1)
    
    forward_result = np.ones((1,k)) / k
    i = 0
    while i <= iterations:
        forward_result = forward_procedure(pdf_location_difference, forward_result, transition_matrix, t)
        backward_result = backward_procedure(pdf_location_difference, transition_matrix, t)
        lam = forward_result * backward_result ### normalize row
        likelihood = lam.sum(axis = 1, keepdims = True) ### need this for later
        lam_normalized = lam / likelihood
        print(f"Iteration: {i}, Likelihood: {likelihood.mean()}")
        i += 1
    
    return lam_normalized

def forward_procedure(pdf_location_difference: np.ndarray, initial_state_distribution: np.ndarray, transition_matrix: np.ndarray, max_timestep:int = 0) -> np.ndarray:
    """ implementation of forward algorithm 

    Args:
        pdf_location_difference (np.ndarray): t x k matrix of probabilities according to normal distribution P(yt | xt = k)
        initial_state_distribution (np.ndarray): (k x 1) vector (will be used recursively)
        transition_matrix (np.ndarray): k x k transition matrix P(xt = k | x(t-1) = k) etc
        max_timestep (int, optional): length of sequence we are using Defaults to 0.

    Returns:
        np.ndarray: (max_timestep +1 x k) vector of estimated forward values
    """
    k,_ = transition_matrix.shape
    out_array = np.zeros((max_timestep + 1, k))
    t = 0 ## base case
    out_array[t,:] = initial_state_distribution[t,:] * pdf_location_difference[t,:]
    t += 1
    while t < max_timestep + 1:
        out_array[t,:] = np.matmul(transition_matrix, out_array[t-1,:]) * pdf_location_difference[t,:]
        t += 1
    return out_array

def backward_procedure(pdf_location_difference: np.ndarray, transition_matrix: np.ndarray, max_timestep:int = 0) -> np.ndarray:
    """ implementation of backward algorithm 

    Args:
        pdf_location_difference (np.ndarray): t x k matrix of probabilities according to normal distribution P(yt | xt = k)

        transition_matrix (np.ndarray): k x k transition matrix P(xt = k | x(t-1) = k) etc
        max_timestep (int, optional): length of sequence we are using Defaults to 0.

    Returns:
        np.ndarray: (max_timestep + 1 x k) vector of estimated forward values
    """
    k,_ = transition_matrix.shape
    out_array = np.zeros((max_timestep + 1, k))
    t = max_timestep ## base case
    out_array[t,:] = 1 ### initial state probs are 1
    t -= 1
    while t >= 0:
        out_array[t, :] = np.matmul(transition_matrix, out_array[t+1,:] * pdf_location_difference[t+1,:])
        t -= 1
    return out_array





if __name__ == "__main__":
    t = 100
    tau_hat = np.ones((2,1)) * .5 
    rho = .99
    sigma_hat = .5
    O = np.random.randn(t,3,2)
    B = np.random.randn(t,2)
    B_new = np.repeat(B[:, np.newaxis, :], 3, axis = -2)
    O_B_stacked = np.stack((O,B_new), axis = -1) ## new matrix with 4 dimensions (t x 2 x k x 2)
    expected_centroid = np.squeeze(np.matmul(O_B_stacked, tau_hat)) 
    
    transition_matrix = np.array([[.94,.03,.03],[.03,.94,.03],[.03,.03,.94]])
    states = np.zeros((t,))
    observations = np.zeros((t,2))
    observations[0,:] = sigma_hat*np.random.randn(1,2) + expected_centroid[0,0,:]
    for time in range(t-1):
        cur_state = int(states[time])
        next_state = np.argmax(np.random.multinomial(1,pvals = transition_matrix[cur_state,:]))

        states[time+1] = next_state
        observations[time+1,:] = sigma_hat*np.random.randn(1,2) + expected_centroid[time,next_state,:]

    print(states)
    estimate_matchup(tau_hat, sigma_hat + .2, rho - .05, observations, O, B, 2)













    

    


