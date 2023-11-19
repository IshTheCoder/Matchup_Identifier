import stan
import pandas as pd


if __name__ == "__main__":
    print("loading necessary data")
    design_matrix = pd.read_csv("blocker_rusher_design_matrix.csv").to_numpy()[:, 1:]
    feature_data = pd.read_csv("strain_design_data.csv")

    target = feature_data["y_jt"]
    strain_velo = 0.5 * (feature_data["strain_rate"]) ** 2

    n_blockers = design_matrix.shape[1]
    n_rushers = len(set(feature_data["nflId_pr"]))
    N = design_matrix.shape[0]
    rusher_identifier = feature_data["nflId_pr"].astype("category").cat.codes + 1

    stan_data = {
        "velocity": strain_velo.values,
        "N": N,
        "rusher_identifier": rusher_identifier.values,
        "X_blockers": design_matrix,
        "acceleration": target.values,
        "N_blockers": n_blockers,
        "N_rushers": n_rushers,
    }

    with open("strain.stan") as f:
        stan_code = f.read()
        f.close()

    print("building stan model")
    posterior = stan.build(stan_code, data=stan_data, random_seed=1)

    print("beginning fit of stan model")
    fit = posterior.sample(num_chains=1, num_samples=1000)
