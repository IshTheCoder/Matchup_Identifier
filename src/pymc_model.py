if __name__ == "__main__":
    import pymc as pm
    import pandas as pd
    import pickle

    print("loading necessary data")
    design_matrix = pd.read_csv("blocker_rusher_design_matrix.csv").to_numpy()[:, 1:]
    feature_data = pd.read_csv("strain_design_data.csv")

    rusher_inverse_map = {
        val: index
        for index, val in pickle.load(open("rusher_encoding.pkl", "rb")).items()
    }
    qb_inverse_map = {
        val: index for index, val in pickle.load(open("qb_encoding.pkl", "rb")).items()
    }

    target = feature_data["y_jt"]
    strain_velo = 0.5 * (feature_data["strain_rate"]) ** 2

    n_blockers = design_matrix.shape[1]
    n_rushers = len(set(feature_data["nflId_pr"]))
    n_qbs = len(qb_inverse_map)
    N = design_matrix.shape[0]
    rusher_identifier = feature_data["nflId_pr"].apply(lambda x: rusher_inverse_map[x])
    qb_identifier = feature_data["nflId_qb"].apply(lambda x: qb_inverse_map[x])

    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters
        beta_rusher = pm.Exponential("beta_rusher", lam=1, shape=n_rushers)
        sigma = pm.HalfNormal("sigma", sigma=1)
        sigma_blocker = pm.HalfNormal("sigma_blocker", sigma=1)
        sigma_qb = pm.HalfNormal("sigma_qb", sigma=1)

        beta_blocker = pm.Normal(
            "beta_blocker", mu=0, sigma=sigma_blocker, shape=n_blockers
        )
        beta_qb = pm.Normal("beta_qb", mu=0, sigma=sigma_qb, shape=n_qbs)
        # Expected value of outcome
        mu = (beta_rusher[rusher_identifier] - beta_qb[qb_identifier]) - pm.math.dot(
            design_matrix, beta_blocker
        ) * strain_velo

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=target)
        print("created basic model")
    with basic_model:
        # draw 1000 posterior samples
        print("beginning sampling")
        idata = pm.sample(chains=2, cores=1, return_inferencedata=True)
        idata.to_netcdf("pymc_posterior_sample_qb.nc")
