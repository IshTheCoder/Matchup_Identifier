if __name__ == "__main__":
    import pymc as pm
    import pandas as pd
    import pickle

    print("loading necessary data")
    feature_data = pd.read_csv("strain_design_data.csv")

    agg_dict = {
        "down": "first",
        "yardsToGo": "first",
        "possessionTeam": "first",
        "defensiveTeam": "first",
        "grouped_pass_blocker_position": "first",
        "grouped_pass_rusher_position": "first",
        "strain_rate": "mean",
        "num_blockers": "first",
        "assigned_blocker_id": "first",
    }

    feature_data = (
        feature_data[
            [
                "playId",
                "gameId",
                "nflId_pr",
                "down",
                "yardsToGo",
                "possessionTeam",
                "defensiveTeam",
                "grouped_pass_blocker_position",
                "grouped_pass_rusher_position",
                "strain_rate",
                "num_blockers",
                "assigned_blocker_id",
            ]
        ]
        .groupby(["playId", "gameId", "nflId_pr"])
        .agg(agg_dict)
        .reset_index()
    )
    feature_data = feature_data[
        ~feature_data["grouped_pass_rusher_position"].isin(["G", "RB"])
    ]

    target = feature_data["strain_rate"]

    n_blockers = len(feature_data["assigned_blocker_id"].unique())
    n_rushers = len(feature_data["nflId_pr"].unique())
    n_offense = len(feature_data["possessionTeam"].unique())
    n_defense = len(feature_data["defensiveTeam"].unique())

    N = feature_data.shape[0]
    rusher_identifier = pd.factorize(feature_data["nflId_pr"])[0]
    blocker_identifier = pd.factorize(feature_data["assigned_blocker_id"])[0]
    offense_identifier = pd.factorize(feature_data["possessionTeam"])[0]
    defense_identifier = pd.factorize(feature_data["defensiveTeam"])[0]

    covariate_columns = [
        "down",
        "grouped_pass_blocker_position",
        "grouped_pass_rusher_position",
    ] + ["yardsToGo", "num_blockers"]

    design_matrix = pd.get_dummies(
        feature_data[covariate_columns],
        columns=[
            "down",
            "grouped_pass_blocker_position",
            "grouped_pass_rusher_position",
        ],
        drop_first=True,
    ).to_numpy()

    n_cols = design_matrix.shape[1]

    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters

        sigma = pm.HalfNormal("sigma", sigma=1)
        sigma_blocker = pm.HalfNormal("sigma_blocker", sigma=1)
        sigma_rusher = pm.HalfNormal("sigma_rusher", sigma=1)
        sigma_offense = pm.HalfNormal("sigma_offense", sigma=1)
        sigma_defense = pm.HalfNormal("sigma_defense", sigma=1)

        beta_covariates = pm.Normal("beta_covariates", mu=0, sigma=10, shape=n_cols)
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        beta_rusher = pm.Normal(
            "beta_rusher", mu=0, sigma=sigma_rusher, shape=n_rushers
        )
        beta_blocker = pm.Normal(
            "beta_blocker", mu=0, sigma=sigma_blocker, shape=n_blockers
        )
        beta_offense = pm.Normal(
            "beta_offense", mu=0, sigma=sigma_offense, shape=n_offense
        )
        beta_defense = pm.Normal(
            "beta_defense", mu=0, sigma=sigma_defense, shape=n_defense
        )
        # Expected value of outcome
        mu = (
            beta_blocker[blocker_identifier]
            + beta_rusher[rusher_identifier]
            + beta_offense[offense_identifier]
            + beta_defense[defense_identifier]
            + pm.math.dot(design_matrix, beta_covariates)
            + intercept
        )

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=target)
        print("created basic model")
    with basic_model:
        # draw 1000 posterior samples
        print("beginning sampling")
        idata = pm.sample(chains=4, cores=1, return_inferencedata=True)
        idata.to_netcdf("pymc_posterior_sample_play_level_strain.nc")
