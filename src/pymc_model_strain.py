if __name__ == "__main__":
    import pymc as pm
    import pandas as pd
    import numpy as np
    import ast

    print("loading necessary data")

    feature_data = pd.read_csv("strain_design_data.csv")
    feature_data["grouped_pass_rusher_position"] = feature_data[
        "officialPosition"
    ].apply(lambda x: "S" if x in ["SS", "FS", "CB"] else x)
    feature_data["grouped_pass_rusher_position"] = feature_data[
        "grouped_pass_rusher_position"
    ].apply(lambda x: "I" if x in ["MLB", "ILB", "LB"] else x)
    feature_data["intercept"] = 1
    plays = pd.read_csv("data/plays.csv")
    feature_data = feature_data.merge(
        plays[
            ["gameId", "playId", "possessionTeam", "defensiveTeam", "down", "yardsToGo"]
        ],
        on=["gameId", "playId"],
    )

    feature_data = feature_data[
        ~feature_data["grouped_pass_rusher_position"].isin(["G", "RB"])
    ]
    feature_data["assignment_dict"] = feature_data["assignment_dict"].apply(
        lambda x: ast.literal_eval(x)
    )
    target = feature_data[["strain_acceleration"]].to_numpy()

    print(f"Target has shape {target.shape}")

    all_blockers = set().union(*(d.keys() for d in feature_data["assignment_dict"]))

    n_blockers = len(all_blockers)
    n_rushers = len(feature_data["nflId_pr"].unique())
    n_offense = len(feature_data["possessionTeam"].unique())
    n_defense = len(feature_data["defensiveTeam"].unique())

    print(
        f"There are {n_blockers} blockers, {n_rushers} rushers, {n_offense} offensive teams, {n_defense} defensive teams"
    )

    blocker_map = {key: val for val, key in enumerate(all_blockers)}

    N = feature_data.shape[0]
    print("making design matrix for blockers")
    blocker_design_matrix = np.zeros((N, n_blockers))
    i = 0
    for _, row in feature_data.iterrows():
        assignment_map = row["assignment_dict"]
        blocker_design_matrix[
            i, [blocker_map[nfl_id] for nfl_id in assignment_map.keys()]
        ] = list(assignment_map.values())
        i += 1
    print("finished making design matrix for blockers")
    rusher_identifier = pd.factorize(feature_data["nflId_pr"])[0]
    offense_identifier = pd.factorize(feature_data["possessionTeam"])[0]
    defense_identifier = pd.factorize(feature_data["defensiveTeam"])[0]

    print(f"blocker effect design matrix has shape {blocker_design_matrix.shape}")
    covariate_columns = [
        "intercept",
        "down",
        "grouped_pass_rusher_position",
        "strain_rate",
    ] + ["yardsToGo"]

    design_matrix = pd.get_dummies(
        feature_data[covariate_columns],
        columns=["down", "grouped_pass_rusher_position"],
        drop_first=True,
    ).to_numpy()

    print(f"design matrix has shape {design_matrix.shape}")

    n_cols = design_matrix.shape[1]

    random_effect_design_matrix = feature_data[["intercept", "strain_rate"]].to_numpy()
    n_cols_re = random_effect_design_matrix.shape[1]

    print(f"Random effect design matrix has shape {random_effect_design_matrix.shape}")
    (
        rusher_id_batch,
        defense_id_batch,
        offense_id_batch,
        design_matrix_batch,
        random_effect_design_matrix_batch,
        blocker_design_matrix_batch,
        target_batch,
    ) = pm.Minibatch(
        rusher_identifier,
        defense_identifier,
        offense_identifier,
        design_matrix,
        random_effect_design_matrix,
        blocker_design_matrix,
        target,
        batch_size=128,
    )

    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters

        sigma = pm.HalfNormal("sigma", sigma=1)
        sigma_blocker = pm.HalfNormal("sigma_blocker", sigma=1)

        sigma_offense = pm.HalfNormal("sigma_offense", sigma=1)
        sigma_defense = pm.HalfNormal("sigma_defense", sigma=1)

        beta_covariates = pm.Normal(
            "beta_covariates", mu=0, sigma=10, shape=(n_cols, 1)
        )

        ## Group-specific effects
        # Hyper prior for the standard deviations

        # prior stddev in intercepts & slopes (variation across rushers):
        sd_dist = pm.Exponential.dist(0.5, shape=(n_cols_re,))

        # get back standard deviations and rho:
        chol, _, _ = pm.LKJCholeskyCov("chol", n=n_cols_re, eta=2.0, sd_dist=sd_dist)

        # population of varying effects:
        beta_rusher_raw = pm.Normal(
            "beta_rusher_raw", 0.0, 1.0, shape=(n_cols_re, n_rushers)
        )
        beta_rusher = pm.Deterministic(
            "beta_rusher", pm.math.dot(chol, beta_rusher_raw).T
        )
        beta_blocker = pm.Normal(
            "beta_blocker", mu=0, sigma=sigma_blocker, shape=(n_blockers, 1)
        )
        beta_offense = pm.Normal(
            "beta_offense", mu=0, sigma=sigma_offense, shape=(n_offense, 1)
        )
        beta_defense = pm.Normal(
            "beta_defense", mu=0, sigma=sigma_defense, shape=(n_defense, 1)
        )
        # Expected value of outcome
        mu = (
            beta_offense[offense_id_batch]
            + beta_defense[defense_id_batch]
            + pm.math.dot(blocker_design_matrix_batch, beta_blocker)
            + pm.math.dot(design_matrix_batch, beta_covariates)
            + pm.math.sum(
                beta_rusher[rusher_id_batch] * random_effect_design_matrix_batch,
                axis=1,
                keepdims=True,
            )
        )

        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal(
            "y_obs",
            mu=mu,
            sigma=sigma,
            observed=target_batch,
            total_size=design_matrix.shape[0],
        )
        print("created basic model")
    with basic_model:
        # draw 1000 posterior samples
        print("beginning variational inference")
        approx = pm.fit(method="advi", n=50000)
        idata = approx.sample(draws=5000, return_inferencedata=True)
        idata.to_netcdf(
            "pymc_posterior_sample_frame_level_strain_response_with_intercept.nc"
        )
