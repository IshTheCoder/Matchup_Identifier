if __name__ == "__main__":
    import pymc as pm
    import pandas as pd
    import numpy as np
    import ast

    print("loading necessary data")
    feature_data = pd.read_csv("strain_design_data.csv")

    feature_data = feature_data[
        ~feature_data["grouped_pass_rusher_position"].isin(["G", "RB"])
    ]
    feature_data["assignment_dict"] = feature_data["assignment_dict"].apply(
        lambda x: ast.literal_eval(x)
    )
    target = feature_data["strain_rate"]

    all_blockers = set().union(*(d.keys() for d in feature_data["assignment_dict"]))

    n_blockers = len(all_blockers)
    n_rushers = len(feature_data["nflId_pr"].unique())
    n_offense = len(feature_data["possessionTeam"].unique())
    n_defense = len(feature_data["defensiveTeam"].unique())

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

    (
        rusher_id_batch,
        blocker_id_batch,
        offense_id_batch,
        defense_id_batch,
        design_matrix_batch,
        blocker_design_matrix_batch,
        target_batch,
    ) = pm.Minibatch(
        rusher_identifier,
        blocker_identifier,
        offense_identifier,
        defense_identifier,
        design_matrix,
        blocker_design_matrix,
        target,
        batch_size=256,
    )
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
            beta_rusher[rusher_id_batch]
            + beta_offense[offense_id_batch]
            + beta_defense[defense_id_batch]
            + pm.math.dot(blocker_design_matrix_batch, beta_blocker)
            + pm.math.dot(design_matrix_batch, beta_covariates)
            + intercept
        )

        # Likelihood (sampling distribution) of observations
        likelihood = pm.Normal(
            "likelihood",
            mu=mu,
            sigma=sigma,
            observed=target_batch,
            total_size=target.shape,
        )
        print("created basic model")
    with basic_model:
        # draw 1000 posterior samples
        print("beginning variational inference")
        # idata = pm.sample(chains = 4, cores = 1, return_inferencedata=True)
        idata = pm.fit(method="advi")
        idata.to_netcdf("pymc_posterior_sample_frame_level_strain.nc")
