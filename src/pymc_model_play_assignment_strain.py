if __name__ == "__main__":
    import pymc as pm
    import pandas as pd
    import numpy as np
    from functools import reduce
    import ast

    def reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + value
        return accumulator

    print("loading necessary data")
    feature_data = pd.read_csv("strain_design_data.csv")

    feature_data["grouped_pass_rusher_position"] = feature_data[
        "officialPosition"
    ].apply(lambda x: "S" if x in ["SS", "FS", "CB"] else x)
    feature_data["grouped_pass_rusher_position"] = feature_data[
        "grouped_pass_rusher_position"
    ].apply(lambda x: "I" if x in ["MLB", "ILB", "LB"] else x)

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

    agg_dict = {
        "down": "first",
        "yardsToGo": "first",
        "possessionTeam": "first",
        "defensiveTeam": "first",
        "grouped_pass_rusher_position": "first",
        "strain_rate": "mean",
        "frameId": "count",
    }

    total_attention = (
        feature_data.groupby(["playId", "gameId", "nflId_pr"])
        .apply(lambda x: reduce(reducer, x.assignment_dict.values.tolist(), {}))
        .reset_index()
    )
    total_attention.rename(inplace=True, axis=1, mapper={0: "assignment_dict"})

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
                "grouped_pass_rusher_position",
                "strain_rate",
                "assignment_dict",
                "frameId",
            ]
        ]
        .groupby(["playId", "gameId", "nflId_pr"])
        .agg(agg_dict)
        .reset_index()
    ).merge(total_attention)

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
        blocker_design_matrix[
            i, [blocker_map[nfl_id] for nfl_id in assignment_map.keys()]
        ] /= row["frameId"]
        i += 1

    print("finished making design matrix for blockers")

    rusher_identifier = pd.factorize(feature_data["nflId_pr"])[0]

    offense_identifier = pd.factorize(feature_data["possessionTeam"])[0]
    defense_identifier = pd.factorize(feature_data["defensiveTeam"])[0]

    covariate_columns = [
        "down",
        "grouped_pass_rusher_position",
    ] + ["yardsToGo"]

    design_matrix = pd.get_dummies(
        feature_data[covariate_columns],
        columns=["down", "grouped_pass_rusher_position"],
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
            pm.math.dot(blocker_design_matrix, beta_blocker)
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
        idata = pm.sample(
            chains=4, return_inferencedata=True, idata_kwargs={"log_likelihood": True}
        )
        idata.to_netcdf("pymc_posterior_sample_play_level_assignment_strain.nc")
