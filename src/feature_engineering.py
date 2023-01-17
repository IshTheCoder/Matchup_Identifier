import pandas as pd
import numpy as np
from itertools import chain


weeks_used = 8
raw_files = ["data/week" + str(i) + ".csv" for i in range(1, weeks_used + 1)]
pass_rush_files = [f"data/sample_data_week_{i}.csv" for i in range(weeks_used)]


def generate_acceleration_data():
    """
    generates acceleration towards qb in y direction hmm data
    """
    data_list = []
    for raw_file, pass_rush_file in zip(raw_files, pass_rush_files):
        data = pd.read_csv(raw_file)
        processed_week_data = pd.read_csv(pass_rush_file)

        ### get timesteps between snap and first event
        event_instances = (
            data.groupby(["gameId", "playId"])
            .apply(
                lambda x: x[x.event != "None"].drop_duplicates(["time"])[
                    ["time", "event"]
                ]
            )
            .reset_index()
        )
        event_timestamp = (
            event_instances.groupby(["gameId", "playId"])
            .apply(
                lambda x: x[
                    x.event.isin(
                        ["pass_forward", "run", "qb_sack", "qb_strip_sack", "ball_snap"]
                    )
                ][["time", "event"]]
            )
            .reset_index()
        )
        event_timestamp["event"] = event_timestamp["event"].apply(
            lambda x: "action" if x != "ball_snap" else x
        )
        event_timestamp["time"] = pd.to_datetime(event_timestamp["time"])
        event_wide = pd.pivot_table(
            event_timestamp[["gameId", "time", "playId", "event"]],
            columns="event",
            values=["time"],
            index=["gameId", "playId"],
        ).reset_index()
        event_wide.columns = ["gameId", "playId", "action", "ball_snap"]

        ### now merge
        merged_event_processed = pd.merge(event_wide, processed_week_data)
        filtered_processed = merged_event_processed[
            (
                pd.to_datetime(merged_event_processed.time)
                <= merged_event_processed.action
            )
            & (
                pd.to_datetime(merged_event_processed.time)
                >= merged_event_processed.ball_snap
            )
        ]
        pr_filtered = filtered_processed

        #### create physics features
        pr_filtered["dx_pr"] = pr_filtered.groupby(
            ["gameId", "playId", "nflId_pr", "nflId"]
        )["x_pr"].transform(lambda x: np.gradient(x, 0.1))
        pr_filtered["dy_pr"] = pr_filtered.groupby(
            ["gameId", "playId", "nflId_pr", "nflId"]
        )["y_pr"].transform(lambda x: np.gradient(x, 0.1))
        pr_filtered["dx_qb"] = pr_filtered.groupby(
            ["gameId", "playId", "nflId_pr", "nflId"]
        )["x_qb"].transform(lambda x: np.gradient(x, 0.1))
        pr_filtered["dy_qb"] = pr_filtered.groupby(
            ["gameId", "playId", "nflId_pr", "nflId"]
        )["y_qb"].transform(lambda x: np.gradient(x, 0.1))
        pr_filtered["x_pr_qb"] = pr_filtered["x_qb"] - pr_filtered["x_pr"]
        pr_filtered["y_pr_qb"] = pr_filtered["y_qb"] - pr_filtered["y_pr"]
        pr_filtered["d2y_pr"] = pr_filtered.groupby(
            ["gameId", "playId", "nflId_pr", "nflId"]
        )["dy_pr"].transform(lambda x: np.gradient(x, 0.1))
        pr_filtered["d2x_pr"] = pr_filtered.groupby(
            ["gameId", "playId", "nflId_pr", "nflId"]
        )["dx_pr"].transform(lambda x: np.gradient(x, 0.1))
        pr_filtered["pr_qb_norm"] = (
            pr_filtered["x_pr_qb"] ** 2 + pr_filtered["y_pr_qb"] ** 2
        )
        pr_filtered["scalar_projection"] = (
            pr_filtered["x_pr_qb"] * pr_filtered["d2x_pr"]
            + pr_filtered["y_pr_qb"] * pr_filtered["d2y_pr"]
        ) / pr_filtered["pr_qb_norm"]
        pr_filtered["d2y_pr_qb"] = (
            pr_filtered["scalar_projection"] * pr_filtered["y_pr_qb"]
        )
        pr_filtered["d2x_pr_qb"] = (
            pr_filtered["scalar_projection"] * pr_filtered["x_pr_qb"]
        )

        ### calculate strain
        pr_filtered["d_ij"] = np.sqrt(
            np.square(pr_filtered["x_pr"] - pr_filtered["x_qb"])
            + np.square(pr_filtered["y_pr"] - pr_filtered["y_qb"])
        )
        pr_filtered["strain_rate"] = pr_filtered.groupby(
            ["gameId", "playId", "nflId_pr", "nflId"]
        )["d_ij"].transform(lambda x: -np.gradient(x, 0.1))
        pr_filtered["strain_rate"] /= pr_filtered["d_ij"]
        pr_filtered["strain_acceleration"] = pr_filtered.groupby(
            ["gameId", "playId", "nflId_pr", "nflId"]
        )["strain_rate"].transform(lambda x: np.gradient(x, 0.1))

        ### append to list
        data_list.append(pr_filtered)

    pd.concat(data_list).to_csv("processed_data/pass_rusher_features.csv", index=False)


def generate_features(
    assignment_data: pd.DataFrame, acceleration_data: pd.DataFrame
) -> Tuple[pd.DataFrame, dict, dict]:
    """_summary_

    Args:
        assignment_data (pd.DataFrame): prob of assignment
        acceleration_data (pd.DataFrame): acceleration based data

    Returns:
        pd.DataFrame: clean df to use for analysis as well as dictionary params
    """
    assignment_data_agg = (
        assignment_data.groupby(["nflId_pr", "playId", "frameId", "gameId"])
        .apply(lambda x: {i: val for val, i in zip(x.assignment_probs, x.nflId)})
        .reset_index()
    )
    assignment_data_agg.rename(axis=1, mapper={0: "assignment_dict"}, inplace=True)
    final_feature_data = assignment_data_agg.merge(
        acceleration_data.drop_duplicates(["time", "nflId_pr"])
    )
    rusher_encode_map = {
        index: val for index, val in enumerate(set(final_feature_data["nflId_pr"]))
    }
    rusher_encode_inverse_map = {
        rusher_encode_map[index]: index for index in rusher_encode_map
    }
    final_feature_data["rusher_id_model"] = final_feature_data["nflId_pr"].apply(
        lambda x: rusher_encode_inverse_map[x]
    )
    blocker_encode_map = {
        index: val
        for index, val in enumerate(
            set(
                chain.from_iterable(
                    [
                        list(item.keys())
                        for item in final_feature_data["assignment_dict"]
                    ]
                )
            )
        )
    }
    final_feature_data_na = final_feature_data.dropna()
    final_feature_data_na = final_feature_data[
        (~np.isinf(final_feature_data.strain_acceleration))
        & (~np.isinf(final_feature_data.strain_rate))
    ]
    return final_feature_data_na, blocker_encode_map, rusher_encode_map


if __name__ == "__main__":
    generate_acceleration_data()
