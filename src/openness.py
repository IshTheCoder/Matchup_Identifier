"""Receiver OPENNESS from the all-22 space-control field (validated S1: per-receiver openness tracks
raw separation at Pearson +0.83; argmax-realizable-openness identifies the target 2.6x over chance).
A receiver's openness = how uncontested his CATCH POINT (lead point x+lead*v) is versus COVERAGE ONLY
(teammates don't cover you): c_i = 1/(1 + sum_d I_d(catch_i)), using the skewed coverage influence
(a trailing defender's suppressed forward reach => receiver more open). Realizable = c_i * throwability
exp(-|catch_i - QB|/D0). Per-frame summaries feed the QB release model (build_qb_force_design)."""
import numpy as np, pandas as pd
import pitch_control as pc


def frame_openness_table(data, cov_r=3.0, d0=15.0, lead=0.5):
    """Per-(gameId,playId,frameId) openness from an all-22 design (build_alltwentytwo_design).
    Returns DataFrame: open_cmax (best uncovered catch point), open_real (best realizable), open_sum
    (total realizable across receivers)."""
    N = len(data["x_qb"])
    cmax = np.zeros(N); real = np.zeros(N); osum = np.zeros(N)
    for n in range(N):
        ri = np.where(data["rec_mask"][n] > 0)[0]; ci = np.where(data["cov_mask"][n] > 0)[0]
        if ri.size == 0 or ci.size == 0:
            continue
        centers = data["x_rec"][n][ri] + lead * data["v_rec"][n][ri]
        Id = np.stack([pc.influence(centers, data["x_cov"][n][d], data["v_cov"][n][d],
                                    data["a_cov"][n][d], 0.0, radius=cov_r) for d in ci])
        c = 1.0 / (1.0 + Id.sum(0))
        dq = np.hypot(centers[:, 0] - data["x_qb"][n, 0], centers[:, 1] - data["x_qb"][n, 1])
        r = c * np.exp(-dq / d0)
        cmax[n] = c.max(); real[n] = r.max(); osum[n] = r.sum()
    return pd.DataFrame({"gameId": data["gameId"], "playId": data["playId"],
                         "frameId": data["frame_id"], "open_cmax": cmax,
                         "open_real": real, "open_sum": osum})
