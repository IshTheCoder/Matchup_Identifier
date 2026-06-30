"""S0: build the all-22 downfield space-control design (week 1) and render the control field
(offense=receivers blue / defense=coverage red) across snap->release frames of an example completed
pass, with the downfield region + receiver-controlled "openness" annotated. Eyeball: offense controls
space where a receiver separates; coords normalized so the offense always attacks +x."""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import feature_engineering as fe
import pitch_control as pc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REC_R, COV_R, FWD = 2.5, 3.0, 35.0
print("building all-22 design (week 1) ...", flush=True)
data, enc = fe.build_alltwentytwo_design_from_files(weeks=range(1, 2))
pickle.dump((data, enc), open("alltwentytwo_design_1wk.pkl", "wb"))
N = len(data["x_qb"]); ps = data["play_start"]
print(f"  N={N:,} frames  plays={len(ps)-1}  K(rec)={data['K']} D(cov)={data['D']}  "
      f"receivers={data['N_rec']} coverage={data['N_cov']}", flush=True)

def fields(n):
    los = float(data["los_x"][n])
    G, gx, gy, cell = pc.field_grid(los, forward=FWD, back=3.0, nx=110, ny=70)
    Irec, ri = pc.team_influences(G, data["x_rec"][n], data["v_rec"][n], data["a_rec"][n],
                                  None, data["rec_mask"][n], radius=REC_R)
    Icov, ci = pc.team_influences(G, data["x_cov"][n], data["v_cov"][n], data["a_cov"][n],
                                  None, data["cov_mask"][n], radius=COV_R)
    PC = pc._sigmoid(Irec.sum(0) - Icov.sum(0))               # P(offense controls)
    down = G[:, 0] > los                                       # downfield region
    wrec = Irec.sum(0) / (Irec.sum(0) + Icov.sum(0) + 1e-9)    # receiver control share
    openness = cell * wrec[down].sum()
    return los, gx, gy, PC, openness

# pick an example: a completed pass with many frames
plays = pd.read_csv("data/plays.csv")[["gameId", "playId", "passResult"]]
pk = pd.DataFrame({"gameId": data["gameId"][ps[:-1]], "playId": data["playId"][ps[:-1]],
                   "p": np.arange(len(ps) - 1), "len": np.diff(ps)}).merge(plays, on=["gameId", "playId"])
cand = pk[(pk.passResult == "C") & (pk.len >= 35)].sort_values("len", ascending=False)
p = int(cand.iloc[0]["p"]); fr = np.arange(ps[p], ps[p + 1])
sel = fr[np.linspace(0, len(fr) - 1, 4).astype(int)]
print(f"  example: game {data['gameId'][fr[0]]} play {data['playId'][fr[0]]}  "
      f"({len(fr)} frames, completed)", flush=True)

fig, axes = plt.subplots(1, 4, figsize=(22, 5.0))
for ax, n in zip(axes, sel):
    los, gx, gy, PC, openness = fields(n)
    ax.contourf(gx, gy, PC.reshape(gx.shape), levels=np.linspace(0, 1, 21), cmap="RdBu", vmin=0, vmax=1)
    ax.axvline(los, color="k", ls="--", lw=1, alpha=0.6)
    rm = data["rec_mask"][n] > 0; cm = data["cov_mask"][n] > 0
    xr, vr = data["x_rec"][n][rm], data["v_rec"][n][rm]
    xc, vc = data["x_cov"][n][cm], data["v_cov"][n][cm]
    ax.scatter(xr[:, 0], xr[:, 1], c="#2166ac", s=60, edgecolor="w", zorder=3, label="receiver")
    ax.scatter(xc[:, 0], xc[:, 1], c="#b2182b", s=60, edgecolor="w", zorder=3, label="coverage")
    ax.quiver(xr[:, 0], xr[:, 1], vr[:, 0], vr[:, 1], color="#2166ac", scale=60, width=0.005, zorder=4)
    ax.quiver(xc[:, 0], xc[:, 1], vc[:, 0], vc[:, 1], color="#b2182b", scale=60, width=0.005, zorder=4)
    ax.scatter(*data["x_qb"][n], marker="*", s=320, c="gold", edgecolor="k", zorder=5)
    if np.isfinite(data["x_ball"][n]).all():
        ax.scatter(*data["x_ball"][n], marker="o", s=40, c="saddlebrown", edgecolor="k", zorder=5)
    ax.set_title(f"frame {int(data['frame_id'][n])}   openness={openness:.1f} yd$^2$", fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("All-22 downfield space control = P(offense controls)  (blue=receiver-controlled / "
             "red=covered; ★=QB, ●=ball, dashed=LOS; offense attacks +x; openness = receiver "
             "control integrated downfield)", fontsize=11)
fig.tight_layout(); fig.savefig("figures/alltwentytwo_openness_play.png", dpi=140)
print("wrote figures/alltwentytwo_openness_play.png", flush=True)
