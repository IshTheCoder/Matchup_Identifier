"""M0: render the pocket space-control field (offense=blockers vs defense=rushers) over frames of a
collapsing play, to eyeball that influences are forward-skewed and the QB's controlled space caves."""
import sys, pickle
import numpy as np
sys.path.insert(0, "src")
import feature_engineering as fe
import pitch_control as pc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

print("building week-0 pocket design ...", flush=True)
data, enc = fe.build_pocket_design_from_files(weeks=range(1))
pickle.dump((data, enc), open("pocket_design_1wk.pkl", "wb"))
N = len(data["x_qb"]); ps, pr = data["play_start"], data["play_row"]
print(f"  N={N:,} frames  Rmax={data['Rmax']} Bmax={data['Bmax']}", flush=True)

# pick a collapsing play: the one containing the frame where a rusher gets closest to the QB
mind = np.array([data["d_rush"][n][data["rmask"][n] > 0].min() if data["rmask"][n].any() else 99.0
                 for n in range(N)])
n_star = int(np.argmin(mind))
p = int(pr[n_star]); fr = np.arange(ps[p], ps[p + 1])
sel = fr[np.linspace(0, len(fr) - 1, 4).astype(int)]
print(f"  play game {data['gameId'][n_star]} play {data['playId'][n_star]}  "
      f"({len(fr)} frames, min rusher-QB dist {mind[n_star]:.2f} yd)", flush=True)

fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), sharex=False, sharey=False)
for ax, n in zip(axes, sel):
    G, gx, gy, _ = pc.qb_grid(data["x_qb"][n], half=8.0, n=70)
    Ir, ri, Ib, bi, Iqb = pc.frame_fields(G, data, n)
    PC = pc.pitch_control(Ir, Ib, Iqb, include_qb=False)            # P(offense/protection controls p)
    ax.contourf(gx, gy, PC.reshape(gx.shape), levels=np.linspace(0, 1, 21), cmap="RdBu", vmin=0, vmax=1)
    bi_m = data["bmask"][n] > 0; ri_m = data["rmask"][n] > 0
    xb, vb = data["x_blk"][n][bi_m], data["v_blk"][n][bi_m]
    xr, vr = data["x_rush"][n][ri_m], data["v_rush"][n][ri_m]
    ax.scatter(xb[:, 0], xb[:, 1], c="#1a9850", s=55, edgecolor="k", zorder=3, label="blocker")
    ax.scatter(xr[:, 0], xr[:, 1], c="#000000", s=55, edgecolor="w", zorder=3, label="rusher")
    ax.quiver(xb[:, 0], xb[:, 1], vb[:, 0], vb[:, 1], color="#1a9850", scale=40, width=0.006, zorder=4)
    ax.quiver(xr[:, 0], xr[:, 1], vr[:, 0], vr[:, 1], color="k", scale=40, width=0.006, zorder=4)
    ax.scatter(*data["x_qb"][n], marker="*", s=340, c="gold", edgecolor="k", zorder=5)
    ax.set_title(f"frame {int(data['frame_id'][n])}", fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Pocket space control = P(offense controls)  (blue=protected, red=rusher-controlled; "
             "green=blockers, black=rushers, ★=QB; arrows=velocity)", fontsize=12)
fig.tight_layout(); fig.savefig("figures/pocket_control_play.png", dpi=140)
print("wrote figures/pocket_control_play.png", flush=True)
