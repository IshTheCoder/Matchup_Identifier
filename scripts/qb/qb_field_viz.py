"""Render the spatial pressure field (sum of per-rusher anisotropic bumps) over the pocket for
the most-pressured play, and rank rushers by the force they exert on the QB (field decomposition).
Uses the best spatial-field fit (qb_spatial_best.pkl)."""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import feature_engineering as fe
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

print("rebuilding week-0 design (with rusher ids) ...", flush=True)
data, enc = fe.build_qb_force_design_from_files(weeks=range(1))
pickle.dump((data, enc), open("qb_force_design_wk0.pkl", "wb"))
best = pickle.load(open("qb_spatial_best.pkl", "rb"))
s, profile = best["samples"], best["profile"]
P = {k: float(np.asarray(s[k]).mean()) for k in ["l_perp", "l_par", "c0", "c_a", "c_int"]}
ip2, ia2 = 1 / P["l_perp"] ** 2, 1 / P["l_par"] ** 2
softplus = lambda x: np.logaddexp(0.0, x)

uhat, dist, rvel = data["uhat"], data["dist"], data["rvel"]
aclose, is_int, rmask = data["aclose"], data["is_interior"], data["rmask"]
xqb = data["x_qb"]
xk = xqb[:, None, :] - dist[..., None] * uhat            # (N,R,2) rusher positions
vhat = rvel / (np.linalg.norm(rvel, axis=-1, keepdims=True) + 0.5)
q = softplus(P["c0"] + P["c_a"] * np.maximum(aclose, 0) + P["c_int"] * is_int) * rmask  # (N,R) charge

# per-rusher force on the QB (gauss profile): F = q * psi * Sigma^-1 d
d = dist[..., None] * uhat
proj = (d * vhat).sum(-1)
Sinv_d = ip2 * d + (ia2 - ip2) * proj[..., None] * vhat
quad = ip2 * (d * d).sum(-1) + (ia2 - ip2) * proj ** 2
psi = np.exp(-0.5 * quad) if profile == "gauss" else np.exp(-np.sqrt(quad + 1e-6))
if profile == "exp":
    Sinv_d = Sinv_d / np.sqrt(quad + 1e-6)[..., None]
F = (q * psi)[..., None] * Sinv_d                        # (N,R,2) per-rusher force on QB
Fmag = np.linalg.norm(F, axis=-1) * rmask               # (N,R)
Ftot = F.sum(1)                                         # (N,2) total predicted force

# =============================================== (1) per-rusher attribution
print("\n=== per-rusher attribution: mean force on QB (yd/s^2 of push generated) ===", flush=True)
players = pd.read_csv("data/players.csv").set_index("nflId")
inv = {v: k for k, v in enc["rusher"].items()}
gid = data["rusher_slot_id"]
nG = len(enc["rusher"])
tot = np.zeros(nG); cnt = np.zeros(nG)
np.add.at(tot, gid[rmask > 0], Fmag[rmask > 0]); np.add.at(cnt, gid[rmask > 0], 1.0)
att = pd.DataFrame({"gid": np.arange(nG), "force": np.where(cnt > 0, tot / np.maximum(cnt, 1), 0), "frames": cnt})
att["nflId"] = att["gid"].map(inv)
att["name"] = att["nflId"].map(players["displayName"]); att["pos"] = att["nflId"].map(players["officialPosition"])
att = att[att["frames"] >= 50].sort_values("force", ascending=False)
att.to_csv("qb_force_attribution_wk0.csv", index=False)
print(att.head(12)[["name", "pos", "force", "frames"]].round(3).to_string(index=False))

# =============================================== (2) render the pressure field
n_star = int(np.argmax(np.linalg.norm(Ftot, axis=1)))   # most-pressured frame
p = int(data["play_row"][n_star]); ps = data["play_start"]
fr = np.arange(ps[p], ps[p + 1])                        # frames of that play
sel = fr[np.linspace(0, len(fr) - 1, 4).astype(int)]    # 4 frames across the play
g, pid = data["gameId"][n_star], data["playId"][n_star]
print(f"\nrendering pressure field: game {g} play {pid}  ({len(fr)} frames)", flush=True)

# grid bounds over the play
pts = np.concatenate([xqb[fr], xk[fr][rmask[fr] > 0]], axis=0)
x0, y0 = pts.min(0) - 3; x1, y1 = pts.max(0) + 3
gx, gy = np.meshgrid(np.linspace(x0, x1, 140), np.linspace(y0, y1, 100))
G = np.stack([gx.ravel(), gy.ravel()], 1)               # (Gpts,2)

fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), sharex=True, sharey=True)
for ax, n in zip(axes, sel):
    m = rmask[n] > 0
    Phi = np.zeros(len(G))
    for k in np.where(m)[0]:
        dk = G - xk[n, k]
        pj = dk @ vhat[n, k]
        qd = ip2 * (dk * dk).sum(1) + (ia2 - ip2) * pj ** 2
        Phi += q[n, k] * (np.exp(-0.5 * qd) if profile == "gauss" else np.exp(-np.sqrt(qd + 1e-6)))
    ax.contourf(gx, gy, Phi.reshape(gx.shape), levels=18, cmap="inferno")
    col = np.where(is_int[n, m] > 0, "#39d0ff", "#7CFC00")
    ax.scatter(xk[n, m, 0], xk[n, m, 1], c=col, s=60, edgecolor="k", zorder=3)
    ax.quiver(xk[n, m, 0], xk[n, m, 1], rvel[n, m, 0], rvel[n, m, 1],
              color="w", scale=40, width=0.006, zorder=4)              # rusher velocity
    ax.scatter(*xqb[n], marker="*", s=320, c="yellow", edgecolor="k", zorder=5)
    ax.quiver(*xqb[n], *Ftot[n], color="red", scale=30, width=0.012, zorder=6)  # predicted push
    ax.set_title(f"frame {int(data['frame_id'][n])}", fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("QB pressure field = sum of per-rusher anisotropic pressure bumps  "
             "(cyan=interior, green=edge; white=rusher velocity; star=QB, red=push on QB)", fontsize=12)
fig.tight_layout(); fig.savefig("figures/qb_field_play.png", dpi=140)
print("wrote figures/qb_field_play.png, qb_force_attribution_wk0.csv")
