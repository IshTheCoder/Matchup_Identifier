"""Front-normalized realized blocker impedance (presentation metric; model stays the
unnormalized total-impedance sum). Per (play, blocker): imp = B_b * sum_j theta(b,j)
= strain removed that play. Front-normalize per play relative to the blockers on the
same play (controls for 4- vs 5-man front / usage). Aggregate per blocker."""
import pickle, numpy as np, pandas as pd
data, enc = pickle.load(open("play_design_phase25.pkl", "rb"))      # unnormalized phase2.5
s = pickle.load(open("play_model_samples_phase25.pkl", "rb"))
players = pd.read_csv("data/players.csv").set_index("nflId")
MIN_SNAPS = 50
inv = {v: k for k, v in enc["blocker"].items()}                      # global idx -> nflId

B = (s["blocker_weight"] @ data["blocker_covariates"].T + s["sigma_blocker"][:, None] * s["z_blocker"]).mean(0)
asg = np.asarray(data["assignment"])            # (N,R,B) theta_bar
bidx = np.asarray(data["blocker_ids"])          # (N,B) global blocker index
mask = np.asarray(data["mask"])                 # (N,R)
eng = asg.sum(1)                                 # (N,B) engagement = sum_j theta(b,j)
present = eng > 0
front = mask.sum(1).astype(int)
imp = eng * B[bidx]                              # (N,B) realized impedance per (play, blocker)

rows = []
for n in range(asg.shape[0]):
    bl = np.where(present[n])[0]
    if len(bl) == 0:
        continue
    mbar = imp[n, bl].mean()
    for b in bl:
        rows.append((int(bidx[n, b]), int(front[n]), float(eng[n, b]),
                     float(imp[n, b]), float(imp[n, b] - mbar)))
df = pd.DataFrame(rows, columns=["bidx", "front", "engagement", "impedance", "rel_impedance"])
g = df.groupby("bidx").agg(snaps=("impedance", "size"), mean_eng=("engagement", "mean"),
                           real_imp=("impedance", "mean"), fn_value=("rel_impedance", "mean"))
g["B_coef"] = [B[i] for i in g.index]
g["nflId"] = [inv[i] for i in g.index]
g["name"] = g["nflId"].map(players["displayName"])
g["pos"] = g["nflId"].map(players["officialPosition"])
g = g[g.snaps >= MIN_SNAPS].reset_index(drop=True)
g.to_csv("blocker_value.csv", index=False)
print(f"blockers >= {MIN_SNAPS} snaps: {len(g)}")
print("Spearman(front-norm value vs raw B coef):",
      round(g.fn_value.corr(g.B_coef, method="spearman"), 3))
for pos in ["T", "G", "C"]:
    d = g[g.pos == pos].sort_values("fn_value", ascending=False)
    print(f"\n== {pos}: top by front-normalized realized impedance ==")
    print(d.head(6)[["name", "fn_value", "real_imp", "B_coef", "mean_eng", "snaps"]].round(3).to_string(index=False))
print("\n== lowest-value blockers (low-engagement releasers expected) ==")
print(g.sort_values("fn_value").head(8)[["name", "pos", "fn_value", "real_imp", "B_coef", "mean_eng"]].round(3).to_string(index=False))
