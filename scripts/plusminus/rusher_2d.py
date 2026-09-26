"""Two dimensions of pass rush: attention commanded (effective blockers drawn,
sum_b theta(b,j)) vs the continuous-time rusher effect R^Delta_j (per-frame STRAIN
acceleration generated, adjusted for the blocking faced). R^Delta_j is read from the dose
model's export (model_outputs/dose_rusher_baseline_summary.parquet, run_blocker_dose_model.py);
attention is computed from the smoothed phase-2.5 assignments."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "src")
import model_io as mio

MIN_SNAPS = 50
players = pd.read_csv("data/players.csv").set_index("nflId")

ad = pd.read_csv("assignment_data_phase25.csv")
ad = ad[ad.nflId_pr >= 0]                          # drop the disengaged (null) state
# effective blockers drawn per (play, frame, rusher), then time-averaged over the play
att = ad.groupby(["gameId", "playId", "frameId", "nflId_pr"], as_index=False)["assignment_probs"].sum()
rp = att.groupby(["gameId", "playId", "nflId_pr"], as_index=False)["assignment_probs"].mean()
# front-normalize: attention relative to the average rusher on the SAME play, so it is
# comparable across 3-, 4-, and 5-man fronts
pmean = rp.groupby(["gameId", "playId"])["assignment_probs"].transform("mean")
rp["rel"] = np.where(pmean > 0, rp["assignment_probs"] / pmean, np.nan)
g = rp.dropna(subset=["rel"]).groupby("nflId_pr").agg(
    attention=("rel", "mean"), snaps=("rel", "size")).reset_index().rename(columns={"nflId_pr": "nflId"})

R = pd.read_parquet(f"{mio.MODEL_DIR}/dose_rusher_baseline_summary.parquet")[["nflId", "mean", "lo", "hi"]]
g = g.merge(R.rename(columns={"mean": "effect"}), on="nflId", how="inner")
g["name"] = g.nflId.map(players["displayName"])
g["pos"] = g.nflId.map(players["officialPosition"]).replace({"DE": "Edge", "OLB": "Edge"})
g = g[g.snaps >= MIN_SNAPS].reset_index(drop=True)
g.to_csv("rusher_2d.csv", index=False)
print(f"rushers: {len(g)}  attention range [{g.attention.min():.2f},{g.attention.max():.2f}]  "
      f"effect [{g.effect.min():.4f},{g.effect.max():.4f}]")
mx, my = g.attention.median(), g.effect.median()
fig,ax=plt.subplots(figsize=(9,7))
groups={"Edge":"#1f77b4","DT":"#d62728","NT":"#2ca02c"}
for pos,c in groups.items():
    d=g[g.pos==pos]; ax.scatter(d.attention,d.effect,s=22,alpha=.55,color=c,label=pos)
other=g[~g.pos.isin(groups)]; ax.scatter(other.attention,other.effect,s=14,alpha=.3,color="0.6",label="other")
ax.axvline(mx,ls="--",color="0.7",lw=1); ax.axhline(my,ls="--",color="0.7",lw=1)
# annotate extremes: top by effect, top by attention, and high-both
lab=set()
for _,r in pd.concat([g.nlargest(6,"effect"), g.nlargest(6,"attention"),
                      g[(g.attention>mx)&(g.effect>my)].nlargest(6,"effect"+"")]).iterrows():
    if r["name"] in lab: continue
    lab.add(r["name"]); ax.annotate(r["name"],(r.attention,r.effect),fontsize=7,
        xytext=(3,3),textcoords="offset points")
ax.set_xlabel("Attention commanded  (front-normalized: blockers drawn vs. average rusher on the play)")
ax.set_ylabel(r"Continuous-time rusher effect $R^{\Delta}_j$  (STRAIN acceleration, adjusted for blocking)")
ax.set_title("Two dimensions of pass rush: pressure generated vs. blocking attention drawn")
ax.legend(loc="lower right",fontsize=8)
fig.tight_layout(); fig.savefig("figures/rusher_2d.png",dpi=160); fig.savefig("figures/rusher_2d.pdf")
print("corr(attention, effect):", round(g.attention.corr(g.effect),3))
print("\nhigh-both (draw doubles AND generate pressure):")
print(g[(g.attention>mx)&(g.effect>my)].nlargest(8,"effect")[["name","pos","attention","effect"]].round(4).to_string(index=False))
print("\nhigh attention, low effect (command doubles, less conversion):")
print(g[(g.attention>mx)&(g.effect<my)].nlargest(6,"attention")[["name","pos","attention","effect"]].round(4).to_string(index=False))
print("\nlow attention, high effect (win without extra attention):")
print(g[(g.attention<mx)&(g.effect>my)].nlargest(6,"effect")[["name","pos","attention","effect"]].round(4).to_string(index=False))
