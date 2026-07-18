"""Two dimensions of pass rush: attention commanded (effective blockers drawn,
sum_b theta(b,j)) vs the plus-minus effect R_j (strain generated, adjusted for blocking)."""
import pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
data, enc = pickle.load(open("play_design_phase25.pkl","rb"))
s = pickle.load(open("play_model_samples_phase25.pkl","rb"))
players = pd.read_csv("data/players.csv").set_index("nflId")
inv = {v:k for k,v in enc["rusher"].items()}
asg = np.asarray(data["assignment"]); mask = np.asarray(data["mask"]).astype(bool)
rid = np.asarray(data["rusher_ids"])
att = asg.sum(2)                                   # (N,R) effective blockers drawn per (play,rusher)
# front-normalize: attention relative to the average rusher on the SAME play, so it is
# comparable across 3-, 4-, and 5-man fronts (matches "Norm. Att." in the attention tables).
pmean = np.array([att[i, mask[i]].mean() if mask[i].any() else np.nan for i in range(att.shape[0])])
rel = np.where(mask & (pmean[:,None] > 0), att / pmean[:,None], np.nan)  # (N,R)
R = (s["rusher_weight"]@data["rusher_covariates"].T + s["sigma_rusher"][:,None]*s["z_rusher"]).mean(0)
# aggregate front-normalized attention per global rusher idx over valid (masked) rows
n=int(data["N_rushers"]); tot=np.zeros(n); cnt=np.zeros(n)
valid = mask & np.isfinite(rel)
np.add.at(tot, rid[valid], rel[valid]); np.add.at(cnt, rid[valid], 1.0)
g=pd.DataFrame({"idx":np.arange(n)})
g["attention"]=np.where(cnt>0, tot/np.maximum(cnt,1), np.nan); g["snaps"]=cnt.astype(int)
g["effect"]=R; g["nflId"]=g.idx.map(inv)
g["name"]=g.nflId.map(players["displayName"]); g["pos"]=g.nflId.map(players["officialPosition"]).replace({"DE":"Edge","OLB":"Edge"})
g=g[(g.snaps>=50)].dropna(subset=["attention"]).reset_index(drop=True)
g.to_csv("rusher_2d.csv", index=False)
print(f"rushers: {len(g)}  attention range [{g.attention.min():.2f},{g.attention.max():.2f}]  effect [{g.effect.min():.2f},{g.effect.max():.2f}]")
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
ax.set_ylabel("Plus-minus effect $R_j$  (strain generated, adjusted for blocking)")
ax.set_title("Two dimensions of pass rush: pressure generated vs. blocking attention drawn")
ax.legend(loc="lower right",fontsize=8)
fig.tight_layout(); fig.savefig("figures/rusher_2d.png",dpi=160); fig.savefig("figures/rusher_2d.pdf")
print("corr(attention, effect):", round(g.attention.corr(g.effect),3))
print("\nhigh-both (draw doubles AND generate pressure):")
print(g[(g.attention>mx)&(g.effect>my)].nlargest(8,"effect")[["name","pos","attention","effect"]].round(3).to_string(index=False))
print("\nhigh attention, low effect (command doubles, less conversion):")
print(g[(g.attention>mx)&(g.effect<my)].nlargest(6,"attention")[["name","pos","attention","effect"]].round(3).to_string(index=False))
print("\nlow attention, high effect (win without extra attention):")
print(g[(g.attention<mx)&(g.effect>my)].nlargest(6,"effect")[["name","pos","attention","effect"]].round(3).to_string(index=False))
