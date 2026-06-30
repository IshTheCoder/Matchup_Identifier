import numpy as np, pandas as pd
NULL=-1
print("loading assignment_data.csv ...", flush=True)
ad = pd.read_csv("assignment_data.csv",
     usecols=["gameId","playId","frameId","nflId","nflId_pr","assignment_probs"],
     dtype={"assignment_probs":"float32"})
players = pd.read_csv("data/players.csv").set_index("nflId")
pos = players["officialPosition"]; name = players["displayName"]

# 1) sanity: per (play,frame,blocker) probs (incl null) sum to 1
s = ad.groupby(["gameId","playId","frameId","nflId"])["assignment_probs"].sum()
print(f"[sanity] per-blocker-frame prob sum: mean={s.mean():.4f} min={s.min():.4f} max={s.max():.4f}")

nullrows = ad[ad.nflId_pr==NULL].copy()           # one per (frame,blocker)
nf = len(nullrows)
print(f"\n[overall] blocker-frames={nf:,}  mean null mass={nullrows.assignment_probs.mean():.3f}  "
      f"frac null>0.5={(nullrows.assignment_probs>0.5).mean():.3f}  frac null>0.9={(nullrows.assignment_probs>0.9).mean():.3f}")

# 2) per blocker position
nullrows["pos"]=nullrows.nflId.map(pos)
pp = nullrows.groupby("pos")["assignment_probs"].agg(mean_null="mean", n="size").sort_values("mean_null",ascending=False)
print("\n[null by blocker position]"); print(pp.to_string())

# 3) null over the course of the play (normalized frame index)
g = nullrows.groupby(["gameId","playId"])
nullrows["fr_rank"]=g["frameId"].rank(method="first")-1
nullrows["fr_n"]=g["frameId"].transform("size")
nullrows["phase"]=np.clip((nullrows.fr_rank/np.maximum(nullrows.fr_n-1,1)*5).astype(int),0,4)
print("\n[null mass by play phase (0=snap .. 4=throw)]")
print(nullrows.groupby("phase")["assignment_probs"].mean().round(3).to_string())

# 4) tackle -> null shed: per tackle, mean null + shed events (null crosses 0.5 upward)
T = nullrows[nullrows.pos=="T"].sort_values(["gameId","playId","nflId","fr_rank"])
def sheds(x): 
    a=(x.to_numpy()>0.5).astype(int); return int(((a[1:]-a[:-1])==1).sum())
grp = T.groupby(["gameId","playId","nflId"])
shed = grp["assignment_probs"].apply(sheds).rename("sheds").reset_index()
meann = grp["assignment_probs"].mean().rename("mean_null").reset_index()
pl = shed.merge(meann,on=["gameId","playId","nflId"])
byT = pl.groupby("nflId").agg(snaps=("sheds","size"),sheds=("sheds","sum"),mean_null=("mean_null","mean")).reset_index()
byT=byT[byT.snaps>=50]; byT["shed_rate"]=byT.sheds/byT.snaps; byT["name"]=byT.nflId.map(name)
print(f"\n[tackles >=50 snaps: {len(byT)}]  shed_rate=sheds/play")
print("most shed-prone:"); print(byT.sort_values("shed_rate",ascending=False).head(8)[["name","shed_rate","mean_null","snaps"]].to_string(index=False))
print("least shed-prone:"); print(byT.sort_values("shed_rate").head(8)[["name","shed_rate","mean_null","snaps"]].to_string(index=False))
