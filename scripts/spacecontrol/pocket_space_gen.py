"""Space Generation Gain (SGG): a pass rusher's credit for near-QB space his TEAMMATES win because
he pulled blocking attention off them (Fernandez-Bornn Wide Open Spaces, Eq. 11). The "drag off a
teammate" is read from the fitted HMM attention theta (no geometric thresholds). Validate against the
HMM front-normalized attention (rusher_2d.csv) -- SGG is the space-control realization of that same
"drawing attention frees teammates" spillover -- and contrast with the rusher's OWN space-won (SOG)."""
import pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import space_gain as sg

W = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MIN_SNAPS = 50
data, enc = pickle.load(open("pocket_design_8wk.pkl", "rb"))
inv = {v: k for k, v in enc["rusher"].items()}                 # slot id -> nflId
players = pd.read_csv("data/players.csv").set_index("nflId")

print(f"computing SGG over {len(data['x_qb'])} frames, window w={W} (theta soft kernel) ...", flush=True)
gen, prs = sg.compute_pocket_generation(data, w=W, pairs=True)
gen["nflId"] = gen["rid"].map(inv)
gen["name"] = gen["nflId"].map(players["displayName"])
gen["pos"] = gen["nflId"].map(players["officialPosition"])

# snaps + own space-won (SOG) + PFF pressure; HMM attention (rusher_2d)
sog = pd.read_csv("pocket_space_gain_rusher.csv")[["rid", "sumSOG", "pff_snaps", "pff_press"]]
att = pd.read_csv("rusher_2d.csv")[["nflId", "attention", "effect"]]
g = gen.merge(sog, on="rid", how="left").merge(att, on="nflId", how="left")
g = g[g["pff_snaps"] >= MIN_SNAPS].reset_index(drop=True)
g["SGG"] = g["sumSGG"] / g["pff_snaps"]                         # per-snap space GENERATED for teammates
g["SR"] = g["sumSR"] / g["pff_snaps"]                           # per-snap space RECEIVED from teammates
g.to_csv("pocket_space_gen.csv", index=False)

print(f"\nrushers >= {MIN_SNAPS} snaps: {len(g)} | generation==reception league total: "
      f"{gen.sumSGG.sum():.1f} vs {gen.sumSR.sum():.1f}")
for m, b, lbl in [("SGG", "attention", "front-norm attention"), ("SGG", "pff_press", "PFF pressure"),
                  ("SGG", "effect", "plus-minus"),
                  ("SR", "attention", "front-norm attention"), ("SR", "pff_press", "PFF pressure"),
                  ("SR", "effect", "plus-minus")]:
    d = g[[m, b]].dropna()
    print(f"  corr({m:3s}, {b:9s})  Pearson {d[m].corr(d[b]):+.3f} / "
          f"Spearman {d[m].corr(d[b], method='spearman'):+.3f}   [{lbl}]")

print("\n== top space GENERATORS (free teammates by drawing blockers) ==")
print(g.sort_values("SGG", ascending=False).head(12)[
    ["name", "pos", "SGG", "attention", "pff_press", "effect"]].round(3).to_string(index=False))
print("\n== top space RECEIVERS (benefit most from teammates' generation) ==")
print(g.sort_values("SR", ascending=False).head(12)[
    ["name", "pos", "SR", "attention", "pff_press", "effect"]].round(3).to_string(index=False))

# ---- within-unit directed flow: who frees whom on the same defensive front ----
asg = pd.read_csv("assignment_data_phase25.csv", usecols=["gameId", "playId", "nflId_pr"]).drop_duplicates()
asg = asg[asg.nflId_pr != -1].merge(
    pd.read_csv("data/plays.csv", usecols=["gameId", "playId", "defensiveTeam"]), on=["gameId", "playId"])
team = asg.groupby("nflId_pr")["defensiveTeam"].agg(lambda s: s.mode().iat[0])   # rusher -> his unit
for who in ["gen", "rec"]:
    nid = prs[f"{who}_rid"].map(inv)
    prs[f"{who}_name"] = nid.map(players["displayName"]); prs[f"{who}_pos"] = nid.map(players["officialPosition"])
    prs[f"{who}_team"] = nid.map(team)
prs = prs[(prs.gen_team == prs.rec_team) & prs.gen_team.notna()]   # same front (sanity)
prs.to_csv("pocket_space_pairs.csv", index=False)

print("\n== top within-unit generator -> receiver flows (who frees whom) ==")
for _, r in prs.sort_values("space", ascending=False).head(18).iterrows():
    print(f"  {r.gen_team}  {r.gen_name} ({r.gen_pos}) -> {r.rec_name} ({r.rec_pos}):  {r.space:.2f}")

# net role within each unit: net = generated - received (>0 net generator, <0 net receiver)
g["net"] = (g.sumSGG - g.sumSR) / g.pff_snaps; g["team"] = g.nflId.map(team)
busiest = (g.assign(vol=g.SGG + g.SR).groupby("team").vol.sum().sort_values(ascending=False).head(3).index)
for tm in busiest:
    u = g[g.team == tm].sort_values("net", ascending=False)
    print(f"\n== {tm} front: net role (generated - received per snap) ==")
    print(u[["name", "pos", "SGG", "SR", "net"]].round(3).to_string(index=False))
