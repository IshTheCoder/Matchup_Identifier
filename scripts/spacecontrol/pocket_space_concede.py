"""Space Conceded (blocker dual of Space Generation): in a pass-off, rusher j is handed from a RELEASING
blocker to a RECEIVING blocker; if j wins near-QB space in that window the blockers conceded it. SCG =
conceded by releasing (handed his man off into space), SCR = conceded by receiving (failed to pick up
the stunt). High concession = poor pass-off / stunt pickup, so unlike the rusher metric it should
correlate POSITIVELY with PFF pressures allowed. Also emits the within-unit releaser->receiver flow."""
import pickle, sys
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, "src")
import space_gain as sg

W = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MIN_SNAPS = 50
data, enc = pickle.load(open("pocket_design_8wk.pkl", "rb"))
inv = {v: k for k, v in enc["blocker"].items()}                # blocker slot id -> nflId
players = pd.read_csv("data/players.csv").set_index("nflId")

print(f"computing Space Conceded over {len(data['x_qb'])} frames, w={W} ...", flush=True)
con, prs = sg.compute_pocket_concession(data, w=W, pairs=True)
con["nflId"] = con["bid"].map(inv)
con["name"] = con["nflId"].map(players["displayName"]); con["pos"] = con["nflId"].map(players["officialPosition"])

pff = pd.read_csv("data/pffScoutingData.csv"); pb = pff[pff.pff_role == "Pass Block"].copy()
pb["pa"] = pb[["pff_sackAllowed", "pff_hitAllowed", "pff_hurryAllowed"]].max(axis=1)
bpff = pb.groupby("nflId").agg(pb_snaps=("pa", "size"), press_allowed=("pa", "mean"),
                               beaten=("pff_beatenByDefender", "mean")).reset_index()
g = con.merge(bpff, on="nflId", how="left")
g = g[g.pb_snaps >= MIN_SNAPS].reset_index(drop=True)
g["SCG"] = g.sumSCG / g.pb_snaps; g["SCR"] = g.sumSCR / g.pb_snaps; g["SC"] = g.SCG + g.SCR
g.to_csv("pocket_space_conceded.csv", index=False)

print(f"\nblockers >= {MIN_SNAPS} snaps: {len(g)} | generation==reception total: "
      f"{con.sumSCG.sum():.1f} vs {con.sumSCR.sum():.1f}")
for m in ["SC", "SCG", "SCR"]:
    for tgt in ["press_allowed", "beaten"]:
        d = g[[m, tgt]].dropna()
        print(f"  corr({m:3s}, {tgt:13s}) Pearson={pearsonr(d[m], d[tgt])[0]:+.3f} "
              f"Spearman={spearmanr(d[m], d[tgt])[0]:+.3f}   (expect POSITIVE = conceding is bad)")
g2 = g.dropna(subset=["press_allowed"]).copy()
g2["x"] = g2.SC - g2.groupby("pos").SC.transform("mean")
g2["y"] = g2.press_allowed - g2.groupby("pos").press_allowed.transform("mean")
print(f"  WITHIN-position (SC vs press_allowed): Pearson={pearsonr(g2.x, g2.y)[0]:+.3f}")

print("\n== most space CONCEDED (worst pass-off / stunt pickup), >=50 snaps ==")
print(g.sort_values("SC", ascending=False).head(12)[
    ["name", "pos", "SC", "SCG", "SCR", "press_allowed"]].round(3).to_string(index=False))
print("\n== least space conceded (best) ==")
print(g.sort_values("SC").head(8)[["name", "pos", "SC", "press_allowed"]].round(3).to_string(index=False))

# within-unit directed flow: releaser -> receiver, grouped by offensive unit
asg = pd.read_csv("assignment_data_phase25.csv", usecols=["gameId", "playId", "nflId"]).drop_duplicates()
asg = asg.merge(pd.read_csv("data/plays.csv", usecols=["gameId", "playId", "possessionTeam"]), on=["gameId", "playId"])
team = asg.groupby("nflId")["possessionTeam"].agg(lambda s: s.mode().iat[0])
for who in ["rel", "rec"]:
    nid = prs[f"{who}_bid"].map(inv)
    prs[f"{who}_name"] = nid.map(players["displayName"]); prs[f"{who}_pos"] = nid.map(players["officialPosition"])
    prs[f"{who}_team"] = nid.map(team)
prs = prs[(prs.rel_team == prs.rec_team) & prs.rel_team.notna()]
prs.to_csv("pocket_concede_pairs.csv", index=False)
print("\n== top within-unit releaser -> receiver concession flows (failed hand-offs into space) ==")
for _, r in prs.sort_values("space", ascending=False).head(14).iterrows():
    print(f"  {r.rel_team}  {r.rel_name} ({r.rel_pos}) -> {r.rec_name} ({r.rec_pos}):  {r.space:.1f}")
