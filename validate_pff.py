"""Validate model-derived metrics against PFF hand-charting:
(A) blocker disengagement / rusher shedding vs PFF pressures-allowed / pressures,
(B) HMM assignment probabilities vs PFF blocker responsibility (pff_nflIdBlockedPlayer)."""
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
MIN = 50

pff = pd.read_csv("data/pffScoutingData.csv")
pb = pff[pff.pff_role == "Pass Block"].copy()
pr = pff[pff.pff_role == "Pass Rush"].copy()
pb["pressure_allowed"] = pb[["pff_sackAllowed","pff_hitAllowed","pff_hurryAllowed"]].max(axis=1)
pr["pressure"] = pr[["pff_sack","pff_hit","pff_hurry"]].max(axis=1)
# per-player PFF rates
blk_pff = pb.groupby("nflId").agg(pff_snaps=("pressure_allowed","size"),
          pff_press_allowed=("pressure_allowed","mean"),
          pff_sack_allowed=("pff_sackAllowed","mean"),
          pff_beaten=("pff_beatenByDefender","mean")).reset_index()
rsh_pff = pr.groupby("nflId").agg(pff_snaps=("pressure","size"),
          pff_press=("pressure","mean"), pff_sack=("pff_sack","mean"),
          pff_hurry=("pff_hurry","mean")).reset_index()

def corr(df, a, b):
    d = df[[a, b]].dropna()
    if len(d) < 10: return (np.nan, np.nan, len(d))
    return (pearsonr(d[a], d[b])[0], spearmanr(d[a], d[b])[0], len(d))

print("="*70)
print("(A1) BLOCKER metrics vs PFF pressures ALLOWED")
sb = pd.read_csv("survival_by_blocker.csv")       # beat_rate, rmst_s, drive_win_rate, ...
bv = pd.read_csv("blocker_value.csv")[["nflId","real_imp","fn_value","mean_eng","B_coef"]]
b = sb.merge(bv, on="nflId", how="outer").merge(blk_pff, on="nflId", how="inner")
b = b[b.pff_snaps >= MIN]
print(f"  n blockers (>= {MIN} PFF snaps): {len(b)}")
for m in ["beat_rate","rmst_s","real_imp","fn_value","mean_eng"]:
    if m in b:
        for tgt in ["pff_press_allowed","pff_sack_allowed","pff_beaten"]:
            r,rho,n = corr(b, m, tgt); print(f"    {m:>10} vs {tgt:<18} Pearson {r:+.3f}  Spearman {rho:+.3f}  (n={n})")

print("="*70)
print("(A2) RUSHER metrics vs PFF pressures OBTAINED")
sr = pd.read_csv("survival_by_rusher.csv")[["nflId","beat_rate","rmst_s"]].rename(columns={"beat_rate":"shed_rate"})
ru = pd.read_csv("rusher_rankings_phase25.csv")[["nflId","effect"]]
r_ = sr.merge(ru, on="nflId", how="outer").merge(rsh_pff, on="nflId", how="inner")
r_ = r_[r_.pff_snaps >= MIN]
print(f"  n rushers (>= {MIN} PFF snaps): {len(r_)}")
for m in ["shed_rate","rmst_s","effect"]:
    if m in r_:
        for tgt in ["pff_press","pff_sack","pff_hurry"]:
            rr,rho,n = corr(r_, m, tgt); print(f"    {m:>10} vs {tgt:<10} Pearson {rr:+.3f}  Spearman {rho:+.3f}  (n={n})")

print("="*70)
print("(B) HMM assignment probs vs PFF blocker responsibility")
ad = pd.read_csv("assignment_data.csv", usecols=["gameId","playId","nflId","nflId_pr","assignment_probs"])
ad = ad[ad.nflId_pr != -1]                                 # drop null sentinel
tav = ad.groupby(["gameId","playId","nflId","nflId_pr"], as_index=False)["assignment_probs"].mean()  # time-avg theta
resp = pb[["gameId","playId","nflId","pff_nflIdBlockedPlayer"]].dropna()
resp["pff_nflIdBlockedPlayer"] = resp["pff_nflIdBlockedPlayer"].astype(int)
# theta our model places on the PFF-charted rusher
m = tav.merge(resp, left_on=["gameId","playId","nflId","nflId_pr"],
              right_on=["gameId","playId","nflId","pff_nflIdBlockedPlayer"], how="inner")
theta_on_pff = m["assignment_probs"]
# argmax agreement: per (g,p,blocker) the most-assigned rusher vs PFF rusher
arg = tav.sort_values("assignment_probs").drop_duplicates(["gameId","playId","nflId"], keep="last")
arg = arg.merge(resp, on=["gameId","playId","nflId"], how="inner")
agree = (arg["nflId_pr"] == arg["pff_nflIdBlockedPlayer"]).mean()
K = tav.groupby(["gameId","playId"])["nflId_pr"].nunique().mean()
print(f"  pairs with a PFF responsibility tag: {len(resp):,}; matched in assignments: {len(m):,}")
print(f"  mean time-avg theta on the PFF-charted rusher: {theta_on_pff.mean():.3f}  (median {theta_on_pff.median():.3f})")
print(f"  chance baseline (1/avg #rushers): {1/K:.3f}")
print(f"  argmax-rusher agreement with PFF responsibility: {agree:.3f}")
