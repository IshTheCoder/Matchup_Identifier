"""Validate the escape-rate rusher rating (u_j, release-hazard charge) against independent PFF
pass-rush charting (pressures / sacks / hits / hurries), benchmarked vs the plus-minus effect."""
import os
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
MIN = 50

pff = pd.read_csv("data/pffScoutingData.csv")
pr = pff[pff.pff_role == "Pass Rush"].copy()
pr["pressure"] = pr[["pff_sack", "pff_hit", "pff_hurry"]].max(axis=1)
rsh_pff = pr.groupby("nflId").agg(pff_snaps=("pressure", "size"), pff_press=("pressure", "mean"),
          pff_sack=("pff_sack", "mean"), pff_hit=("pff_hit", "mean"), pff_hurry=("pff_hurry", "mean")).reset_index()

esc = pd.read_csv("qb_escape_rating_8wk.csv")[["nflId", "u", "frames", "pos"]].rename(columns={"u": "escape_u"})
pm_file = "rusher_rankings_phase25.csv" if os.path.exists("rusher_rankings_phase25.csv") else "rusher_rankings.csv"
pm = pd.read_csv(pm_file)[["nflId", "effect"]].rename(columns={"effect": "plusminus"})
sh = (pd.read_csv("survival_by_rusher.csv")[["nflId", "beat_rate"]].rename(columns={"beat_rate": "shed_rate"})
      if os.path.exists("survival_by_rusher.csv") else None)

m = esc.merge(rsh_pff, on="nflId", how="inner").merge(pm, on="nflId", how="left")
if sh is not None:
    m = m.merge(sh, on="nflId", how="left")
m = m[m.pff_snaps >= MIN].reset_index(drop=True)
edge_int = m[m.pos.isin(["DE", "OLB", "DT", "NT"])]
print(f"n rushers (>= {MIN} PFF snaps): {len(m)}   (true edge/interior DL: {len(edge_int)})\n")


def corr(df, a, b):
    d = df[[a, b]].dropna()
    if len(d) < 10:
        return np.nan, np.nan, len(d), np.nan
    r, p = pearsonr(d[a], d[b]); rho, _ = spearmanr(d[a], d[b])
    return r, rho, len(d), p


def block(df, label):
    print(f"--- {label} ---")
    metrics = ["escape_u", "plusminus"] + (["shed_rate"] if "shed_rate" in df else [])
    for metric in metrics:
        for tgt in ["pff_press", "pff_sack", "pff_hurry", "pff_hit"]:
            r, rho, n, p = corr(df, metric, tgt)
            star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else ""))
            print(f"  {metric:>10} vs {tgt:<10} Pearson {r:+.3f}{star:<3} Spearman {rho:+.3f}  (n={n})")
        print()


block(m, "ALL pass rushers (incl. DB/LB blitzers)")
block(edge_int, "Edge + interior DL only (DE/OLB/DT/NT)")
m.sort_values("escape_u", ascending=False).to_csv("qb_escape_vs_pff.csv", index=False)
print("wrote qb_escape_vs_pff.csv")
