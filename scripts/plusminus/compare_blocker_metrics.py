"""Compare the CONTINUOUS-TIME first-difference blocker rating (filtered-assignment delta B_b) against
the PLAY-LEVEL blocker metrics (realized impedance fn_value/real_imp, plus-minus effect), all vs the
same PFF pressures-allowed / beaten targets on a common blocker set. Answers 'how much does the
continuous measure improve over play-level?'. Emits a comparison table + a delta leaderboard.
Usage: python3 compare_blocker_metrics.py [delta_rankings.csv]"""
import sys
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

dfile = sys.argv[1] if len(sys.argv) > 1 else "blocker_dose_rankings_filtered_baseline_mcmc.csv"
DESIGN = "continuous_design_delta_filtered_8wk.pkl"   # the design the dose model was fit on
delta = pd.read_csv(dfile).rename(columns={"effect": "delta"})          # dose B_b + 95% CI + PFF targets + pos/snaps
imp = pd.read_csv("blocker_value.csv")[["nflId", "real_imp", "fn_value", "fn_lo", "fn_hi"]]
pm = pd.read_csv("blocker_rankings_phase25.csv")[["nflId", "effect", "hdi_lo", "hdi_hi"]].rename(columns={"effect": "plusminus"})


def _total_dose(path=DESIGN):
    """per-blocker total dose sum_{j,t} theta(b,j,t) from the fitted design -> DataFrame(nflId, tot_dose)

    The dose model's B_b is a per-unit-dose effect, so it is not comparable across blockers who
    engage at different rates (the same reason the play-level coefficient is reported as realized
    impedance). Summing the dose a blocker actually absorbed over all rusher-frames gives the
    exposure to multiply it by; since sum_j theta(b,j,t) <= 1, this is his engaged-frame count.
    Padded blocker slots carry dose 0 and so contribute nothing.
    """
    import pickle
    d, enc = pickle.load(open(path, "rb"))
    dose, bid = np.asarray(d["assignment"]), np.asarray(d["blocker_ids"])
    tot = np.zeros(int(d["N_blockers"]))
    np.add.at(tot, bid.ravel(), dose.ravel())
    inv = {v: k for k, v in enc["blocker"].items()}
    return pd.DataFrame({"nflId": [inv[i] for i in range(len(tot))], "tot_dose": tot})


m = delta.merge(imp, on="nflId", how="inner").merge(pm, on="nflId", how="left")
m = m.merge(_total_dose(), on="nflId", how="left")
m = m[(m.pb_snaps >= 50)].dropna(subset=["press_allowed"]).reset_index(drop=True)
# realized dose: strain deceleration actually delivered per snap. Same units as the play-level
# realized impedance (STRAIN removed per snap), so the two leaderboards are on a common scale.
m["dose_real"] = m.delta * m.tot_dose / m.pb_snaps
print(f"common blockers (>=50 PFF snaps): {len(m)}  (delta source: {dfile})\n", flush=True)

# all four ratings are oriented 'higher = better blocker' -> expect NEGATIVE vs pressures/beaten
METRICS = [("Continuous ($\\theta_t$)", "delta"), ("Realized impedance (FN)", "fn_value"),
           ("Realized impedance", "real_imp"), ("Play-level plus-minus", "plusminus")]
TGT = [("pressures allowed", "press_allowed"), ("beaten", "beaten")]
rows = []
print(f"{'metric':28s} " + "  ".join(f"{t[0]:>20s}" for t in TGT), flush=True)
for label, col in METRICS:
    cells = []
    for _, tcol in TGT:
        d = m[[col, tcol]].dropna(); r, p = pearsonr(d[col], d[tcol]); rho, _ = spearmanr(d[col], d[tcol])
        cells.append((r, rho, len(d), p))
    rows.append((label, col, cells))
    print(f"{label:28s} " + "  ".join(f"r={c[0]:+.3f} rho={c[1]:+.3f}" for c in cells), flush=True)

# do continuous and play-level AGREE? (are they measuring the same thing or complementary?)
print(f"\nagreement: corr(delta, fn_value)={pearsonr(m.delta, m.fn_value)[0]:+.3f}  "
      f"corr(delta, plus-minus)={pearsonr(m.delta.values, m.plusminus.fillna(m.plusminus.mean()))[0]:+.3f}", flush=True)
# within-position (the harder test) for each
print("\nWITHIN-position Pearson vs pressures-allowed (demeaned by pos):", flush=True)
for label, col in METRICS:
    d = m[[col, "press_allowed", "pos"]].dropna().copy()
    d["x"] = d[col] - d.groupby("pos")[col].transform("mean")
    d["y"] = d.press_allowed - d.groupby("pos").press_allowed.transform("mean")
    print(f"  {label:28s} {pearsonr(d.x, d.y)[0]:+.3f}", flush=True)

# CI-width sanity check: continuous (1.1M frame-obs) should have TIGHTER posteriors than the play-level
# models (~8.5k plays). Compare median 95%-CI width, and the width normalized by the between-blocker SD
# of the estimate (a unitless precision ratio, since the three metrics live on different scales).
print("\nCI-width sanity check (95% CI half-...; tighter = more data/better identified):", flush=True)
ci = [("Continuous dose", "delta", "lo", "hi"), ("Realized impedance (FN)", "fn_value", "fn_lo", "fn_hi"),
      ("Play-level plus-minus", "plusminus", "hdi_lo", "hdi_hi")]
for label, col, lo, hi in ci:
    d = m[[col, lo, hi]].dropna(); w = (d[hi] - d[lo]); sd = d[col].std()
    print(f"  {label:24s} median CI width={w.median():.4f}  | normalized (width/SD)={w.median()/sd:.3f}", flush=True)

# within-position press Pearson per metric (for the table)
wp = {}
for label, col in METRICS:
    d = m[[col, "press_allowed", "pos"]].dropna().copy()
    d["x"] = d[col] - d.groupby("pos")[col].transform("mean")
    d["y"] = d.press_allowed - d.groupby("pos").press_allowed.transform("mean")
    wp[col] = pearsonr(d.x, d.y)[0]

# ---- comparison table (raw vs press/beaten + within-position press) ----
# Symbol for each rating, as the manuscript defines it. The continuous-time effect is B^Delta_b, NOT
# B_b: it is a different parameter on a different scale (per-frame deceleration vs per-play peak
# suppression), and the two would otherwise be indistinguishable in adjacent rows of this table.
SYM = {"delta": "$B^{\\Delta}_b$", "fn_value": "$\\mathrm{FN}_b$",
       "real_imp": "$\\mathrm{Imp}_b$", "plusminus": "$B_b$"}   # keys match METRICS above
body = "".join(
    f"{lab} & {SYM[col]} & {c[0][0]:+.3f} & {c[0][1]:+.3f} & {c[1][0]:+.3f} & {c[1][1]:+.3f} & {wp[col]:+.3f} \\\\\n"
    for lab, col, c in rows)
open("tables/blocker_continuous_vs_play.tex", "w").write(
    "% Continuous-time (delta STRAIN, filtered assignments) vs play-level blocker metrics, all vs PFF.\n"
    "\\begin{table}[h!]\n\\centering\n\\footnotesize\n\\begin{tabular}{llccccc}\n\\toprule\n"
    " & & \\multicolumn{2}{c}{vs pressures allowed} & \\multicolumn{2}{c}{vs beaten} & within-pos. \\\\\n"
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\nBlocker rating & Symbol & Pearson & Spearman & Pearson & Spearman & press. (P) \\\\\n"
    "\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n\\caption{Blocker ratings against independent "
    "PFF charting on a common set of " + str(len(m)) + " linemen ($\\geq 50$ snaps). The Symbol column "
    "gives the quantity being correlated: the continuous-time per-frame effect $B^{\\Delta}_b$ of "
    "Section~\\ref{continuous}, the front-normalized and raw realized impedance $\\mathrm{FN}_b$ and "
    "$\\mathrm{Imp}_b$ of Eqs.~\\ref{eq:imp}--\\ref{eq:fn}, and the play-level plus-minus effect $B_b$ "
    "of Section~\\ref{plusminus}. All ratings are "
    "oriented higher${=}$better, so a correct association is negative. The final column demeans both "
    "the rating and the outcome within position group, which removes the systematic differences in "
    "pressure allowed between tackles, guards, and centers.}\n"
    "\\label{tab:blocker_cont_vs_play}\n\\end{table}\n")
print("\nwrote tables/blocker_continuous_vs_play.tex", flush=True)

# ---- delta leaderboard (top by B_b, >=100 snaps) ----
top = m[m.pb_snaps >= 100].sort_values("delta", ascending=False).head(12)
lb = "".join(f"{r['name']} & {r.pos} & {r.delta:+.3f} & $[{r.lo:+.3f},\\,{r.hi:+.3f}]$ & "
             f"{r.dose_real:+.3f} & {r.press_allowed:.2f} \\\\\n"
             for _, r in top.iterrows())
open("tables/blocker_delta_leaders.tex", "w").write(
    "% Top linemen by continuous-time first-difference rating (filtered-assignment delta B_b).\n"
    "\\begin{table}[h!]\n\\centering\n\\footnotesize\n\\begin{tabular}{llcccc}\n\\toprule\n"
    "Name & Pos & $B_b$ & 95\\% CI & Real. & PFF press.\\ all. \\\\\n\\midrule\n" + lb +
    "\\bottomrule\n\\end{tabular}\n\\caption{Top pass blockers by the continuous-time rating "
    "$B_b$ --- the deceleration of an assigned rusher's strain per unit blocking $\\theta(b,j,t)$, "
    "conditioning on the time-$t$ state; $\\geq 100$ snaps. Because $B_b$ is a per-unit-assignment "
    "effect, we also report the realized deceleration it delivers per snap (Real.), "
    "$B_b\\sum_{j,t}\\theta(b,j,t)$ averaged over the blocker's snaps. This is the frame-level "
    "counterpart of the play-level realized impedance of Table~\\ref{tab:blocker_value}}"
    "\n\\label{tab:blocker_delta_leaders}\n\\end{table}\n")
print("wrote tables/blocker_delta_leaders.tex", flush=True)

# realized dose vs raw B_b: does weighting by engagement change the ranking or the validation?
t100 = m[m.pb_snaps >= 100]
print(f"\nrealized dose (B_b x dose per snap): median={m.dose_real.median():.3f} "
      f"(play-level realized impedance median={m.real_imp.median():.3f}) | "
      f"Spearman(raw, realized) over {len(t100)} blockers >=100 snaps = "
      f"{spearmanr(t100.delta, t100.dose_real)[0]:.3f}", flush=True)
for tlabel, tcol in TGT:
    print(f"  vs {tlabel:18s} raw r={pearsonr(m.delta, m[tcol])[0]:+.3f}   "
          f"realized r={pearsonr(m.dose_real, m[tcol])[0]:+.3f}", flush=True)
