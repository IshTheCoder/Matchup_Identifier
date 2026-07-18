"""Generate the dynamic space gain/loss tables for the paper:
 - tables/pocket_sog_rusher.tex: rusher Space Occupation Gain leaderboard (Bornn Table-1 style).
 - tables/pocket_sol_position.tex: lineman Space Occupation Loss by position (the structural finding
   that SOL grows C<G<T and tracks pressures-allowed between, but not within, position)."""
import pandas as pd

def esc(s):
    return str(s).replace("&", "\\&").replace("'", "")

# ---- rusher SOG leaderboard (top 15 by total gain) ----
r = pd.read_csv("pocket_space_gain_rusher.csv")
r = r[(r.frames >= 200) & (r.pff_snaps >= 50)].sort_values("sumSOG", ascending=False).head(15)
rows = "".join(f"{esc(x.displayName)} & {x.officialPosition} & {x.sumSOG:.0f} & {x.muSOG:.3f} & "
               f"{100*x.active:.0f}\\% & {x.pff_press:.2f} \\\\\n" for _, x in r.iterrows())
open("tables/pocket_sog_rusher.tex", "w").write(
    "% Rusher Space Occupation Gain leaderboard (Bornn SOG analog).\n"
    "\\begin{table}[h!]\n\\centering\n\\footnotesize\n\\begin{tabular}{llcccc}\n\\toprule\n"
    "Name & Pos & $\\sum$SOG & $\\mu$SOG & Active & PFF press \\\\\n\\midrule\n" + rows +
    "\\bottomrule\n\\end{tabular}\n\\caption{Top pass rushers by total pocket Space Occupation Gain "
    "($\\sum$SOG, yd$^2$ of value-weighted near-QB control won), with mean per-gaining-frame intensity "
    "($\\mu$SOG), the share of gain earned at running pace (Active), and the PFF pressure rate. The "
    "gain-based ranking surfaces established rushers and correlates with PFF pressure at $+0.36$ "
    "($+0.40$ among edge/interior DL).}\n\\label{tab:pocket_sog_rusher}\n\\end{table}\n")
print("wrote tables/pocket_sog_rusher.tex")

# ---- lineman SOL by position (structural finding) ----
b = pd.read_csv("pocket_space_lost_blocker.csv")
b = b[(b.frames >= 400) & (b.pb_snaps >= 50)].dropna(subset=["press_allowed"])
g = b[b.officialPosition.isin(["C", "G", "T"])].groupby("officialPosition")
tab = g.agg(n=("rate", "size"), rate=("rate", "mean"), pa=("press_allowed", "mean")).reindex(["C", "G", "T"])
rows = "".join(f"{p} & {int(x.n)} & {1000*x.rate:.1f} & {x.pa:.3f} \\\\\n" for p, x in tab.iterrows())
open("tables/pocket_sol_position.tex", "w").write(
    "% Lineman Space Occupation Loss by position (structural, not individual-skill).\n"
    "\\begin{table}[h!]\n\\centering\n\\footnotesize\n\\begin{tabular}{lccc}\n\\toprule\n"
    "Pos & $n$ & SOL rate ($\\times 10^{3}$) & PFF press.\\ allowed \\\\\n\\midrule\n" + rows +
    "\\bottomrule\n\\end{tabular}\n\\caption{Lineman pocket Space Occupation Loss (per-frame rate of "
    "value-weighted near-QB control conceded) by position. SOL grows monotonically from centers to "
    "guards to tackles and tracks pressures allowed \\emph{between} positions ($+0.28$), but the "
    "within-position correlation with pressures allowed is $\\approx 0$: SOL measures the positional "
    "difficulty of the assignment (tackles defend more open, contested space), not individual "
    "protection skill.}\n\\label{tab:pocket_sol_position}\n\\end{table}\n")
print("wrote tables/pocket_sol_position.tex")
