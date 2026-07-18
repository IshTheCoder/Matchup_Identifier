"""Pocket space-generation tables (read saved CSVs; no recompute):
 - tables/pocket_pair_flow.tex (MAIN TEXT): the within-unit directed flow --- who generates space for
   whom on the same defensive front (Bornn space generation -> reception, theta as the soft drag kernel).
 - tables/pocket_sgg.tex (APPENDIX): per-rusher Space Generation Gain leaderboard."""
import pandas as pd

def esc(s):
    return str(s).replace("&", "\\&").replace("'", "")

# -------------------------------------------- within-unit pair flow (main text) --
p = pd.read_csv("pocket_space_pairs.csv").sort_values("space", ascending=False).head(12)
p[["gen_pos", "rec_pos"]] = p[["gen_pos", "rec_pos"]].replace({"DE": "Edge", "OLB": "Edge"})  # merge edge
rows = "".join(
    f"{r.gen_team} & {esc(r.gen_name)} ({r.gen_pos}) & {esc(r.rec_name)} ({r.rec_pos}) & "
    f"{r.space:.1f} \\\\\n" for _, r in p.iterrows())
open("tables/pocket_pair_flow.tex", "w").write(
    "% Within-unit directed space-generation flow (generator -> receiver on the same front).\n"
    "\\begin{table}[h!]\n\\centering\n\\footnotesize\n\\begin{tabular}{lllc}\n\\toprule\n"
    "Unit & Generator & Receiver & Space \\\\\n\\midrule\n" + rows +
    "\\bottomrule\n\\end{tabular}\n\\caption{The strongest within-front space-generation flows: for each "
    "pair of teammates, the value-weighted near-QB space one rusher (the \\emph{generator}) frees for "
    "the other (the \\emph{receiver}) by drawing a blocker's assignment away, summed over eight weeks. "
    "The directed flow recovers recognizable pairings --- a double-team magnet springing his teammate "
    "(Aaron Donald freeing his nose tackle; Dalvin Tomlinson freeing Danielle Hunter off the edge; Maxx "
    "Crosby opening the interior) --- the same generation and reception of \\citet{fernandez2018wide}, "
    "here attributed pair-by-pair through the assignment probabilities.}\n"
    "\\label{tab:pocket_pair_flow}\n\\end{table}\n")
print("wrote tables/pocket_pair_flow.tex")

# ------------------------------------------------ per-rusher SGG leaderboard (appendix) --
g = pd.read_csv("pocket_space_gen.csv")
g["pos"] = g["pos"].replace({"DE": "Edge", "OLB": "Edge"})     # merge edge rushers
g = g.sort_values("SGG", ascending=False).head(15)
rows = "".join(
    f"{esc(x['name'])} & {x['pos']} & {x['SGG']:.3f} & {x['attention']:.2f} & "
    f"{x['pff_press']:.2f} & {x['effect']:+.2f} \\\\\n" for _, x in g.iterrows())
open("tables/pocket_sgg.tex", "w").write(
    "% Per-rusher Space Generation Gain leaderboard (assignment-weighted Bornn space generation).\n"
    "\\begin{table}[h!]\n\\centering\n\\footnotesize\n\\begin{tabular}{llcccc}\n\\toprule\n"
    "Name & Pos & SGG & Att. & PFF press & $+/-$ \\\\\n\\midrule\n" + rows +
    "\\bottomrule\n\\end{tabular}\n\\caption{Top pass rushers by Space Generation Gain (SGG): the "
    "value-weighted near-QB space won by teammates they freed by drawing a blocker's assignment away, "
    "per pass snap. Att.\\ is the rusher's front-normalized blocking attention, PFF press his own "
    "pressure rate, and $+/-$ his plus-minus effect. The leading generators are interior linemen who "
    "command double teams --- high attention but low personal pressure and plus-minus --- isolating "
    "the complementary, team-benefit dimension of pass rushing.}\n\\label{tab:pocket_sgg}\n\\end{table}\n")
print("wrote tables/pocket_sgg.tex")
print(p[["gen_team", "gen_name", "rec_name", "space"]].to_string(index=False))
