"""tables/pocket_concede_flow.tex: within-unit releaser->receiver space-conceded flow (the blocker-side
dual of the rusher generation flow). Who hands a rusher off into space on the same offensive line."""
import pandas as pd

def esc(s):
    return str(s).replace("&", "\\&").replace("'", "")

p = pd.read_csv("pocket_concede_pairs.csv").sort_values("space", ascending=False).head(12)
rows = "".join(
    f"{r.rel_team} & {esc(r.rel_name)} ({r.rel_pos}) & {esc(r.rec_name)} ({r.rec_pos}) & {r.space:.1f} \\\\\n"
    for _, r in p.iterrows())
open("tables/pocket_concede_flow.tex", "w").write(
    "% Within-unit releaser->receiver space-conceded flow (blocker dual of the generation flow).\n"
    "\\begin{table}[h!]\n\\centering\n\\footnotesize\n\\begin{tabular}{lllc}\n\\toprule\n"
    "Unit & Releaser & Receiver & Conceded \\\\\n\\midrule\n" + rows +
    "\\bottomrule\n\\end{tabular}\n\\caption{The strongest within-front \\emph{space-conceded} flows: for "
    "each pair of teammates, the value-weighted near-QB space a rusher wins as his block is handed from a "
    "\\emph{releasing} blocker to a \\emph{receiving} blocker (a pass-off), summed over eight weeks --- the "
    "blocker-side dual of the generation flow (Table~\\ref{tab:pocket_pair_flow}). These are the "
    "offensive-line pairs whose hand-offs on stunts and twists most often leak penetration (guard-to-"
    "tackle and guard-to-center exchanges dominate), a coordination failure the discrete per-play "
    "charting tag cannot localize.}\n\\label{tab:pocket_concede_flow}\n\\end{table}\n")
print("wrote tables/pocket_concede_flow.tex")
print(p[["rel_team", "rel_name", "rec_name", "space"]].to_string(index=False))
