"""Tables for the opponent-adjusted block-hold model: a position-grouped leaderboard (hold rating +
95% CI + expected hold time vs an average rusher) and matchup hold-times (vs average vs elite edge).
Expected hold = discrete-survival sum over the dwell hazard, capped at the ~4s play window."""
import pickle
import numpy as np, pandas as pd

DT, KMAX = 0.1, 40
ev = pd.read_csv("block_failure_events.csv")
d = np.maximum(ev.time.to_numpy(int), 1)
fr = np.arange(d.sum()) - np.repeat(np.cumsum(d) - d, d) + 1
frm, frs = fr.mean(), fr.std()                                   # same standardization as the fit
s = pickle.load(open("block_hold_discrete_samples.pkl", "rb"))
a, g1, g2, beta = (float(s[k].mean()) for k in ("alpha", "g1", "g2", "beta_opp"))
sig = lambda x: 1 / (1 + np.exp(-x))

def beat_prob(u_b, R):
    """P(block beaten within the ~4s play window) under the discrete dwell hazard, percent."""
    k = np.arange(1, KMAX + 1); ts = (k - frm) / frs
    h = sig(a + g1 * ts + g2 * ts ** 2 + beta * R + u_b)
    return 100.0 * (1.0 - np.prod(1.0 - h))

R_ELITE = 1.5                                                    # ~top-quartile standardized rusher effect
hold = pd.read_csv("block_hold_discrete_ratings.csv")
hold["u_b"] = -hold["hold"]
hold["beat_avg"] = [beat_prob(u, 0.0) for u in hold.u_b]
hold["beat_elite"] = [beat_prob(u, R_ELITE) for u in hold.u_b]
q = hold[(hold.spells >= 40) & hold.pos.isin(["T", "G", "C"])].dropna(subset=["press_allowed"])

def esc(s):
    return str(s).replace("'", "")

def subtable(pos, n=5):
    dd = q[q.pos == pos].sort_values("hold", ascending=False).head(n)
    rows = "".join(f"{esc(r['name'])} & {r.hold:+.2f} & $[{r.lo:+.2f},\\,{r.hi:+.2f}]$ & {r.beat_avg:.0f}\\% & {r.beat_elite:.0f}\\% \\\\\n"
                   for _, r in dd.iterrows())
    return ("\\begin{subtable}{0.49\\textwidth}\n\\centering\n\\footnotesize\n\\begin{tabular}{lcccc}\n"
            "\\toprule\nName & Hold & 95\\% CI & beat\\% & beat\\%$_{e}$ \\\\\n\\midrule\n"
            + rows + "\\bottomrule\n\\end{tabular}\n\\caption{%s}\n\\end{subtable}\n" % pos)

tex = ("% Opponent-adjusted block-hold rating (discrete-time failure hazard) by position.\n"
       "\\begin{table}[h!]\n\\centering\n" + subtable("T") + "\\hfill\n" + subtable("G") + "\n\\bigskip\n"
       + subtable("C") + "\n\\caption{Top pass protectors by the opponent-adjusted block-hold rating "
       "(higher $=$ holds longer NET of the rushers faced), with its $95\\%$ credible interval and the "
       "implied probability the block is beaten within the $4$\\,s window against an \\emph{average} "
       "rusher (beat\\%) and an \\emph{elite} edge rusher (beat\\%$_e$). The opponent term moves elite "
       "tackles who face the best rushers (e.g.\\ Trent Williams) from the bottom of the marginal "
       "hazard ranking to the top.}\n\\label{tab:block_hold}\n\\end{table}\n")
open("tables/block_hold.tex", "w").write(tex)
print("wrote tables/block_hold.tex")
print(f"\nbeat-probability illustration (within play, avg rusher -> elite edge):")
for _, r in q.sort_values("hold", ascending=False).head(6).iterrows():
    print(f"  {r['name']:22s} {r.pos}  {r.beat_avg:.0f}% -> {r.beat_elite:.0f}%   (hold={r.hold:+.2f})")
print(f"  vs a WORST-ranked tackle:")
for _, r in q[q.pos=='T'].sort_values('hold').head(3).iterrows():
    print(f"  {r['name']:22s} {r.pos}  {r.beat_avg:.0f}% -> {r.beat_elite:.0f}%   (hold={r.hold:+.2f})")
