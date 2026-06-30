"""M3 leaderboards: format pocket_rusher_space.csv / pocket_blocker_space.csv into position-grouped
subtables (top-N per position by the space metric, with bootstrap-over-plays 95% CI), matching
tables/rusher_plusminus.tex style. -> tables/pocket_rusher_space.tex, tables/pocket_blocker_space.tex"""
import pandas as pd

def esc(s):
    return str(s).replace("&", "\\&").replace("'", "")

def subtable(df, metric, pos, topn=5):
    d = df[(df.officialPosition == pos)].sort_values(metric, ascending=False).head(topn)
    rows = "".join(f"{esc(r.displayName)} & {r[metric]:.2f} & $[{r.lo:.2f},\\,{r.hi:.2f}]$ \\\\\n"
                   for _, r in d.iterrows())
    return ("\\begin{subtable}{0.48\\textwidth}\n\\centering\n\\footnotesize\n"
            "\\begin{tabular}{lcc}\n\\toprule\nName & %s & 95\\%% CI \\\\\n\\midrule\n%s"
            "\\bottomrule\n\\end{tabular}\n\\caption{%s}\n\\end{subtable}\n" % (metric_label, rows, pos))

def build(csv, metric, positions, minplays, caption, label, out, per_table=4):
    """Chunk the position subtables into separate table floats (<=per_table each) -- a single float
    holding all subtables is taller than a page and LaTeX silently defers/drops it."""
    df = pd.read_csv(csv); df = df[df.plays >= minplays]
    df["officialPosition"] = df["officialPosition"].replace({"DE": "Edge", "OLB": "Edge"})  # merge edge
    blocks = [subtable(df, metric, p) for p in positions if (df.officialPosition == p).any()]
    tex = "%% %s\n" % caption
    for t0 in range(0, len(blocks), per_table):
        chunk = blocks[t0:t0 + per_table]
        body = "".join(blk + ("\\hfill\n" if i % 2 == 0 else "\n\\bigskip\n") for i, blk in enumerate(chunk))
        cap = ("\\caption{%s}\n\\label{%s}\n" % (caption, label)) if t0 + per_table >= len(blocks) else ""
        tex += "\\begin{table}[h!]\n\\centering\n%s%s\\end{table}\n\n" % (body, cap)
    open(out, "w").write(tex); print(f"wrote {out} ({len(blocks)} blocks in {-(-len(blocks)//per_table)} floats)")

metric_label = "Space won"
build("pocket_rusher_space.csv", "space_won", ["Edge", "DT", "NT", "ILB", "MLB", "SS", "FS", "CB"],
      20, "Per-rusher pocket space-won (control share over the QB danger region, yd$^2$; higher = "
      "collapses more of the QB's space), by position, bootstrap 95\\% CI over plays.",
      "tab:pocket_rusher_space", "tables/pocket_rusher_space.tex")
# NOTE: no blocker table -- pocket space control is rusher/defense-sided; the blocker space metrics
# are a within-position null (a one-line limitation in the paper), so we do not publish a leaderboard.
