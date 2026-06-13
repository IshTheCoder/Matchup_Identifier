"""Generate tables/ (.tex) and figures/ from the HMM assignments + fitted params:
  - attention rankings per rusher position group (top/bottom), normalized per N-man front
  - pass-blocker assignment-entropy ranking (switchiness), normalized per N-man front
  - a figure of the tau parameters (where blockers stand on the QB->rusher line)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

MIN_SNAPS = 50
players = pd.read_csv("data/players.csv").set_index("nflId")
pos, name = players["officialPosition"], players["displayName"]

print("loading assignment_data ...", flush=True)
ad = pd.read_csv("assignment_data.csv")
ad["front"] = ad.groupby(["gameId", "playId"])["nflId_pr"].transform("nunique")  # N-man front

# ---------------------------------------------------------------- attention --
# attention drawn by rusher = sum over blockers of assignment prob (effective blockers)
att = ad.groupby(["gameId", "playId", "frameId", "nflId_pr"], as_index=False).agg(
    attention=("assignment_probs", "sum"), front=("front", "first"))
# normalize per front: relative to the average rusher on the same play-frame
att["rel"] = att["attention"] / att.groupby(["gameId", "playId", "frameId"])["attention"].transform("mean")
# per (rusher, play) then per rusher
rp = att.groupby(["nflId_pr", "gameId", "playId"]).agg(
    attention=("attention", "mean"), rel=("rel", "mean")).reset_index()
rush = rp.groupby("nflId_pr").agg(
    attention=("attention", "mean"), rel=("rel", "mean"), snaps=("attention", "size")).reset_index()
rush["pos"] = rush["nflId_pr"].map(pos)
rush["name"] = rush["nflId_pr"].map(name)
rush = rush[rush["snaps"] >= MIN_SNAPS]

# ----------------------------------------------------------------- entropy --
# blocker's marginal assignment distribution over rushers (mean theta over frames);
# entropy measures how much a blocker spreads / switches across rushers.
bk = ad.groupby(["gameId", "playId", "nflId", "nflId_pr"], as_index=False)["assignment_probs"].mean()
bk["plogp"] = np.where(bk["assignment_probs"] > 0,
                       bk["assignment_probs"] * np.log(bk["assignment_probs"]), 0.0)
grp = bk.groupby(["gameId", "playId", "nflId"]).agg(H=("plogp", "sum"), N=("assignment_probs", "size"))
grp["Hnorm"] = np.where(grp["N"] > 1, (-grp["H"]) / np.log(grp["N"].clip(lower=2)), 0.0)  # normalize per front
blk = grp.groupby("nflId")["Hnorm"].agg(entropy="mean", snaps="size").reset_index()
blk["pos"] = blk["nflId"].map(pos)
blk["name"] = blk["nflId"].map(name)
blk = blk[blk["snaps"] >= MIN_SNAPS]


def _rows(df, cols, fmt):
    return " \\\\\n".join(" & ".join(fmt[c](r[c]) for c in cols) for _, r in df.iterrows()) + " \\\\"


# ------------------------------------------------ attention table (.tex) --
RUSH_GROUPS = ["DE", "DT", "NT", "OLB"]


def _subtable(df, cap):
    body = _rows(df[["name", "rel"]], ["name", "rel"],
                 {"name": lambda x: str(x), "rel": lambda x: f"{x:.2f}"})
    # \resizebox scales the (long-name) 4-across tabular to its 0.24\textwidth box so the
    # subtable row never overflows \textwidth regardless of name lengths.
    return ("\\begin{subtable}{0.24\\textwidth}\n\\centering\n\\resizebox{\\linewidth}{!}{%\n"
            "\\begin{tabular}{lc}\n"
            "\\toprule\nName & Norm. Att. \\\\\n\\midrule\n" + body +
            f"\n\\bottomrule\n\\end{{tabular}}}}\n\\caption{{{cap}}}\n\\end{{subtable}}")


def _front_table(rank_top, label, caption):
    subs = []
    for g in RUSH_GROUPS:
        d = rush[rush["pos"] == g].sort_values("rel", ascending=not rank_top).head(5)
        subs.append(_subtable(d, g))
    return ("\\begin{table}[h!]\n\\centering\n" + "\n\\hfill\n".join(subs) +
            f"\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


with open("tables/attention_rankings.tex", "w") as f:
    f.write("% Attention = effective blockers drawn by a rusher, normalized per play to\n"
            "% the average rusher (so it is comparable across 3-, 4-, and 5-man fronts).\n")
    f.write(_front_table(True, "tab:att_top_norm",
            f"Top 5 pass rushers by front-normalized attention, by position (min {MIN_SNAPS} snaps)."))
    f.write("\n")
    f.write(_front_table(False, "tab:att_bot_norm",
            f"Bottom 5 pass rushers by front-normalized attention, by position (min {MIN_SNAPS} snaps)."))

# ------------------------------------------------- entropy table (.tex) --
def _ent_block(df):
    return _rows(df[["name", "pos", "entropy"]], ["name", "pos", "entropy"],
                 {"name": str, "pos": str, "entropy": lambda x: f"{x:.3f}"})


top = blk.sort_values("entropy", ascending=False).head(10)
bot = blk.sort_values("entropy", ascending=True).head(10)
with open("tables/blocker_entropy.tex", "w") as f:
    f.write("% Normalized assignment entropy of a pass blocker: how much they spread /\n"
            "% switch coverage across rushers (1 = uniform over the front, 0 = one rusher).\n")
    f.write("\\begin{table}[h!]\n\\centering\n")
    for d, cap in [(top, "Highest (most switching)"), (bot, "Lowest (stickiest)")]:
        f.write("\\begin{subtable}{0.48\\textwidth}\n\\centering\n\\footnotesize\n\\begin{tabular}{llc}\n"
                "\\toprule\nName & Pos & Norm. Entropy \\\\\n\\midrule\n"
                + _ent_block(d) + "\n\\bottomrule\n\\end{tabular}\n"
                f"\\caption{{{cap}}}\n\\end{{subtable}}\n\\hfill\n")
    f.write("\\caption{Pass blockers ranked by front-normalized assignment entropy "
            "(min " + str(MIN_SNAPS) + " snaps).}\n"
            "\\label{tab:blocker_entropy}\n\\end{table}\n")

# ----------------------------------------------------- tau figure --
fp = pd.read_csv("fitted_params_jax.csv")
fp["tau_r"] = fp["tau"].map(lambda s: float(np.fromstring(str(s).replace("\n", " ").strip().strip("[]"), sep=" ")[0]))
fp = fp.sort_values("tau_r").reset_index(drop=True)
labels = {"T": "Tackle", "G": "Guard", "C": "Center", "TE": "Tight End",
          "RB": "Running Back", "FB": "Fullback", "WR": "Wide Receiver"}

# multi-level stagger + leader lines so the tight T/G/C/TE/FB cluster stays legible
levels = [0.32, -0.32, 0.52, -0.52, 0.72, -0.72]
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.axhline(0, color="0.6", lw=2, zorder=1)
ax.scatter(fp["tau_r"], np.zeros(len(fp)), s=110, color="#1f5fb4", zorder=3)
for i, (_, r) in enumerate(fp.iterrows()):
    y = levels[i % len(levels)]
    ax.annotate(f"{labels.get(r['position'], r['position'])} ({r['tau_r']:.2f})",
                (r["tau_r"], 0), (r["tau_r"], y), ha="center",
                va="bottom" if y > 0 else "top", fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8))
ax.text(0, -0.95, "Quarterback", ha="center", va="top", fontsize=11, fontweight="bold")
ax.text(1, -0.95, "Rusher", ha="center", va="top", fontsize=11, fontweight="bold")
ax.set_xlim(-0.08, 1.08); ax.set_ylim(-1.05, 1.0)
ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel(r"$\tau_r$: position along the QB$\rightarrow$rusher line")
for s in ["top", "left", "right"]:
    ax.spines[s].set_visible(False)
ax.set_title(r"Where each blocking group stands between the quarterback and its rusher", fontsize=12)
fig.tight_layout()
fig.savefig("figures/tau_positions.png", dpi=200)
fig.savefig("figures/tau_positions.pdf")
print("wrote tables/attention_rankings.tex, tables/blocker_entropy.tex, figures/tau_positions.{png,pdf}")
print(f"\nrushers>= {MIN_SNAPS} snaps: {len(rush)} | blockers: {len(blk)}")
print("top-3 switchiest blockers:", top.head(3)[["name", "pos", "entropy"]].values.tolist())
