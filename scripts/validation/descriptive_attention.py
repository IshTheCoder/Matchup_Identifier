"""Descriptive (pre-causal) attention analyses over the Phase-2.5 frame-level assignments.

Two quantities, both per (game, play, frame, rusher):
  dose   n_{j,t} = sum_b theta(b,j,t)   -- effective blockers on rusher j (null dropped)
  strain y_{j,t} = -v/d                 -- per-rusher pressure (fe._strain_per_rusher_frame)

Comp 1  "value of being doubled" (confound exhibit): mean strain by dose bin and the
        pooled vs within-rusher slope of strain on dose. The pooled slope is POSITIVE
        (doubles chase winners) -- the wrong sign for a causal suppression effect.
Comp 2  "value of drawing attention" (spillover): own strain regressed on the attention
        teammates draw, with play x rusher fixed effects -- the within-play, within-rusher
        "a teammate drew the double, I came free" number.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import feature_engineering as fe
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MIN_SNAPS = 50
KEY = ["gameId", "playId", "nflId_pr"]
FKEY = ["gameId", "playId", "frameId", "nflId_pr"]

# ---------------------------------------------------------------- load + strain --
print("loading 8 weeks of smoothed tracking ...", flush=True)
sample = pd.concat([pd.read_csv(f"data/sample_data_week_{i}.csv") for i in range(8)],
                   ignore_index=True)
rf, _ = fe._strain_per_rusher_frame(sample)          # one row per (g,p,frame,rusher)
strain = rf[FKEY + ["strain"]].copy()
print(f"  strain rows: {len(strain):,}", flush=True)
del sample

# ----------------------------------------------------------------- dose --
print("loading Phase-2.5 assignments ...", flush=True)
ad = pd.read_csv("assignment_data_phase25.csv")
ad = ad[ad["nflId_pr"] != -1]                        # drop disengaged/null sentinel
dose = (ad.groupby(FKEY)["assignment_probs"].sum().rename("dose").reset_index())
print(f"  dose rows: {len(dose):,}", flush=True)
del ad

df = strain.merge(dose, on=FKEY, how="inner")        # inner = snap->action window
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["strain", "dose"])
players = pd.read_csv("data/players.csv").set_index("nflId")
df["pos"] = df["nflId_pr"].map(players["officialPosition"])
df["edge"] = np.where(df["pos"].isin(["DE", "OLB"]), "Edge",
              np.where(df["pos"].isin(["DT", "NT"]), "Interior", "Other"))
# snaps per rusher (for min-snap filtering of FE designs)
snaps = df.groupby("nflId_pr")["dose"].size().rename("snaps")
df = df.join(snaps, on="nflId_pr")
print(f"  merged rusher-frames: {len(df):,}  (rushers: {df.nflId_pr.nunique()})", flush=True)


def cluster_ols(y, X, cluster):
    """OLS with cluster-robust (by play) SEs. X includes its own intercept if wanted."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    # cluster meat
    meat = np.zeros((X.shape[1], X.shape[1]))
    cl = pd.Series(cluster).astype("category").cat.codes.to_numpy()
    for c in np.unique(cl):
        m = cl == c
        Xg = X[m]; ug = u[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    G = len(np.unique(cl)); n, k = X.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    V = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    return beta, se


# ============================================================ COMP 1: dose -> strain
print("\n=== COMP 1: value of being doubled (descriptive) ===", flush=True)
# bins
df["bin"] = pd.cut(df["dose"], [0.5, 1.5, 2.5, 9],
                   labels=["Singled (~1)", "Doubled (~2)", "3+"])
tab = (df.dropna(subset=["bin"]).groupby("bin", observed=True)["strain"]
       .agg(mean="mean", median="median", n="size").reset_index())
print(tab.to_string(index=False))
# by edge/interior
tab_pos = (df.dropna(subset=["bin"]).groupby(["edge", "bin"], observed=True)["strain"]
           .mean().reset_index().pivot(index="edge", columns="bin", values="strain"))
print("\nmean strain by group x dose bin:\n", tab_pos.round(4).to_string())

# pooled slope  strain ~ 1 + dose
b_pool, se_pool = cluster_ols(df["strain"], np.c_[np.ones(len(df)), df["dose"]],
                              df["gameId"].astype(str) + "_" + df["playId"].astype(str))
# within-rusher slope: demean strain & dose by rusher, then regress (no intercept)
g = df.groupby("nflId_pr")
df["strain_w"] = df["strain"] - g["strain"].transform("mean")
df["dose_w"] = df["dose"] - g["dose"].transform("mean")
b_wr, se_wr = cluster_ols(df["strain_w"], df["dose_w"].to_numpy()[:, None],
                          df["gameId"].astype(str) + "_" + df["playId"].astype(str))
print(f"\npooled   slope strain~dose: {b_pool[1]:+.4f} (SE {se_pool[1]:.4f})  [intercept {b_pool[0]:+.4f}]")
print(f"within-rusher slope:        {b_wr[0]:+.4f} (SE {se_wr[0]:.4f})")
COMP1 = dict(tab=tab, tab_pos=tab_pos, b_pool=b_pool, se_pool=se_pool, b_wr=b_wr, se_wr=se_wr)

# ---- figure: dose-response (binned mean strain w/ 95% CI) overall + edge/interior
fig, ax = plt.subplots(figsize=(7, 4.2))
order = ["Singled (~1)", "Doubled (~2)", "3+"]
for grp, col in [("Edge", "#1f77b4"), ("Interior", "#d62728")]:
    d = df[df.edge == grp].dropna(subset=["bin"])
    m = d.groupby("bin", observed=True)["strain"].agg(["mean", "sem"]).reindex(order)
    ax.errorbar(range(len(order)), m["mean"], yerr=1.96 * m["sem"], marker="o",
                capsize=3, label=grp, color=col)
allm = df.dropna(subset=["bin"]).groupby("bin", observed=True)["strain"].agg(["mean", "sem"]).reindex(order)
ax.errorbar(range(len(order)), allm["mean"], yerr=1.96 * allm["sem"], marker="s",
            capsize=3, label="All", color="0.3", lw=2)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
ax.set_ylabel("Mean STRAIN"); ax.set_xlabel("Attention dose (effective blockers)")
ax.set_title("Dose-response: more attention, less strain\n"
             "(frame-level suppression; selection attenuates the magnitude)", fontsize=10)
ax.legend(); fig.tight_layout()
fig.savefig("figures/dose_response.png", dpi=160); fig.savefig("figures/dose_response.pdf")
print("wrote figures/dose_response.{png,pdf}")

# ============================================================ COMP 2: spillover (centerpiece)
# Target = the CURRENT rusher's strain; regressor = the attention given to the OTHER rushers
# on the same frame, A_{-k,t} = sum_{j != k} n_{j,t} (total engaged blockers minus own coverage).
print("\n=== COMP 2: value of drawing attention / spillover (descriptive) ===", flush=True)
fr = df.groupby(["gameId", "playId", "frameId"])
df["dose_sum_frame"] = fr["dose"].transform("sum")
df["n_rushers_frame"] = fr["dose"].transform("size")
df["others_attention"] = df["dose_sum_frame"] - df["dose"]   # A_{-k,t}

sp = df[(df["n_rushers_frame"] > 1) & (df["snaps"] >= MIN_SNAPS)].copy()
cl = sp["gameId"].astype(str) + "_" + sp["playId"].astype(str)
# (1) naive cross-sectional: own strain ~ others' attention
b_naive, se_naive = cluster_ols(sp["strain"], np.c_[np.ones(len(sp)), sp["others_attention"]], cl)
# (2) play x rusher FE: demean within (game, play, rusher)
grp = sp.groupby(["gameId", "playId", "nflId_pr"])
sp["y_w"] = sp["strain"] - grp["strain"].transform("mean")
sp["oth_w"] = sp["others_attention"] - grp["others_attention"].transform("mean")
sp["own_w"] = sp["dose"] - grp["dose"].transform("mean")
b_fe, se_fe = cluster_ols(sp["y_w"], sp["oth_w"].to_numpy()[:, None], cl)
# (3) joint FE: own strain ~ others' attention + own coverage (separates spillover vs suppression)
b_joint, se_joint = cluster_ols(sp["y_w"], np.c_[sp["oth_w"], sp["own_w"]], cl)
print(f"naive   own ~ others'-attention:            {b_naive[1]:+.4f} (SE {se_naive[1]:.4f})")
print(f"FE      own ~ others'-attention:            {b_fe[0]:+.4f} (SE {se_fe[0]:.4f})")
print(f"jointFE own ~ others'({b_joint[0]:+.4f} SE {se_joint[0]:.4f}) + own-dose({b_joint[1]:+.4f} SE {se_joint[1]:.4f})")
COMP2 = dict(b_naive=b_naive, se_naive=se_naive, b_fe=b_fe, se_fe=se_fe,
             b_joint=b_joint, se_joint=se_joint, n=len(sp))

# ---- figure: among SINGLED rushers (own coverage held ~fixed), own strain (within-rusher
# demeaned) vs attention on OTHER rushers -- isolates the spillover from own-coverage suppression
single = sp[(sp["dose"] >= 0.75) & (sp["dose"] < 1.5)].copy()
single["obin"] = pd.cut(single["others_attention"], [-0.01, 1, 2, 3, 20],
                        labels=["$<1$", "1--2", "2--3", "$>3$"])
mb = single.dropna(subset=["obin"]).groupby("obin", observed=True)["y_w"].agg(["mean", "sem"])
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.errorbar(range(len(mb)), mb["mean"], yerr=1.96 * mb["sem"], marker="o", capsize=3, color="#2ca02c")
ax.axhline(0, ls="--", color="0.6", lw=1)
ax.set_xticks(range(len(mb))); ax.set_xticklabels(mb.index.tolist())
ax.set_ylabel("Own STRAIN, within-rusher demeaned")
ax.set_xlabel(r"Attention drawn by OTHER rushers  ($\sum_{j\neq k} n_{j,t}$, effective blockers)")
ax.set_title("Spillover (singled rushers): own strain rises as others draw blocking\n"
             "(own coverage held fixed; 'a teammate drew the double, I came free')", fontsize=10)
fig.tight_layout()
fig.savefig("figures/spillover.png", dpi=160); fig.savefig("figures/spillover.pdf")
print(f"wrote figures/spillover.{{png,pdf}}  (singled-rusher frames: {len(single):,})")

# ============================================================ write tables
def w(p, s):
    open(p, "w").write(s); print("wrote", p)

# Comp 1 table
r = COMP1
rows = "\n".join(f"{t['bin']} & {t['mean']:.4f} & {t['median']:.4f} & {int(t['n']):,} \\\\"
                 for _, t in r["tab"].iterrows())
w("tables/dose_response.tex", f"""\\begin{{table}}[h!]
\\centering
\\footnotesize
\\begin{{tabular}}{{lccr}}
\\toprule
Attention dose & Mean STRAIN & Median STRAIN & Rusher-frames \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\caption{{Descriptive dose--response of pressure on attention over the Phase-2.5 frame-level
assignments. Dose is the effective-blocker count $n_{{j,t}}=\\sum_b\\theta(b,j,t)$ on a rusher at a
frame. More attention coincides with \\emph{{less}} strain --- a singled rusher averages
{r['tab']['mean'].iloc[0]:.3f}, a doubled one {r['tab']['mean'].iloc[1]:.3f} --- and the pooled slope of
STRAIN on dose is ${r['b_pool'][1]:+.4f}$ (cluster-robust SE {r['se_pool'][1]:.4f}, clustered by
play). The suppressive effect of extra blockers dominates at the frame level, so unlike a season-level
``doubles chase winners'' correlation the descriptive sign is already correct. But the magnitude is
\\emph{{not}} the causal effect. Part of the pooled slope is a between-rusher artifact (interior
rushers are both doubled more and generate less strain for positional reasons); restricting to within
a rusher (rusher fixed effect) the slope shrinks to ${r['b_wr'][0]:+.4f}$ (SE {r['se_wr'][0]:.4f}).
What remains is biased \\emph{{toward zero}} by selection: a defense commits the second blocker
precisely in the frames a rusher is threatening (a high-strain state), so the doubled frames are
over-sampled from moments whose strain would have been high anyway, masking part of the suppression.
The true causal effect of an added blocker is therefore \\emph{{more}} negative than these descriptive
slopes; recovering its unbiased magnitude would require a causal design (for instance exploiting
stunts and twists as exogenous reallocations of attention), which we leave to future work.}}
\\label{{tab:dose_response}}
\\end{{table}}
""")

# Comp 2 table
c = COMP2
w("tables/spillover.tex", f"""\\begin{{table}}[h!]
\\centering
\\footnotesize
\\begin{{tabular}}{{lcc}}
\\toprule
Specification & Coefficient & Cluster-robust SE \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\emph{{Outcome: the current rusher's STRAIN $y_{{k,t}}$}}}} \\\\
\\quad on attention to others $\\sum_{{j\\neq k}} n_{{j,t}}$ (no FE) & ${c['b_naive'][1]:+.4f}$ & {c['se_naive'][1]:.4f} \\\\
\\quad on attention to others (play$\\times$rusher FE) & ${c['b_fe'][0]:+.4f}$ & {c['se_fe'][0]:.4f} \\\\
\\quad on attention to others, controlling own coverage (FE) & ${c['b_joint'][0]:+.4f}$ & {c['se_joint'][0]:.4f} \\\\
\\quad \\emph{{(own coverage $n_{{k,t}}$ coefficient, same joint fit)}} & ${c['b_joint'][1]:+.4f}$ & {c['se_joint'][1]:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Descriptive spillover: the current rusher's pressure $y_{{k,t}}$ regressed on the attention
the \\emph{{other}} rushers command on the same frame, $A_{{-k,t}}=\\sum_{{j\\neq k}} n_{{j,t}}$
($n={c['n']:,}$ rusher-frames, rushers with $\\geq{MIN_SNAPS}$ snaps; SEs clustered by play). The
play$\\times$rusher fixed effect differences out the rusher's own quality and the play/scheme context,
so identification is frame-to-frame, within-play variation in where the blocking is committed. When
the other rushers draw more attention, the focal rusher generates more strain ($\\beta_1>0$): the
conserved-attention mechanism --- each blocker's assignment sums to one across rushers --- means
attention spent elsewhere is attention removed from the focal rusher, who comes correspondingly
freer. The joint fit separates this spillover from own-coverage suppression: holding the rusher's own
coverage fixed, attention on others still raises his strain (${c['b_joint'][0]:+.4f}$), while his own
coverage suppresses it (${c['b_joint'][1]:+.4f}$) --- the two faces of conserved attention, separately
identified because the number of engaged blockers varies frame to frame. This is the ``a teammate drew
the double, I came free'' effect made quantitative, and the descriptive counterpart of the attention
axis in Figure~\\ref{{fig:rusher_2d}}.}}
\\label{{tab:spillover}}
\\end{{table}}
""")

print("\nDONE.")
