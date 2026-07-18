"""Containment-break impulse response (reduced-form spatiotemporal causal check).

EVENT: a rusher "breaks containment" at frame t0 when his blocker sheds him -- he goes
from engaged (attention dose >= ENG_HI) to free (dose <= ENG_LO) on the phase-2.5 assignments.

QUESTION (local projection / Jorda impulse response): does STRAIN rise in the frames AFTER a
break, for (a) the breaking rusher himself (own_break -> own strain; near-mechanical), and
(b) his teammates (teammate_break -> own strain; the QB-mediated spillover -- the QB is forced
toward the other rushers)?  For each horizon h we regress a rusher's strain at t+h on the
break indicator at t, controlling for his baseline strain and dose at t, with play fixed
effects and SEs clustered by play. Negative horizons are placebo leads: a clean causal read
requires beta_h ~ 0 for h<0 (no pre-trend) and beta_h > 0 building after h=0.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import feature_engineering as fe
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ENG_HI, ENG_LO = 0.75, 0.50     # engaged vs free thresholds (dose = sum_b theta(b,j,t))
HMIN, HMAX = -5, 12             # impulse-response horizons in frames (0.1 s each)
FKEY = ["gameId", "playId", "frameId", "nflId_pr"]
RKEY = ["gameId", "playId", "nflId_pr"]

# ----------------------------------------------------------------- load strain + dose --
print("loading 8 weeks tracking + phase-2.5 assignments ...", flush=True)
sample = pd.concat([pd.read_csv(f"data/sample_data_week_{i}.csv") for i in range(8)],
                   ignore_index=True)
rf, _ = fe._strain_per_rusher_frame(sample)
strain = rf[FKEY + ["strain"]].copy()
del sample
ad = pd.read_csv("assignment_data_phase25.csv")
ad = ad[ad["nflId_pr"] != -1]
dose = ad.groupby(FKEY)["assignment_probs"].sum().rename("dose").reset_index()
del ad
df = strain.merge(dose, on=FKEY, how="inner").replace([np.inf, -np.inf], np.nan).dropna()
df = df.sort_values(RKEY + ["frameId"]).reset_index(drop=True)
print(f"  rusher-frames: {len(df):,}", flush=True)

# ----------------------------------------------------------------- define break events --
g = df.groupby(RKEY)
dose_prev = g["dose"].shift(1)
frame_prev = g["frameId"].shift(1)
contig = (df["frameId"] - frame_prev == 1)
df["own_break"] = ((dose_prev >= ENG_HI) & (df["dose"] <= ENG_LO) & contig).astype(float)
# at least one OTHER rusher breaks at the same frame on the same play
fr = df.groupby(["gameId", "playId", "frameId"])
df["n_break_frame"] = fr["own_break"].transform("sum")
df["n_rushers_frame"] = fr["dose"].transform("size")
df["teammate_break"] = ((df["n_break_frame"] - df["own_break"] >= 1) & (df["n_rushers_frame"] > 1)).astype(float)
print(f"  own-break events: {int(df['own_break'].sum()):,} | "
      f"rusher-frames with a teammate breaking: {int(df['teammate_break'].sum()):,}")


def cluster_ols(y, X, cl):
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtXi = np.linalg.inv(X.T @ X); beta = XtXi @ (X.T @ y); u = y - X @ beta
    code = pd.Series(cl).astype("category").cat.codes.to_numpy()
    meat = np.zeros((X.shape[1],) * 2)
    for c in np.unique(code):
        s = X[code == c].T @ u[code == c]; meat += np.outer(s, s)
    G = len(np.unique(code)); n, k = X.shape
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * XtXi @ meat @ XtXi
    return beta, np.sqrt(np.diag(V))


def impulse(treat_col):
    """local projection: strain_{t+h} ~ treat_t + strain_t + dose_t, play FE, cluster by play."""
    out = []
    gg = df.groupby(RKEY)
    for h in range(HMIN, HMAX + 1):
        y_lead = gg["strain"].shift(-h)
        f_lead = gg["frameId"].shift(-h)
        ok = (f_lead - df["frameId"] == h) & y_lead.notna()
        d = df.loc[ok, [treat_col, "strain", "dose", "gameId", "playId"]].copy()
        d["y"] = y_lead[ok].to_numpy()
        cl = d["gameId"].astype(str) + "_" + d["playId"].astype(str)
        # play fixed effect by demeaning, then regress y ~ treat + strain_t + dose_t
        for c in ["y", treat_col, "strain", "dose"]:
            d[c] = d[c] - d.groupby(cl.values)[c].transform("mean")
        b, se = cluster_ols(d["y"], np.c_[d[treat_col], d["strain"], d["dose"]], cl)
        out.append((h, b[0], se[0]))
        print(f"    h={h:+d}  beta={b[0]:+.4f}  se={se[0]:.4f}", flush=True)
    return pd.DataFrame(out, columns=["h", "beta", "se"])


print("\n=== OWN break -> own strain ===", flush=True)
own = impulse("own_break")
print("\n=== TEAMMATE break -> own strain (QB-mediated spillover) ===", flush=True)
team = impulse("teammate_break")
own.to_csv("containment_own.csv", index=False)
team.to_csv("containment_teammate.csv", index=False)

# ----------------------------------------------------------------- figure --
fig, ax = plt.subplots(figsize=(8, 4.6))
for d, lab, col in [(own, "Own break $\\to$ own strain", "#1f77b4"),
                    (team, "Teammate break $\\to$ own strain", "#d62728")]:
    ax.plot(d["h"] * 0.1, d["beta"], "-o", ms=4, color=col, label=lab)
    ax.fill_between(d["h"] * 0.1, d["beta"] - 1.96 * d["se"], d["beta"] + 1.96 * d["se"],
                    color=col, alpha=0.18)
ax.axvline(0, ls="--", color="0.5", lw=1); ax.axhline(0, ls="-", color="0.7", lw=0.8)
ax.set_xlabel("Seconds relative to the containment break")
ax.set_ylabel(r"$\Delta$ STRAIN (local-projection coefficient)")
ax.set_title("Does breaking containment raise pressure?  Impulse response of STRAIN\n"
             "(play FE, clustered by play; $h<0$ are placebo leads)", fontsize=10)
ax.legend(); fig.tight_layout()
fig.savefig("figures/containment_break.png", dpi=160)
fig.savefig("figures/containment_break.pdf")
print("\nwrote figures/containment_break.{png,pdf}, containment_{own,teammate}.csv")
pk = team.loc[team["beta"].idxmax()]
print(f"peak TEAMMATE effect: beta={pk['beta']:+.4f} (se {pk['se']:.4f}) at {pk['h']*0.1:+.1f}s")
print(f"placebo (mean |beta| at h<0, teammate): {team.loc[team.h<0,'beta'].abs().mean():.4f}")
