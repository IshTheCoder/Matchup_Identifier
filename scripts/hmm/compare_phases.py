"""Cross-phase comparison of the play-level plus-minus effects + survival summary."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # play_model_samples.pkl holds jax arrays; unpickling them
                                                # must not touch the (CPU-only) analysis's cuda backend
import pickle, numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PH = {"0 (raw)": "phase0_results", "1 ($\\rho_j$)": "phase1",
      "2 (null)": "phase2", "2.5 (struct null)": "phase25"}
def path(tag, kind):
    base = PH[tag]
    return f"{base}/{kind}.csv" if base == "phase0_results" else f"{kind}_{base}.csv"

# sigma per phase (phase0 from its pkl; others from logs)
sig = {}
sig["0 (raw)"] = float(np.asarray(pickle.load(open("play_model_samples.pkl","rb"))["sigma"]).mean())
for tag, base in [("1 ($\\rho_j$)","phase1"),("2 (null)","phase2"),("2.5 (struct null)","phase25")]:
    import re
    txt = open(f"phase_play_{base}.log").read()
    m = re.search(r"sigma=([0-9.]+)", txt); sig[tag] = float(m.group(1)) if m else np.nan

def load(kind, col):
    d = {}
    for tag in PH:
        df = pd.read_csv(path(tag, kind))
        d[tag] = df.set_index("nflId")[col]
    return d

rb = load("blocker_rankings", "effect"); ru = load("rusher_rankings", "effect")
qb = load("qb_suppression", "suppression")
ref = "0 (raw)"
rows = []
for tag in PH:
    b = pd.concat([rb[ref], rb[tag]], axis=1, join="inner"); b.columns=["r","x"]
    r = pd.concat([ru[ref], ru[tag]], axis=1, join="inner"); r.columns=["r","x"]
    q = pd.concat([qb[ref], qb[tag]], axis=1, join="inner"); q.columns=["r","x"]
    rows.append({"phase": tag, "sigma": round(sig[tag],3),
                 "n_block": rb[tag].shape[0],
                 "blk_rho_vs_raw": round(spearmanr(b.r,b.x).correlation,3),
                 "rush_rho_vs_raw": round(spearmanr(r.r,r.x).correlation,3),
                 "qb_rho_vs_raw": round(spearmanr(q.r,q.x).correlation,3),
                 "blk_mean_abs": round(rb[tag].abs().mean(),3),
                 "blk_p95_abs": round(rb[tag].abs().quantile(.95),3)})
comp = pd.DataFrame(rows); comp.to_csv("phase_comparison.csv", index=False)
print(comp.to_string(index=False))

# tables/phase_comparison.tex — the exact table the paper \input's, generated from `comp`.
import os
os.makedirs("tables", exist_ok=True)
ROW_LABEL = {"0 (raw)": "Raw", "1 ($\\rho_j$)": "$+$ per-player stickiness",
             "2 (null)": "$+$ null state", "2.5 (struct null)": "$+$ structured null"}
with open("tables/phase_comparison.tex", "w") as f:
    f.write("\\begin{table}[h!]\n\\centering\n\\footnotesize\n")
    f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
    f.write("Assignments & $\\sigma$ & blk $\\rho_S$ & rush $\\rho_S$ & QB $\\rho_S$ "
            "& blk mean$|e|$ & blk p95$|e|$ \\\\\n\\midrule\n")
    for _, r in comp.iterrows():
        lab = ROW_LABEL.get(r["phase"], r["phase"])
        f.write(f"{lab} & {r.sigma:.3f} & {r.blk_rho_vs_raw:.3f} & {r.rush_rho_vs_raw:.3f} "
                f"& {r.qb_rho_vs_raw:.3f} & {r.blk_mean_abs:.3f} & {r.blk_p95_abs:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
    f.write("\\caption{Play-level plus-minus across assignment refinements: residual $\\sigma$, "
            "Spearman rank-correlation of effects vs.\\ the raw-assignment fit ($\\rho_S$), and "
            "blocker effect magnitude. Rankings are highly preserved while null de-biasing sharpens "
            "blocker effects.}\n")
    f.write("\\label{tab:phase_comparison}\n\\end{table}\n")
print("wrote tables/phase_comparison.tex")

# figure: blocker effect raw vs phase2.5 (rank preserved, magnitude sharpened)
b = pd.concat([rb["0 (raw)"], rb["2.5 (struct null)"]], axis=1, join="inner"); b.columns=["raw","p25"]
fig, ax = plt.subplots(1,2, figsize=(11,4.2))
ax[0].scatter(b.raw, b.p25, s=14, alpha=.6, color="#1f5fb4")
lim=[min(b.raw.min(),b.p25.min())-.02, max(b.raw.max(),b.p25.max())+.02]
ax[0].plot(lim,lim,"--",color="0.6"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("Blocker effect — Phase 0 (raw assignments)")
ax[0].set_ylabel("Blocker effect — Phase 2.5 (structured null)")
ax[0].set_title(f"Blocker plus-minus: rank preserved (Spearman {spearmanr(b.raw,b.p25).correlation:.2f}),\nmagnitude sharpened by de-biasing")
tags=list(PH); xs=range(len(tags))
ax[1].plot(xs,[comp.set_index('phase').loc[t,'blk_mean_abs'] for t in tags],"o-",label="mean |effect|")
ax[1].plot(xs,[comp.set_index('phase').loc[t,'blk_p95_abs'] for t in tags],"s-",label="95th pct |effect|")
ax[1].set_xticks(list(xs)); ax[1].set_xticklabels(["0","1","2","2.5"]); ax[1].set_xlabel("Phase")
ax[1].set_ylabel("blocker |effect|"); ax[1].legend(); ax[1].set_title("Blocker effect magnitude grows as\nattention is de-biased")
fig.tight_layout(); fig.savefig("figures/phase_comparison.png", dpi=160); fig.savefig("figures/phase_comparison.pdf")
print("wrote phase_comparison.csv, figures/phase_comparison.{png,pdf}")
