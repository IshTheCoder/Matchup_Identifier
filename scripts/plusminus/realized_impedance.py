"""Front-normalized realized blocker impedance (presentation metric; model stays the
unnormalized total-impedance sum). Per (play, blocker): imp = B_b * sum_j theta(b,j)
= strain removed that play. Front-normalize per play relative to the blockers on the
same play (controls for 4- vs 5-man front / usage). Aggregate per blocker."""
import pickle, numpy as np, pandas as pd
data, enc = pickle.load(open("play_design_phase25.pkl", "rb"))      # unnormalized phase2.5
s = pickle.load(open("play_model_samples_phase25.pkl", "rb"))
players = pd.read_csv("data/players.csv").set_index("nflId")
MIN_SNAPS = 50
inv = {v: k for k, v in enc["blocker"].items()}                      # global idx -> nflId

B = (s["blocker_weight"] @ data["blocker_covariates"].T + s["sigma_blocker"][:, None] * s["z_blocker"]).mean(0)
asg = np.asarray(data["assignment"])            # (N,R,B) theta_bar
bidx = np.asarray(data["blocker_ids"])          # (N,B) global blocker index
mask = np.asarray(data["mask"])                 # (N,R)
eng = asg.sum(1)                                 # (N,B) engagement = sum_j theta(b,j)
present = eng > 0
front = mask.sum(1).astype(int)
imp = eng * B[bidx]                              # (N,B) realized impedance per (play, blocker)

# --- 95% credible interval for the front-normalized realized impedance ---
# FN is linear in the blocker coefficients, so propagate the full posterior of B through the
# same per-play peer-normalization and per-blocker averaging, then take percentiles over draws.
Bd = (s["blocker_weight"] @ data["blocker_covariates"].T
      + s["sigma_blocker"][:, None] * s["z_blocker"])      # (D, N_blockers) posterior draws
ndraw, n_global = Bd.shape
flat_gid = bidx[present]                                    # (P,) present slot -> global blocker idx
cnt_blk = np.bincount(flat_gid, minlength=n_global).astype(float)   # snaps per global blocker
cntp = present.sum(1)                                       # (N,) blockers present per play
fnsum = np.zeros((ndraw, n_global))
for i in range(0, ndraw, 200):                             # chunk draws to bound memory
    Bc = Bd[i:i + 200]                                     # (c, N_blockers)
    impc = eng[None] * Bc[:, bidx]                         # (c, N, B)
    mbar = (impc * present[None]).sum(2) / np.maximum(cntp[None], 1)   # (c, N) per-play peer mean
    relp = (impc - mbar[:, :, None])[:, present]           # (c, P) front-normalized, present only
    out = np.zeros((Bc.shape[0], n_global))
    np.add.at(out, (np.arange(Bc.shape[0])[:, None], flat_gid[None, :]), relp)
    fnsum[i:i + 200] = out
fnval_draws = fnsum / np.maximum(cnt_blk[None], 1)         # (D, N_blockers) per-blocker FN per draw
fn_lo = np.percentile(fnval_draws, 2.5, axis=0)
fn_hi = np.percentile(fnval_draws, 97.5, axis=0)

rows = []
for n in range(asg.shape[0]):
    bl = np.where(present[n])[0]
    if len(bl) == 0:
        continue
    mbar = imp[n, bl].mean()
    for b in bl:
        rows.append((int(bidx[n, b]), int(front[n]), float(eng[n, b]),
                     float(imp[n, b]), float(imp[n, b] - mbar)))
df = pd.DataFrame(rows, columns=["bidx", "front", "engagement", "impedance", "rel_impedance"])
g = df.groupby("bidx").agg(snaps=("impedance", "size"), mean_eng=("engagement", "mean"),
                           real_imp=("impedance", "mean"), fn_value=("rel_impedance", "mean"))
g["B_coef"] = [B[i] for i in g.index]
g["fn_lo"] = fn_lo[g.index.to_numpy()]
g["fn_hi"] = fn_hi[g.index.to_numpy()]
g["nflId"] = [inv[i] for i in g.index]
g["name"] = g["nflId"].map(players["displayName"])
g["pos"] = g["nflId"].map(players["officialPosition"])
g = g[g.snaps >= MIN_SNAPS].reset_index(drop=True)
g.to_csv("blocker_value.csv", index=False)
print(f"blockers >= {MIN_SNAPS} snaps: {len(g)}")
print("Spearman(front-norm value vs raw B coef):",
      round(g.fn_value.corr(g.B_coef, method="spearman"), 3))
for pos in ["T", "G", "C"]:
    d = g[g.pos == pos].sort_values("fn_value", ascending=False)
    print(f"\n== {pos}: top by front-normalized realized impedance ==")
    print(d.head(6)[["name", "fn_value", "real_imp", "B_coef", "mean_eng", "snaps"]].round(3).to_string(index=False))
print("\n== lowest-value blockers (low-engagement releasers expected) ==")
print(g.sort_values("fn_value").head(8)[["name", "pos", "fn_value", "real_imp", "B_coef", "mean_eng"]].round(3).to_string(index=False))


# ------------------------------------------------ tables/blocker_value.tex --
def _sub(pos):
    d = g[g.pos == pos].sort_values("fn_value", ascending=False).head(5)
    rows = " \\\\\n".join(
        f"{r['name']} & {r['real_imp']:.3f} & {r['fn_value']:+.3f} & "
        f"$[{r['fn_lo']:+.3f},\\,{r['fn_hi']:+.3f}]$" for _, r in d.iterrows()) + " \\\\"
    return ("\\begin{subtable}{0.48\\textwidth}\n\\centering\n\\footnotesize\n"
            "\\begin{tabular}{lccc}\n\\toprule\n"
            "Name & Imp. & FN & 95\\% CI \\\\\n\\midrule\n" + rows
            + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + pos + "}\n\\end{subtable}")


subs = [_sub("T"), _sub("G"), _sub("C")]
body = "\n\\hfill\n".join(subs[:2]) + "\n\n\\bigskip\n" + subs[2] + "\n"
with open("tables/blocker_value.tex", "w") as f:
    f.write("\\begin{table}[h!]\n\\centering\n" + body
            + "\\caption{Top pass blockers by \\emph{realized impedance} --- mean strain removed per "
            "snap $\\overline{B_b\\sum_j\\theta(b,j)}$ (Imp., STRAIN units) --- and its front-normalized "
            "version (FN: impedance above same-front peers), min 50 snaps; brackets are 95\\% posterior "
            "credible intervals for FN. Unlike the raw coefficient, this down-weights blockers who "
            "rarely engage (e.g.\\ a release-heavy fullback).}\n"
            "\\label{tab:blocker_value}\n\\end{table}\n")
print("wrote tables/blocker_value.tex")
