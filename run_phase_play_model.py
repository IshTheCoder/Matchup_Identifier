"""Run the play-level (non-continuous) plus-minus model on ONE phase's assignments and
write phase-tagged rankings + tables. Usage: python run_phase_play_model.py PHASE ASSIGN.csv

Produces (suffix = _PHASE):
  play_design_PHASE.pkl, play_model_samples_PHASE.pkl
  rusher_rankings_PHASE.csv, blocker_rankings_PHASE.csv, qb_suppression_PHASE.csv
  tables/rusher_plusminus_PHASE.tex, tables/blocker_plusminus_PHASE.tex, tables/qb_suppression_PHASE.tex
"""
import os, sys, pickle, time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import numpy as np, pandas as pd, jax
jax.config.update("jax_enable_x64", True)
import feature_engineering as fe
from models import RusherPlusMinusModel

PHASE, APATH = sys.argv[1], sys.argv[2]
NORMALIZE = len(sys.argv) > 3 and sys.argv[3] == "--normalize"
MIN_SNAPS = 50
RUSH, BLK = ["DE", "DT", "NT", "OLB"], ["T", "G", "C"]
players = pd.read_csv("data/players.csv").set_index("nflId")

t0 = time.time()
print(f"[{PHASE}] building play design from {APATH} (normalize_attention={NORMALIZE}) ...", flush=True)
data, enc = fe.build_play_design_from_files(assignment_path=APATH, normalize_attention=NORMALIZE)
pickle.dump((data, enc), open(f"play_design_{PHASE}.pkl", "wb"))
print(f"[{PHASE}] plays={data['outcome'].shape[0]} valid_obs={int(data['mask'].sum())} "
      f"built in {time.time()-t0:.0f}s; running MCMC ...", flush=True)

m = RusherPlusMinusModel()
m.run_mcmc_inference(data, num_warmup=1000, num_samples=2000, num_chains=1)
samples = m.get_posterior_samples()
pickle.dump(samples, open(f"play_model_samples_{PHASE}.pkl", "wb"))
sigma = float(np.asarray(samples["sigma"]).mean())
try:
    div = int(np.asarray(m.mcmc.get_extra_fields().get("diverging", np.array([0])).sum()))
except Exception:
    div = -1
print(f"[{PHASE}] MCMC done in {time.time()-t0:.0f}s | sigma={sigma:.3f} divergences={div}", flush=True)

# --- snaps ---
mask = np.asarray(data["mask"]); rid = np.asarray(data["rusher_ids"])
bid = np.asarray(data["blocker_ids"]); asg = np.asarray(data["assignment"])
rsnap = np.bincount(rid[mask > 0].ravel(), minlength=int(data["N_rushers"]))
bsnap = np.bincount(bid[(asg.sum(1) > 0)].ravel(), minlength=int(data["N_blockers"]))
qsnap = np.bincount(np.asarray(data["quarterback_ids"]), minlength=int(data["N_quarterbacks"]))


def rankings(cov, w, z, sig, enc_key, snaps):
    draws = samples[w] @ data[cov].T + samples[sig][:, None] * samples[z]
    inv = {v: k for k, v in enc[enc_key].items()}
    ids = [inv[i] for i in range(draws.shape[1])]
    df = pd.DataFrame({"nflId": ids,
                       "name": [players["displayName"].get(i, "?") for i in ids],
                       "pos": [players["officialPosition"].get(i, "?") for i in ids],
                       "effect": draws.mean(0),
                       "hdi_lo": np.percentile(draws, 3, axis=0),
                       "hdi_hi": np.percentile(draws, 97, axis=0),
                       "snaps": snaps})
    return df[df.snaps >= MIN_SNAPS].sort_values(["pos", "effect"], ascending=[True, False]).reset_index(drop=True)


bl = rankings("blocker_covariates", "blocker_weight", "z_blocker", "sigma_blocker", "blocker", bsnap)
ru = rankings("rusher_covariates", "rusher_weight", "z_rusher", "sigma_rusher", "rusher", rsnap)
bl.to_csv(f"blocker_rankings_{PHASE}.csv", index=False)
ru.to_csv(f"rusher_rankings_{PHASE}.csv", index=False)

# --- QB suppression ---
qd = samples["quarterback_weight"] @ data["quarterback_covariates"].T \
    + samples["sigma_quarterback"][:, None] * samples["z_quarterback"]
qinv = {v: k for k, v in enc["qb"].items()}
qids = [qinv[i] for i in range(qd.shape[1])]
qb = pd.DataFrame({"nflId": qids, "name": [players["displayName"].get(i, "?") for i in qids],
                   "suppression": -qd.mean(0), "snaps": qsnap})
qb = qb[qb.snaps >= MIN_SNAPS].sort_values("suppression", ascending=False).reset_index(drop=True)
qb.to_csv(f"qb_suppression_{PHASE}.csv", index=False)


def _sub(df, cap, w):
    body = " \\\\\n".join(f"{r['name']} & {r['effect']:.2f}" for _, r in df.iterrows()) + " \\\\"
    return ("\\begin{subtable}{" + w + "\\textwidth}\n\\centering\n\\footnotesize\n\\begin{tabular}{lc}\n"
            "\\toprule\nName & Effect \\\\\n\\midrule\n" + body
            + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + cap + "}\n\\end{subtable}")


def _tbl(groups, df, top, label, cap, w):
    subs = [_sub(df[df.pos == g].sort_values("effect", ascending=not top).head(5), g, w) for g in groups]
    return ("\\begin{table}[h!]\n\\centering\n" + "\n\\hfill\n".join(subs)
            + "\n\\caption{" + cap + "}\n\\label{" + label + "}\n\\end{table}\n")


NOTE = f"(Phase {PHASE}, min {MIN_SNAPS} snaps)"
with open(f"tables/rusher_plusminus_{PHASE}.tex", "w") as f:
    f.write(_tbl(RUSH, ru, True, f"tab:rusher_pm_{PHASE}_top", f"Top 5 rushers by plus-minus, by position {NOTE}.", "0.24"))
    f.write("\n" + _tbl(RUSH, ru, False, f"tab:rusher_pm_{PHASE}_bot", f"Bottom 5 rushers {NOTE}.", "0.24"))
with open(f"tables/blocker_plusminus_{PHASE}.tex", "w") as f:
    f.write(_tbl(BLK, bl, True, f"tab:blocker_pm_{PHASE}_top", f"Top 5 pass blockers by plus-minus, by position {NOTE}.", "0.32"))
    f.write("\n" + _tbl(BLK, bl, False, f"tab:blocker_pm_{PHASE}_bot", f"Bottom 5 pass blockers {NOTE}.", "0.32"))
with open(f"tables/qb_suppression_{PHASE}.tex", "w") as f:
    f.write("\\begin{table}[h!]\n\\centering\n")
    for d, c in [(qb.head(10), "Most strain-suppressing"), (qb.tail(10).iloc[::-1], "Least strain-suppressing")]:
        body = " \\\\\n".join(f"{r['name']} & {r['suppression']:.3f}" for _, r in d.iterrows()) + " \\\\"
        f.write("\\begin{subtable}{0.48\\textwidth}\n\\centering\n\\footnotesize\n\\begin{tabular}{lc}\n"
                "\\toprule\nName & Strain Suppr. \\\\\n\\midrule\n" + body
                + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + c + "}\n\\end{subtable}\n\\hfill\n")
    f.write(f"\\caption{{Quarterbacks by strain suppression {NOTE}.}}\n\\label{{tab:qb_supp_{PHASE}}}\n\\end{{table}}\n")

print(f"[{PHASE}] wrote rankings + tables. top blockers:", flush=True)
for g in BLK:
    d = bl[bl.pos == g].head(3)
    print(f"  {g}: " + ", ".join(f"{r['name']} {r['effect']:.2f}" for _, r in d.iterrows()))
print(f"[{PHASE}] total {time.time()-t0:.0f}s", flush=True)
