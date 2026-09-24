"""Fit the continuous-time AR(1) max-strain plus-minus model on all 8 weeks (NUTS, 1 chain;
slow over ~1.1M rusher-frames but matches the play-level config) and write rusher/blocker
rankings by position, thresholded by play-level pass snaps (consistent with the play-level
tables)."""
import os, sys, pickle, time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "src")
sys.path.insert(0, "model")

import numpy as np
import pandas as pd
import jax

jax.config.update("jax_enable_x64", True)

import feature_engineering as fe
from models import ContinuousRusherPlusMinusModel

MIN_SNAPS = 50

t0 = time.time()
print("building continuous (play,rusher,frame) design (8 weeks)...", flush=True)
data, enc = fe.build_continuous_design_from_files()
print(f"  built in {time.time()-t0:.0f}s | N_obs={data['outcome'].shape[0]} "
      f"rushers={data['N_rushers']} blockers={data['N_blockers']} qbs={data['N_quarterbacks']}", flush=True)
pickle.dump((data, enc), open("continuous_design_8wk.pkl", "wb"))

m = ContinuousRusherPlusMinusModel()
print("running MCMC: 1 chain, 1000 warmup, 2000 samples (this will be slow) ...", flush=True)
t1 = time.time()
m.run_mcmc_inference(data, num_warmup=1000, num_samples=2000, num_chains=1)
samples = m.get_posterior_samples()
print(f"  MCMC done in {time.time()-t1:.0f}s", flush=True)
pickle.dump(samples, open("continuous_model_samples.pkl", "wb"))

rho = np.asarray(samples["rho_ar"])
print(f"rho_ar: mean={rho.mean():.3f} 94% HDI=({np.percentile(rho,3):.3f},{np.percentile(rho,97):.3f})")
print(f"sigma (residual sd): {float(np.asarray(samples['sigma']).mean()):.3f}", flush=True)
try:
    div = int(np.asarray(m.mcmc.get_extra_fields().get("diverging", np.array([0])).sum()))
    print(f"divergences: {div}", flush=True)
except Exception:
    pass

# --- play-level pass-snap counts keyed by nflId (consistent threshold across models) ---
pdata, penc = pickle.load(open("play_design_8wk.pkl", "rb"))
pmask = np.asarray(pdata["mask"]); prid = np.asarray(pdata["rusher_ids"])
pbid = np.asarray(pdata["blocker_ids"]); pasg = np.asarray(pdata["assignment"])
rsnap = np.bincount(prid[pmask > 0].ravel(), minlength=int(pdata["N_rushers"]))
bsnap = np.bincount(pbid[(pasg.sum(1) > 0)].ravel(), minlength=int(pdata["N_blockers"]))
rinv, binv = {v: k for k, v in penc["rusher"].items()}, {v: k for k, v in penc["blocker"].items()}
snaps_by_nfl = {rinv[i]: int(rsnap[i]) for i in range(len(rsnap))}
for i in range(len(bsnap)):  # blockers can also be rushers; keep max snap evidence
    snaps_by_nfl[binv[i]] = max(snaps_by_nfl.get(binv[i], 0), int(bsnap[i]))

players = pd.read_csv("data/players.csv").set_index("nflId")


def rankings(cov_key, w_key, z_key, sig_key, enc_key):
    draws = samples[w_key] @ data[cov_key].T + samples[sig_key][:, None] * samples[z_key]
    inv = {v: k for k, v in enc[enc_key].items()}
    ids = [inv[i] for i in range(draws.shape[1])]
    df = pd.DataFrame({
        "nflId": ids,
        "name": [players["displayName"].get(i, "?") for i in ids],
        "pos": [players["officialPosition"].get(i, "?") for i in ids],
        "effect": draws.mean(0),
        "hdi_lo": np.percentile(draws, 3, axis=0),
        "hdi_hi": np.percentile(draws, 97, axis=0),
        "snaps": [snaps_by_nfl.get(i, 0) for i in ids],
    })
    df = df[df["snaps"] >= MIN_SNAPS]
    return df.sort_values(["pos", "effect"], ascending=[True, False]).reset_index(drop=True)


bl = rankings("blocker_covariates", "blocker_weight", "z_blocker", "sigma_blocker", "blocker")
ru = rankings("rusher_covariates", "rusher_weight", "z_rusher", "sigma_rusher", "rusher")
bl.to_csv("continuous_blocker_rankings.csv", index=False)
ru.to_csv("continuous_rusher_rankings.csv", index=False)


def _rows(df):
    return " \\\\\n".join(f"{r['name']} & {r['effect']:.2f}" for _, r in df.iterrows()) + " \\\\"


def _subtable(df, cap, width, sym):
    return ("\\begin{subtable}{" + width + "\\textwidth}\n\\centering\n\\footnotesize\n"
            "\\begin{tabular}{lc}\n\\toprule\nName & " + sym + " \\\\\n\\midrule\n"
            + _rows(df) + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + cap + "}\n\\end{subtable}")


def _table(groups, df, top, label, caption, width, sym):
    subs = [_subtable(df[df["pos"] == g].sort_values("effect", ascending=not top).head(5), g, width, sym)
            for g in groups]
    return ("\\begin{table}[h!]\n\\centering\n" + "\n\\hfill\n".join(subs)
            + "\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\end{table}\n")


RUSH, BLK = ["DE", "DT", "NT", "OLB"], ["T", "G", "C"]
NOTE = f"(min {MIN_SNAPS} pass snaps)"
with open("tables/rusher_plusminus_continuous.tex", "w") as f:
    f.write("% Continuous-time (AR(1)) plus-minus rusher effect by position.\n")
    f.write(_table(RUSH, ru, True, "tab:rusher_pm_cont_top",
                   f"Top 5 pass rushers by continuous-time plus-minus effect, by position {NOTE}.", "0.24", "$R_j$"))
    f.write("\n")
    f.write(_table(RUSH, ru, False, "tab:rusher_pm_cont_bot",
                   f"Bottom 5 pass rushers by continuous-time plus-minus effect, by position {NOTE}.", "0.24", "$R_j$"))
with open("tables/blocker_plusminus_continuous.tex", "w") as f:
    f.write("% Continuous-time (AR(1)) plus-minus blocker effect by position.\n")
    f.write(_table(BLK, bl, True, "tab:blocker_pm_cont_top",
                   f"Top 5 pass blockers by continuous-time plus-minus effect, by position {NOTE}.", "0.32", "$B_b$"))
    f.write("\n")
    f.write(_table(BLK, bl, False, "tab:blocker_pm_cont_bot",
                   f"Bottom 5 pass blockers by continuous-time plus-minus effect, by position {NOTE}.", "0.32", "$B_b$"))

print(f"\nrushers kept: {len(ru)} | blockers kept: {len(bl)}")
for g in RUSH:
    d = ru[ru["pos"] == g].head(3)
    print(f"  rusher {g:>3}: " + ", ".join(f"{r['name']} {r['effect']:.2f}({int(r['snaps'])})" for _, r in d.iterrows()))
for g in BLK:
    d = bl[bl["pos"] == g].head(3)
    print(f"  blocker {g:>2}: " + ", ".join(f"{r['name']} {r['effect']:.2f}({int(r['snaps'])})" for _, r in d.iterrows()))
print("wrote continuous_{rusher,blocker}_rankings.csv, tables/{rusher,blocker}_plusminus_continuous.tex")
print(f"total {time.time()-t0:.0f}s", flush=True)
