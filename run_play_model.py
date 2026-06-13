"""Fit the play-level max-strain plus-minus model on all 8 weeks (1-chain MCMC) and
write rusher/blocker rankings."""
import os, sys, pickle, time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "src")
sys.path.insert(0, "model")

import numpy as np
import pandas as pd
import jax

jax.config.update("jax_enable_x64", True)

import feature_engineering as fe
from models import RusherPlusMinusModel

t0 = time.time()
print("building play-level data dict (8 weeks)...", flush=True)
data, enc = fe.build_play_design_from_files()
print(f"  built in {time.time()-t0:.0f}s | plays={data['outcome'].shape[0]} "
      f"rushers={data['N_rushers']} blockers={data['N_blockers']} "
      f"valid_obs={int(data['mask'].sum())}", flush=True)
pickle.dump((data, enc), open("play_design_8wk.pkl", "wb"))

m = RusherPlusMinusModel()
print("running MCMC: 1 chain, 1000 warmup, 2000 samples ...", flush=True)
t1 = time.time()
m.run_mcmc_inference(data, num_warmup=1000, num_samples=2000, num_chains=1)
samples = m.get_posterior_samples()
print(f"  MCMC done in {time.time()-t1:.0f}s", flush=True)
pickle.dump(samples, open("play_model_samples.pkl", "wb"))

players = pd.read_csv("data/players.csv").set_index("nflId")


def rank(cov_key, w_key, z_key, sig_key, enc_key):
    # effect = covariates @ weight + sigma * z  (per posterior draw)
    draws = samples[w_key] @ data[cov_key].T + samples[sig_key][:, None] * samples[z_key]
    mean = draws.mean(0)
    lo, hi = np.percentile(draws, [3, 97], axis=0)
    inv = {v: k for k, v in enc[enc_key].items()}
    ids = [inv[i] for i in range(len(mean))]
    return pd.DataFrame({
        "nflId": ids,
        "name": [players["displayName"].get(i, "?") for i in ids],
        "pos": [players["officialPosition"].get(i, "?") for i in ids],
        "effect": mean, "hdi_lo": lo, "hdi_hi": hi,
    }).sort_values("effect", ascending=False).reset_index(drop=True)


bl = rank("blocker_covariates", "blocker_weight", "z_blocker", "sigma_blocker", "blocker")
ru = rank("rusher_covariates", "rusher_weight", "z_rusher", "sigma_rusher", "rusher")
bl.to_csv("blocker_rankings.csv", index=False)
ru.to_csv("rusher_rankings.csv", index=False)

print(f"\nsigma (residual sd): {float(samples['sigma'].mean()):.3f}", flush=True)
try:
    div = int(np.asarray(m.mcmc.get_extra_fields().get("diverging", np.array([0])).sum()))
    print(f"divergences: {div}")
except Exception:
    pass
print("\n=== TOP 10 BLOCKERS (best suppressors, high effect) ===")
print(bl.head(10).to_string(index=False))
print("\n=== TOP 10 RUSHERS (most pressure, high effect) ===")
print(ru.head(10).to_string(index=False))
print("\nwrote blocker_rankings.csv, rusher_rankings.csv, play_model_samples.pkl", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
