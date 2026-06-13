"""Phase 1: refit the assignment HMM on all 8 weeks with hierarchical per-blocker rho
(Beta-Binomial empirical Bayes), production emission config (von Mises kappa=1, conditional-
logit matchup prior). Writes Phase-1 params to new paths so the Phase-0 baseline is preserved.

  fitted_params_jax_phase1.csv  (adds rho_ab = per-position Beta (a,b))
  rho_by_player_phase1.csv      (nflId, position, rho_j, n_transitions)
"""
import os, sys, time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "src")

import jax
jax.config.update("jax_enable_x64", True)

import baum_welch_jax as bwj

t0 = time.time()
print("Phase-1 hierarchical HMM fit: 8 weeks, vonmises kappa=1, logit prior", flush=True)
bwj.fit_by_position(
    csv_paths=[f"data/sample_data_week_{i}.csv" for i in range(8)],
    n_iter=15,
    tau0=(0.8, 0.2), sigma0=2.0, rho0=0.95,
    orientation="vonmises", conc=1.0,
    prior="logit", rho_mode="hierarchical",
    out_path="fitted_params_jax_phase1.csv",
    rho_by_player_path="rho_by_player_phase1.csv",
)
print(f"Phase-1 HMM fit done in {time.time()-t0:.0f}s", flush=True)
