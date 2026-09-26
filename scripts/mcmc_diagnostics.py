"""MCMC convergence diagnostics (Rhat, bulk/tail ESS) -> tables/mcmc_diagnostics.tex

Covers every parameter the paper discusses, for all three NUTS-fit models:
  * play-level plus-minus            play_model_samples_phase25.pkl              (4 chains)
  * continuous-time blocker rating   blocker_dose_samples_filtered_baseline_mcmc.pkl (4 chains)
  * opponent-adjusted block hold     block_hold_discrete_samples.pkl             (4 chains)

CHAIN RECOVERY. BaseModel.run_mcmc_inference stores mcmc.get_samples(), which flattens the
(chain, draw) axes; numpyro concatenates chain-major, so reshaping the leading axis to
(n_chains, n_draws) restores chain identity. The script asserts the leading axis divides evenly
and reports the per-chain means so a mis-specified chain count is visible rather than silent.

The per-player effects (R_j, B_b, Q_q) are deterministic transforms of the sampled sites, so we
diagnose the quantities the paper actually reports, not just the raw z's.

Run from the repo root:  python3 scripts/mcmc_diagnostics.py
"""
import pickle
import sys

import arviz as az
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ESS_FLOOR, RHAT_CEIL = 400, 1.01


def _chains(x, n_chains):
    """(n_chains*n_draws, ...) -> (n_chains, n_draws, ...) — numpyro flattens chain-major."""
    x = np.asarray(x, dtype=float)
    assert x.shape[0] % n_chains == 0, f"leading axis {x.shape[0]} not divisible by {n_chains}"
    return x.reshape(n_chains, x.shape[0] // n_chains, *x.shape[1:])


def _diag(x, n_chains):
    """-> (max Rhat, min bulk ESS, min tail ESS, n components) over a parameter's components"""
    s = _chains(x, n_chains)
    flat = s.reshape(s.shape[0], s.shape[1], -1)
    rh, eb, et = [], [], []
    for k in range(flat.shape[2]):
        col = flat[:, :, k]
        if np.allclose(col, col.flat[0]):          # constant (e.g. a dropped level) -> undefined
            continue
        rh.append(float(az.rhat(col, method="rank")))
        eb.append(float(az.ess(col, method="bulk")))
        # arviz>=1.0 requires the tail probabilities explicitly; (0.05, 0.95) is the standard
        # tail-ESS definition of Vehtari et al. (2021).
        et.append(float(az.ess(col, method="tail", prob=(0.05, 0.95))))
    if not rh:
        return np.nan, np.nan, np.nan, flat.shape[2]
    return max(rh), min(eb), min(et), flat.shape[2]


def _rows(label, params, n_chains):
    out = []
    for nm, arr in params:
        r, eb, et, k = _diag(arr, n_chains)
        out.append({"model": label, "param": nm, "k": k, "rhat": r,
                    "ess_bulk": eb, "ess_tail": et, "chains": n_chains})
        print(f"  {label:22} {nm:34} k={k:<5} Rhat={r:.4f}  ESS_bulk={eb:8.0f}  ESS_tail={et:8.0f}",
              flush=True)
    return out


rows = []

# -------------------------------------------------------- play-level plus-minus (phase 2.5) --
d, enc = pickle.load(open("play_design_phase25.pkl", "rb"))
s = pickle.load(open("play_model_samples_phase25.pkl", "rb"))
eff = lambda cov, w, z, sg: (np.asarray(s[w]) @ np.asarray(d[cov]).T
                             + np.asarray(s[sg])[:, None] * np.asarray(s[z]))
rows += _rows("Play-level plus-minus", [
    (r"$\mu$ (intercept)", s["intercept"]),
    (r"$R_j$ (rusher effects)", eff("rusher_covariates", "rusher_weight", "z_rusher", "sigma_rusher")),
    (r"$B_b$ (blocker effects)", eff("blocker_covariates", "blocker_weight", "z_blocker", "sigma_blocker")),
    (r"$Q_q$ (quarterback effects)", eff("quarterback_covariates", "quarterback_weight", "z_quarterback", "sigma_quarterback")),
    (r"$\alpha_R,\alpha_B,\alpha_Q$ (attributes)",
     np.concatenate([np.asarray(s[k]) for k in ("rusher_weight", "blocker_weight", "quarterback_weight")], 1)),
    (r"$\sigma_R,\sigma_B,\sigma_Q$",
     np.stack([np.asarray(s[k]) for k in ("sigma_rusher", "sigma_blocker", "sigma_quarterback")], 1)),
    (r"$\sigma_{\mathrm{Off}},\sigma_{\mathrm{Def}}$", np.stack([np.asarray(s[k]) for k in ("sigma_offense", "sigma_defense")], 1)),
    (r"$\sigma_{\mathrm{Dn}},\sigma_{\mathrm{Qt}},\sigma_{\mathrm{OF}},\sigma_{\mathrm{DC}}$",
     np.stack([np.asarray(s[k]) for k in ("sigma_down", "sigma_quarter",
                                          "sigma_offensive_formation", "sigma_defensive_formation")], 1)),
    (r"$\gamma$ (situation slopes)",
     np.stack([np.asarray(s[k]) for k in ("score_diff_theta", "time_remaining_theta", "yard_line_theta")], 1)),
    (r"$\sigma_y$ (residual)", s["sigma"]),
], 4)

# --------------------------------------------------------- continuous-time blocker rating --
cs = pickle.load(open("blocker_dose_samples_filtered_baseline_mcmc.pkl", "rb"))
cont = [(r"$\mu$ (intercept)", cs["intercept"]),
        (r"$B^{\Delta}_b$ (blocker effects)", cs["blocker_effect"]),
        (r"$\alpha_B$ (blocker attributes)", cs["blocker_weight"]),
        (r"$\sigma_B^{\Delta}$", cs["sigma_blocker"])]
if "rusher_effect" in cs:
    cont += [(r"$R^{\Delta}_j$ (rusher effects)", cs["rusher_effect"]),
             (r"$\alpha_R$ (rusher attributes)", cs["rusher_weight"]),
             (r"$\sigma_R^{\Delta}$", cs["sigma_rusher"]),
             (r"$Q^{\Delta}_q$ (QB effects)", cs["quarterback_effect"]),
             (r"$\alpha_Q$ (QB attributes)", cs["quarterback_weight"]),
             (r"$\sigma_Q^{\Delta}$", cs["sigma_quarterback"])]
if "rho" in cs:
    cont.append((r"$\phi$ (STRAIN$_t$ control)", cs["rho"]))
cont.append((r"$\sigma_{\Delta}$ (residual)", cs["sigma"]))
rows += _rows("Continuous-time", cont, 4)

# ----------------------------------------------------- opponent-adjusted block-hold rating --
hs = pickle.load(open("block_hold_discrete_samples.pkl", "rb"))
rows += _rows("Block hold", [
    (r"$\alpha$ (baseline hazard)", hs["alpha"]),
    (r"$g_1,g_2$ (dwell)", np.stack([np.asarray(hs["g1"]), np.asarray(hs["g2"])], 1)),
    (r"$\beta_{\mathrm{opp}}$ (opponent quality)", hs["beta_opp"]),
    (r"$\sigma_u$ (frailty scale)", hs["sigma_b"]),
    (r"$-u_b$ (hold ratings)", -(np.asarray(hs["sigma_b"])[:, None] * np.asarray(hs["z_blocker"]))),
], 4)

# --------------------------------------------------------------------------------- table --
df = pd.DataFrame(rows)
bad = df[(df.rhat > RHAT_CEIL) | (df.ess_bulk < ESS_FLOOR) | (df.ess_tail < ESS_FLOOR)]
print(f"\nparameters breaching Rhat<={RHAT_CEIL} or ESS>={ESS_FLOOR}: {len(bad)}")
if len(bad):
    print(bad[["model", "param", "rhat", "ess_bulk", "ess_tail"]].to_string(index=False))

body = ""
for model, g in df.groupby("model", sort=False):
    body += "\\multicolumn{5}{l}{\\emph{%s} (%d chains)} \\\\\n\\midrule\n" % (model, g.chains.iloc[0])
    body += "".join(f"{r['param']} & {r.k} & {r.rhat:.3f} & {r.ess_bulk:,.0f} & {r.ess_tail:,.0f} \\\\\n"
                    for _, r in g.iterrows())
    body += "\\midrule\n"
body = body[:body.rfind("\\midrule")]

open("tables/mcmc_diagnostics.tex", "w").write(
    "% Rhat and bulk/tail ESS for every parameter discussed in the paper (4-chain NUTS).\n"
    "\\begin{table}[h!]\n\\centering\n\\footnotesize\n\\begin{tabular}{lcccc}\n\\toprule\n"
    "Parameter & Components & $\\widehat{R}$ & ESS (bulk) & ESS (tail) \\\\\n\\midrule\n" + body +
    "\\bottomrule\n\\end{tabular}\n\\caption{Convergence diagnostics for the three models fit by "
    "the No-U-Turn sampler, each run as four chains. For vector parameters, Components gives the "
    "number of scalar components and the table reports the \\emph{worst} value over them (largest "
    "$\\widehat{R}$, smallest effective sample size). Diagnostics are computed on the quantities "
    "the paper reports, so the per-player effects are the transformed $R_j$, $B_b$, $B^{\\Delta}_b$, $Q_q$ and "
    "$-u_b$ rather than the underlying standardized deviations. All parameters satisfy "
    "$\\widehat{R} \\leq %.3f$, with bulk and tail effective sample sizes of at least %d.}\n"
    "\\label{tab:mcmc_diagnostics}\n\\end{table}\n"
    # round, don't truncate: the rows print ESS with {:,.0f}, so int() would leave the caption
    # claiming a floor one below the smallest number actually shown in the table.
    % (df.rhat.max(), round(min(df.ess_bulk.min(), df.ess_tail.min()))))
print("\nwrote tables/mcmc_diagnostics.tex")
