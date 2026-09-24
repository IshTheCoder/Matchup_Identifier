"""Dose-response blocker model (ContinuousBlockerDoseModel): DELTA STRAIN(t->t+1) ~ intercept
- sum_b theta(b,j,t) B_b [+ rho*STRAIN_t]. Treatment is the assignment LEVEL (dose) at t, not its
change -- so it credits SUSTAINED engagement the first-difference (dtheta) model cannot see. Pass
"baseline" to add the rho*STRAIN_t control (mean-reversion test). SVI fit; B_b validated vs PFF
pressures allowed (raw + within-position). Compare to run_blocker_delta_model.py."""
import os, sys, time, pickle
os.environ.setdefault("JAX_PLATFORMS", "cpu")
N_CHAINS = 4
# set before the jax backend initializes so the chains run in parallel on CPU cores
import numpyro; numpyro.set_host_device_count(N_CHAINS)
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import feature_engineering as fe
from models import ContinuousBlockerDoseModel

args = [a for a in sys.argv[1:] if a not in ("mcmc", "baseline")]
USE_MCMC = "mcmc" in sys.argv
CONTROL_BASELINE = "baseline" in sys.argv
asg = args[0] if args else "assignment_data_phase25_filtered.csv"
tag = "_filtered" if "filtered" in asg else ""
otag = tag + ("_baseline" if CONTROL_BASELINE else "") + ("_mcmc" if USE_MCMC else "")
pkl = f"continuous_design_delta{tag}_8wk.pkl"
try:
    data, enc = pickle.load(open(pkl, "rb")); assert "assignment" in data
    print(f"loaded {pkl}", flush=True)
except Exception:
    print(f"building delta design from {asg} ...", flush=True); t0 = time.time()
    data, enc = fe.build_continuous_design_from_files(weeks=range(8), assignment_path=asg)
    pickle.dump((data, enc), open(pkl, "wb")); print(f"  built in {time.time()-t0:.0f}s", flush=True)
print(f"  assignments: {asg}  control_baseline={CONTROL_BASELINE}", flush=True)

# DOSE transform: outcome = DELTA STRAIN; regressor = theta(.,j,t) (the dose at t)
data["outcome"] = data["outcome"] - data["strain_current"]
data["dose"] = data["assignment"]
if CONTROL_BASELINE:
    data["baseline"] = data["strain_current"]
N = len(data["outcome"]); meandose = float(data["dose"].sum(1).mean())
print(f"N_obs={N:,} N_blockers={data['N_blockers']}  dSTRAIN mean={data['outcome'].mean():+.4f} "
      f"std={data['outcome'].std():.4f}  mean total dose/row={meandose:.3f}", flush=True)

m = ContinuousBlockerDoseModel()
if USE_MCMC:
    print(f"inference: MCMC (NUTS, {N_CHAINS} chains, 800/800)", flush=True)
    m.run_mcmc_inference(data, num_warmup=800, num_samples=800, num_chains=N_CHAINS)
else:
    print("inference: SVI (AutoNormal, 20000 steps)", flush=True)
    m.run_svi_inference(data, num_steps=20000, lr=5e-3)
s = m.get_posterior_samples(); pickle.dump(s, open(f"blocker_dose_samples{otag}.pkl", "wb"))
extra = f"  rho={float(s['rho'].mean()):+.4f}" if CONTROL_BASELINE else ""
print(f"  intercept={float(s['intercept'].mean()):+.4f}  sigma={float(s['sigma'].mean()):.4f}  "
      f"sigma_blocker={float(s['sigma_blocker'].mean()):.4f}{extra}", flush=True)

draws = s["blocker_effect"]                         # centered: posterior draws of B_b directly (D, N_blockers)
inv = {v: k for k, v in enc["blocker"].items()}
players = pd.read_csv("data/players.csv").set_index("nflId")
import model_io as mio                               # export posterior to model_outputs/*.parquet (R-accessible)
mio.export_effect(f"dose_blocker{'_baseline' if CONTROL_BASELINE else ''}", draws,
                  [inv[i] for i in range(draws.shape[1])], players)
df = pd.DataFrame({
    "nflId": [inv[i] for i in range(draws.shape[1])],
    "effect": draws.mean(0), "lo": np.percentile(draws, 2.5, 0), "hi": np.percentile(draws, 97.5, 0)})
df["name"] = df.nflId.map(players["displayName"]); df["pos"] = df.nflId.map(players["officialPosition"])
pff = pd.read_csv("data/pffScoutingData.csv"); pb = pff[pff.pff_role == "Pass Block"].copy()
pb["pa"] = pb[["pff_sackAllowed", "pff_hitAllowed", "pff_hurryAllowed"]].max(axis=1)
bpff = pb.groupby("nflId").agg(pb_snaps=("pa", "size"), press_allowed=("pa", "mean"),
                               beaten=("pff_beatenByDefender", "mean")).reset_index()
df = df.merge(bpff, on="nflId", how="left"); df.to_csv(f"blocker_dose_rankings{otag}.csv", index=False)

q = df[df.pb_snaps >= 50].dropna(subset=["press_allowed"])
print(f"\n=== Blocker DOSE effect B_b vs PFF (n={len(q)}; expect NEGATIVE = good blocker) ===", flush=True)
print(f"  benchmarks: realized-impedance -0.23; dtheta first-difference -0.25 (filtered)", flush=True)
for tgt in ["press_allowed", "beaten"]:
    print(f"  effect vs {tgt:13s} Pearson={pearsonr(q.effect, q[tgt])[0]:+.3f}  Spearman={spearmanr(q.effect, q[tgt])[0]:+.3f}")
q2 = q.copy(); q2["e_dm"] = q2.effect - q2.groupby("pos").effect.transform("mean")
q2["pa_dm"] = q2.press_allowed - q2.groupby("pos").press_allowed.transform("mean")
print(f"  WITHIN-position (effect vs press_allowed, demeaned): Pearson={pearsonr(q2.e_dm, q2.pa_dm)[0]:+.3f}")
for pos in ["T", "G", "C"]:
    sdf = q[q.pos == pos]
    if len(sdf) > 10:
        print(f"    {pos}-only (n={len(sdf)}): Spearman={spearmanr(sdf.effect, sdf.press_allowed)[0]:+.3f}")
print("\ntop 10 blockers by B_b (best, >=50 snaps):", flush=True)
for _, x in q.sort_values("effect", ascending=False).head(10).iterrows():
    print(f"  {x['name']:22s} {x.pos:2s} B={x.effect:+.3f} [{x.lo:+.3f},{x.hi:+.3f}] pa={x.press_allowed:.2f} ({int(x.pb_snaps)} snaps)")
print("bottom 6 by B_b (worst):", flush=True)
for _, x in q.sort_values("effect").head(6).iterrows():
    print(f"  {x['name']:22s} {x.pos:2s} B={x.effect:+.3f} pa={x.press_allowed:.2f}")
