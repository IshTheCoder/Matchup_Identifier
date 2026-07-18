"""First-difference blocker model (ContinuousBlockerDeltaModel): DELTA STRAIN = STRAIN_{t+1}-STRAIN_t
~ intercept - sum_b dtheta(b,j) B_b. Differencing cancels the rusher fixed effect AND every play-level
control, so B_b is identified from how strain responds when a blocker's engagement SHIFTS. SVI fit;
per-blocker B_b validated vs PFF pressures allowed (raw + within-position)."""
import sys, time, pickle
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, "src"); sys.path.insert(0, "model")
import feature_engineering as fe
from models import ContinuousBlockerDeltaModel

# CLI: arg1 = assignment file (default smoothed); pass "mcmc" anywhere to use NUTS instead of SVI.
args = [a for a in sys.argv[1:] if a != "mcmc"]
USE_MCMC = "mcmc" in sys.argv
asg = args[0] if args else "assignment_data_phase25.csv"
tag = "_filtered" if "filtered" in asg else ""
otag = tag + ("_mcmc" if USE_MCMC else "")
pkl = f"continuous_design_delta{tag}_8wk.pkl"
try:
    data, enc = pickle.load(open(pkl, "rb")); assert "assignment_next" in data
    print(f"loaded {pkl}", flush=True)
except Exception:
    print(f"building delta design from {asg} ...", flush=True); t0 = time.time()
    data, enc = fe.build_continuous_design_from_files(weeks=range(8), assignment_path=asg)
    pickle.dump((data, enc), open(pkl, "wb")); print(f"  built in {time.time()-t0:.0f}s", flush=True)
print(f"  assignments: {asg}", flush=True)

# first-difference transform
data["outcome"] = data["outcome"] - data["strain_current"]
data["dtheta"] = data["assignment_next"] - data["assignment"]
N = len(data["outcome"]); nz = float(np.mean(np.abs(data["dtheta"]).sum(1) > 1e-6))
print(f"N_obs={N:,} N_blockers={data['N_blockers']}  dSTRAIN mean={data['outcome'].mean():+.4f} "
      f"std={data['outcome'].std():.4f}  rows w/ nonzero dtheta={nz:.2f}", flush=True)

m = ContinuousBlockerDeltaModel()
if USE_MCMC:
    print("inference: MCMC (NUTS, 1 chain, 800 warmup / 800 samples)", flush=True)
    m.run_mcmc_inference(data, num_warmup=800, num_samples=800, num_chains=1)
else:
    print("inference: SVI (AutoNormal, 20000 steps)", flush=True)
    m.run_svi_inference(data, num_steps=20000, lr=5e-3)
s = m.get_posterior_samples(); pickle.dump(s, open(f"blocker_delta_samples{otag}.pkl", "wb"))
print(f"  intercept={float(s['intercept'].mean()):+.4f}  sigma={float(s['sigma'].mean()):.4f}  "
      f"sigma_blocker={float(s['sigma_blocker'].mean()):.4f}", flush=True)

# per-blocker rating B_b = Bc @ a_B + sigma_blocker * z_blocker
draws = s["blocker_weight"] @ data["blocker_covariates"].T + s["sigma_blocker"][:, None] * s["z_blocker"]
inv = {v: k for k, v in enc["blocker"].items()}
players = pd.read_csv("data/players.csv").set_index("nflId")
df = pd.DataFrame({
    "nflId": [inv[i] for i in range(draws.shape[1])],
    "effect": draws.mean(0), "lo": np.percentile(draws, 2.5, 0), "hi": np.percentile(draws, 97.5, 0)})
df["name"] = df.nflId.map(players["displayName"]); df["pos"] = df.nflId.map(players["officialPosition"])
pff = pd.read_csv("data/pffScoutingData.csv"); pb = pff[pff.pff_role == "Pass Block"].copy()
pb["pa"] = pb[["pff_sackAllowed", "pff_hitAllowed", "pff_hurryAllowed"]].max(axis=1)
bpff = pb.groupby("nflId").agg(pb_snaps=("pa", "size"), press_allowed=("pa", "mean"),
                               beaten=("pff_beatenByDefender", "mean")).reset_index()
df = df.merge(bpff, on="nflId", how="left"); df.to_csv(f"blocker_delta_rankings{otag}.csv", index=False)

q = df[df.pb_snaps >= 50].dropna(subset=["press_allowed"])
print(f"\n=== Blocker DELTA effect B_b vs PFF (n={len(q)}; expect NEGATIVE = good blocker, fewer pressures) ===", flush=True)
print(f"  benchmarks: realized-impedance fn_value -0.23, real_imp -0.13", flush=True)
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
