"""M3 steps 2-3: 8-week per-player pocket metrics.
 - rusher SPACE-WON  = control share over the QB-danger region (yd^2), per frame.
 - blocker SPACE-DENIED = theta-weighted leave-one-out: sum_j theta(b,j)*(S_j^{-b} - S_j), the
   assignment-weighted reduction in his assigned rushers' space-won when blocker b's influence is
   removed (reuses the per-frame influences -- pure share algebra, no extra exp calls).
Accumulate per (player, play) for bootstrap-over-plays CIs. Benchmarks vs PFF; writes leaderboards."""
import sys, pickle
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, "src")
import pitch_control as pc

data, enc = pickle.load(open("pocket_design_8wk.pkl", "rb"))
N = len(data["x_qb"]); RAD = 5.0; pr = data["play_row"]; theta = data["theta"]
inv_rush = {v: k for k, v in enc["rusher"].items()}
inv_blk = {v: k for k, v in enc["blocker"].items()}

# (player, play) -> [sum, cnt]
r_space = defaultdict(lambda: [0.0, 0]); r_dist = defaultdict(lambda: [0.0, 0])
b_den = defaultdict(lambda: [0.0, 0])
b_rs = defaultdict(lambda: [0.0, 0.0])   # simpler variant: [sum_j theta*S_j, sum_j theta]
for n in range(N):
    G, gx, gy, cell = pc.qb_grid(data["x_qb"][n], half=6.0, n=40)
    inreg = ((G[:, 0] - data["x_qb"][n, 0]) ** 2 + (G[:, 1] - data["x_qb"][n, 1]) ** 2) <= RAD ** 2
    Ir, ri, Ib, bi, Iqb = pc.frame_fields(G, data, n)
    if len(ri) == 0:
        continue
    p = int(pr[n])
    Ir_r = Ir[:, inreg]; Iqb_r = Iqb[inreg]
    Ib_r = Ib[:, inreg] if len(bi) else np.zeros((0, inreg.sum()))
    tot = Ir_r.sum(0) + Ib_r.sum(0) + Iqb_r + 1e-9                  # total influence over region
    S = cell * (Ir_r / tot).sum(1)                                 # rusher space-won (Kr,)
    for j, slot in enumerate(ri):
        rid = int(data["rush_slot_id"][n, slot])
        r_space[(rid, p)][0] += S[j]; r_space[(rid, p)][1] += 1
        r_dist[(rid, p)][0] += float(data["d_rush"][n, slot]); r_dist[(rid, p)][1] += 1
    if len(bi):                                                    # blocker space-denied (LOO)
        tot_b = tot[None, :] - Ib_r                                # (Kb,M) remove blocker b
        S_b = cell * (Ir_r[None] / tot_b[:, None]).sum(2)          # (Kb,Kr) rusher space w/o b
        denied = np.maximum(S_b - S[None, :], 0.0)                 # (Kb,Kr)
        th = theta[n][np.ix_(bi, ri)]                              # (Kb,Kr) HMM weights
        sd = (th * denied).sum(1)                                  # (Kb,) weighted space denied (LOO)
        wsp = (th * S[None, :]).sum(1); wt = th.sum(1)             # (Kb,) weighted assigned-rusher space level
        for i, slot in enumerate(bi):
            bid = int(data["blk_slot_id"][n, slot])
            b_den[(bid, p)][0] += sd[i]; b_den[(bid, p)][1] += 1
            b_rs[(bid, p)][0] += wsp[i]; b_rs[(bid, p)][1] += wt[i]

def per_player(d):                                                 # {(id,play):[sum,cnt]} -> id->list of play-means
    out = defaultdict(list)
    for (pid, _), (s, c) in d.items():
        if c:
            out[pid].append(s / c)
    return out

def boot_ci(vals, B=400):
    v = np.array(vals)
    if len(v) < 3:
        return v.mean(), np.nan, np.nan
    idx = np.random.randint(0, len(v), (B, len(v)))
    m = v[idx].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

rpm = per_player(r_space); rdm = per_player(r_dist); bpm = per_player(b_den); bsm = per_player(b_rs)
players = pd.read_csv("data/players.csv")[["nflId", "displayName", "officialPosition"]]

rr = pd.DataFrame([(inv_rush[r], *boot_ci(v), len(v),
                    np.mean(rdm.get(r, [np.nan]))) for r, v in rpm.items()],
                  columns=["nflId", "space_won", "lo", "hi", "plays", "mean_dist"]).merge(players, on="nflId", how="left")
rsl = {b: np.mean(v) for b, v in bsm.items()}                      # simpler: assigned-rusher space level
bb = pd.DataFrame([(inv_blk[b], *boot_ci(v), len(v), rsl.get(b, np.nan)) for b, v in bpm.items()],
                  columns=["nflId", "space_denied", "lo", "hi", "plays", "rusher_space"]).merge(players, on="nflId", how="left")

pff = pd.read_csv("data/pffScoutingData.csv")
prr = pff[pff.pff_role == "Pass Rush"].copy(); prr["pr"] = prr[["pff_sack", "pff_hit", "pff_hurry"]].max(axis=1)
rrate = prr.groupby("nflId").agg(press_rate=("pr", "mean"), pff_snaps=("pr", "size")).reset_index()
pbb = pff[pff.pff_role == "Pass Block"].copy()
pbb["pa"] = pbb[["pff_sackAllowed", "pff_hitAllowed", "pff_hurryAllowed"]].max(axis=1)
brate = pbb.groupby("nflId").agg(press_allowed=("pa", "mean"), pff_snaps=("pa", "size")).reset_index()
rr = rr.merge(rrate, on="nflId", how="left"); bb = bb.merge(brate, on="nflId", how="left")
rr.to_csv("pocket_rusher_space.csv", index=False); bb.to_csv("pocket_blocker_space.csv", index=False)

print("=== M3 rusher SPACE-WON vs PFF pressure rate (>=20 plays) ===", flush=True)
q = rr[rr.plays >= 20].dropna(subset=["press_rate"])
print(f"  n={len(q)}  Pearson={pearsonr(q.space_won, q.press_rate)[0]:+.3f}  "
      f"Spearman={spearmanr(q.space_won, q.press_rate)[0]:+.3f}  (benchmark plus-minus +0.67)")
print(f"  circularity corr(space_won, mean_dist)={pearsonr(q.space_won, q.mean_dist)[0]:+.3f}")
print("\n=== M3 blocker SPACE-DENIED vs PFF pressures-allowed (>=20 plays) -- expect NEGATIVE ===", flush=True)
b = bb[bb.plays >= 20].dropna(subset=["press_allowed"])
print(f"  LOO space_denied: n={len(b)}  Pearson={pearsonr(b.space_denied, b.press_allowed)[0]:+.3f}  "
      f"Spearman={spearmanr(b.space_denied, b.press_allowed)[0]:+.3f}  (benchmark realized-impedance -0.23)")
b2 = b.dropna(subset=["rusher_space"])
print(f"  simpler assigned-rusher-space level (LOW=good -> expect POSITIVE vs pressures-allowed): "
      f"n={len(b2)}  Pearson={pearsonr(b2.rusher_space, b2.press_allowed)[0]:+.3f}  "
      f"Spearman={spearmanr(b2.rusher_space, b2.press_allowed)[0]:+.3f}")
print("\ntop 15 rushers (space-won, >=20 plays):")
for _, x in q.sort_values("space_won", ascending=False).head(15).iterrows():
    print(f"  {x.displayName:22s} {x.officialPosition:3s} {x.space_won:5.2f} [{x.lo:.2f},{x.hi:.2f}]  press={x.press_rate:.2f}  plays={int(x.plays)}")
print("\ntop 15 blockers (space-denied, >=20 plays):")
for _, x in b.sort_values("space_denied", ascending=False).head(15).iterrows():
    print(f"  {x.displayName:22s} {x.officialPosition:3s} {x.space_denied:5.3f} [{x.lo:.3f},{x.hi:.3f}]  pa={x.press_allowed:.2f}  plays={int(x.plays)}")
