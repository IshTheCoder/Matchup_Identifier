"""M3 step 1: build the 8-week pocket design (QB+rushers+blockers kinematics) WITH the phase-2.5 HMM
weights theta[n,b_slot,r_slot], pickle for the rusher/blocker space metrics."""
import sys, pickle
sys.path.insert(0, "src")
import feature_engineering as fe

data, enc = fe.build_pocket_design_from_files(weeks=range(8),
                                              assignment_path="assignment_data_phase25.csv")
pickle.dump((data, enc), open("pocket_design_8wk.pkl", "wb"))
print(f"N={len(data['x_qb']):,} frames  plays={len(data['play_start'])-1}  "
      f"Rmax={data['Rmax']} Bmax={data['Bmax']}  rushers={data['N_rushers']} blockers={data['N_blockers']}",
      flush=True)
th = data["theta"]
print(f"theta shape={th.shape}  nonzero entries={int((th>0).sum()):,}  "
      f"mean theta-sum per (frame,rusher)={th.sum(1).mean():.3f}", flush=True)
