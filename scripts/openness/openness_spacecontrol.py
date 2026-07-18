"""S2 step 1: compute per-frame receiver OPENNESS over all 8 weeks from the all-22 space-control
field, week-by-week to bound memory. -> openness_spacecontrol.csv (joined into the QB release design)."""
import sys
import pandas as pd
sys.path.insert(0, "src")
import feature_engineering as fe
import openness as op

tabs = []
for w in range(1, 9):
    data, enc = fe.build_alltwentytwo_design_from_files(weeks=range(w, w + 1))
    t = op.frame_openness_table(data)
    tabs.append(t)
    print(f"  week {w}: {len(t):,} frames  mean open_sum={t.open_sum.mean():.3f}", flush=True)
allt = pd.concat(tabs, ignore_index=True)
allt.to_csv("openness_spacecontrol.csv", index=False)
print(f"wrote openness_spacecontrol.csv ({len(allt):,} frames)", flush=True)
