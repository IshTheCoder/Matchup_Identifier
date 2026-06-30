"""Phase 2.5, FILTERED (causal) assignments: emit the forward-only posterior P(state_t | obs_{1:t})
instead of the smoothed gamma, so the per-frame blocker->rusher assignment carries no look-ahead.
Same fitted params / structured null as run_phase25_assignments.py. -> assignment_data_phase25_filtered.csv"""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu"); sys.path.insert(0, "src")
import jax; jax.config.update("jax_enable_x64", True)
import produce_assignments as pa
t0 = time.time()
pa.produce_assignments(
    fitted_path="fitted_params_jax_phase25.csv",
    rho_by_player_path="rho_by_player_phase25.csv",
    out_path="assignment_data_phase25_filtered.csv", c_b=-5.0, filtered=True)
print(f"phase-2.5 FILTERED assignments done in {time.time()-t0:.0f}s", flush=True)
