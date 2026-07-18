"""Phase 2.5: fit the structured null transition {rho, p_fail per-player; rho_null per-position}
on all 8 weeks, holding the Phase-1 emission fixed."""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu"); sys.path.insert(0, "src")
import jax; jax.config.update("jax_enable_x64", True)
import baum_welch_jax as bwj
t0 = time.time()
print("Phase-2.5 structured-null transition fit: 8 weeks", flush=True)
bwj.fit_null_transition_by_position(
    csv_paths=[f"data/sample_data_week_{i}.csv" for i in range(8)],
    n_iter=15, c_b=-5.0, conc=1.0, orientation="vonmises", null_init=0.05,
    fitted_path="fitted_params_jax_phase1.csv", rho_by_player_path="rho_by_player_phase1.csv",
    out_path="fitted_params_jax_phase25.csv", out_player_path="rho_by_player_phase25.csv")
print(f"Phase-2.5 fit done in {time.time()-t0:.0f}s", flush=True)
