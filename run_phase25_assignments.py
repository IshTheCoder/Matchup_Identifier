"""Phase 2.5: regenerate assignment_data.csv with the structured null transition
(per-player rho + p_fail, per-position rho_null), c_b=-5.0."""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu"); sys.path.insert(0, "src")
import jax; jax.config.update("jax_enable_x64", True)
import produce_assignments as pa
t0 = time.time()
pa.produce_assignments(
    fitted_path="fitted_params_jax_phase25.csv",
    rho_by_player_path="rho_by_player_phase25.csv",
    out_path="assignment_data.csv", c_b=-5.0)
print(f"phase-2.5 assignments done in {time.time()-t0:.0f}s", flush=True)
