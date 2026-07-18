"""Phase 2: regenerate assignment_data_phase2.csv with the disengaged (null) state layered on the
Phase-1 hierarchical fit (per-player rho), c_b = -5.0. Null rows carry nflId_pr = -1.
Phase-tagged (stable) — the robustness sweep reads this by name."""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu"); sys.path.insert(0, "src")
import jax; jax.config.update("jax_enable_x64", True)
import produce_assignments as pa
t0 = time.time()
pa.produce_assignments(
    fitted_path="fitted_params_jax_phase1.csv",
    rho_by_player_path="rho_by_player_phase1.csv",
    out_path="assignment_data_phase2.csv",
    c_b=-5.0,
)
print(f"phase-2 assignments done in {time.time()-t0:.0f}s", flush=True)
