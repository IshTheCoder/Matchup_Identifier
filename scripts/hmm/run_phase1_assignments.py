"""Regenerate assignment_data_phase1.csv from the Phase-1 hierarchical HMM fit (per-player rho).
Phase-tagged (stable) — the robustness sweep reads this by name."""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "src")
import jax; jax.config.update("jax_enable_x64", True)
import produce_assignments as pa
t0 = time.time()
pa.produce_assignments(
    fitted_path="fitted_params_jax_phase1.csv",
    rho_by_player_path="rho_by_player_phase1.csv",
    out_path="assignment_data_phase1.csv",
)
print(f"phase-1 assignments done in {time.time()-t0:.0f}s", flush=True)
