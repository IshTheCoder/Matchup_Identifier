"""Benchmark HMM fits under different emission / rho-estimator choices.

Loads the per-position voxels ONCE (the expensive step), then re-fits the same data
under several configurations -- the closed-form EM is fast, so the marginal cost of
each extra config is small:

  orientation in {none, current, paper}      (position-only / legacy Beta / paper Beta)
  rho_mode    in {mle, play_normalized}       (pool transitions / weight each play 1/t_i)

Writes benchmark_params.csv and prints a per-position comparison so we can see how the
paper-faithful Beta and the play-length normalization move tau_r / sigma / rho relative
to what we currently have in fitted_params_jax.csv.

Run:  python src/benchmark_orientation.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

import jax

jax.config.update("jax_enable_x64", True)

import baum_welch_jax as bwj

POSITIONS = ("T", "C", "RB", "TE", "WR", "FB", "G")

# (label, orientation, rho_mode, conc). von Mises kappa from the sweep: kappa=1 keeps
# location dominant for every position (worst-case RB ratio 0.31); kappa=2 lets
# orientation matter more while staying dominant.
CONFIGS = [
    ("current/mle", "current", "mle", 2.0),
    ("none/mle", "none", "mle", 2.0),
    ("vonmises_k1/mle", "vonmises", "mle", 1.0),
    ("vonmises_k2/mle", "vonmises", "mle", 2.0),
    ("paper_a0.1/mle", "paper", "mle", 0.1),
    ("vonmises_k1/play_norm", "vonmises", "play_normalized", 1.0),
]


def run_benchmark(csv_paths=None, n_iter=15, tau0=(0.8, 0.2), sigma0=2.0, rho0=0.95):
    voxels = bwj.load_position_voxels(csv_paths, POSITIONS)

    rows = []
    for position in POSITIONS:
        B_list, O_list, D_list, _ = voxels[position]
        n_poss = len(B_list)
        if not B_list:
            print(f"{position}: no usable possessions; skipping", flush=True)
            continue
        groups = bwj.group_and_pad(B_list, O_list, D_list)  # built once, reused per config
        for label, orientation, rho_mode, conc in CONFIGS:
            params = bwj.init_params(tau0, sigma0, rho0)
            params, _, lls = bwj.run_em(params, groups, n_iter, orientation, rho_mode, conc)
            tau = np.asarray(params.tau)
            rows.append({
                "position": position,
                "config": label,
                "orientation": orientation,
                "rho_mode": rho_mode,
                "tau_r": round(float(tau[0]), 4),
                "tau_q": round(float(tau[1]), 4),
                "sigma": round(float(params.sigma), 4),
                "rho": round(float(params.rho), 4),
                "n_possessions": n_poss,
                "final_loglik": round(float(lls[-1]), 1),
            })
            print(f"{position:>3} {label:<18} tau_r={tau[0]:.3f} sigma={float(params.sigma):.3f} "
                  f"rho={float(params.rho):.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("benchmark_params.csv", index=False)
    print("\nwrote benchmark_params.csv", flush=True)

    # readable per-position comparison of the headline parameters
    print("\n=== tau_r (rusher weight) by config ===")
    print(df.pivot(index="position", columns="config", values="tau_r").to_string())
    print("\n=== sigma by config ===")
    print(df.pivot(index="position", columns="config", values="sigma").to_string())
    print("\n=== rho by config ===")
    print(df.pivot(index="position", columns="config", values="rho").to_string())
    return df


if __name__ == "__main__":
    run_benchmark()
