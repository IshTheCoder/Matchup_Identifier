#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Run the FULL paper pipeline: every fit + table + figure the manuscript needs,
# in dependency order. Thin wrapper over run_model.sh.
#
#   playpm     HMM (steps 1-2) + play-level & 1.1M-frame dose plus-minus,
#              attention/entropy tables, rusher_2d figure
#   shedding   KM engagement/shed tables + opponent-adjusted hold model (NUTS)
#   pocket     Fernandez-Bornn pocket space control tables + pocket_control figure
#   robustness phase sweep -> phase_comparison.tex
#   validate   external PFF validation -> pff_validation.tex
#              (reads playpm + shedding outputs, so it MUST run last)
#
# The HMM is fit ONCE in `playpm`; the rest reuse its assignments via SKIP_HMM=1.
# GPU is the default backend. Overrides are passed straight through to run_model.sh:
#   JAX_PLATFORMS=cpu ./run_paper.sh     # force CPU for every fit
#   GPU_ID=1          ./run_paper.sh     # pick a different GPU
#   SKIP_HMM=1        ./run_paper.sh     # reuse existing assignments (skip the HMM even in playpm)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

# playpm fits the shared HMM unless the caller already set SKIP_HMM=1 in the environment.
./run_model.sh playpm

# everything after reuses playpm's assignments.
SKIP_HMM=1 ./run_model.sh shedding
SKIP_HMM=1 ./run_model.sh pocket
SKIP_HMM=1 ./run_model.sh robustness
SKIP_HMM=1 ./run_model.sh validate

echo ""
echo "=== paper pipeline complete: all fits, tables, and figures regenerated ==="
