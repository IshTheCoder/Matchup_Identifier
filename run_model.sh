#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline:
#   1. run HMM            (fit emission + structured-null transition, phase 2.5)
#   2. export HMM params  (produce assignment_data.csv — what the models consume)
#   3. run {model}        (model-specific fit driver)
#   4. export model params(parquet under model_outputs/ for cross-language access)
#   5. produce tables     (model-specific .tex)
#   6. produce figures    (model-specific; Python figs in the GPU container,
#                          R figs in the R container)
#
# Steps 1–2 are fixed. Steps 3–6 are selected from the per-model registry below.
# ──────────────────────────────────────────────────────────────────────────────

# ── Containers / workdir ──────────────────────────────────────────────────────
CTR_GPU="football"          # GPU container: HMM, model fits, exports, Python figures
CTR_R="r-new-football"      # R container:   make_figures.R
CONTAINER_WORKDIR="/home/joyvan/work"
EXEC="docker exec -w $CONTAINER_WORKDIR"

GPU_ID="${GPU_ID:-0}"               # GPU the fits run on (override: GPU_ID=1 ./run_model.sh ...)
SKIP_HMM="${SKIP_HMM:-0}"           # 1 → reuse existing assignments, skip steps 1–2 (HMM is shared/expensive)

# ── Fixed HMM stage (steps 1–2) ───────────────────────────────────────────────
# run HMM: phase-1 hierarchical emission fit → phase-2.5 structured-null transition fit
HMM_RUN=(
    "scripts/hmm/run_phase1_hmm.py"
    "scripts/hmm/run_phase25_fit.py"
)
# export HMM params: regenerate assignment_data.csv from the phase-2.5 params
HMM_EXPORT=(
    "scripts/hmm/run_phase25_assignments.py"
)

# ── Shared model-params export (step 4) ───────────────────────────────────────
# Drivers already write parquet on each fit; this re-exports from the pickles to be safe.
MODEL_EXPORT="scripts/export_parquet.py"

# ── Per-model registry (steps 3, 5, 6) ────────────────────────────────────────
# Each model sets four arrays. Empty array = that stage is skipped.
#   RUN     : fit driver(s)          — run in $CTR_GPU (python3)
#   TABLES  : .tex generator(s)      — run in $CTR_GPU (python3)
#   FIGS_PY : matplotlib figure(s)   — run in $CTR_GPU (python3)
#   FIGS_R  : R figure(s)            — run in $CTR_R   (Rscript)
# Add a model by adding a case branch. List the known models with: ./run_model.sh --list
select_model() {
    case "$1" in
        playpm)   # play-level (phase 2.5) STRAIN plus-minus: rusher / blocker / QB effects
            RUN=(
                "scripts/plusminus/run_phase_play_model.py phase25 assignment_data_phase25.csv"
            )
            TABLES=(
                "scripts/plusminus/regenerate_rankings.py"
                "scripts/plusminus/realized_impedance.py"
            )
            FIGS_PY=(
                "scripts/plusminus/rusher_2d.py"        # writes rusher_2d.csv (input to make_figures.R)
            )
            FIGS_R=(
                "src/make_figures.R"                    # tau_positions + rusher_2d figures
            )
            ;;
        shedding) # opponent-adjusted block-failure survival (hold / shed ratings)
            RUN=(
                "scripts/shedding/run_block_survival_model.py"
            )
            TABLES=(
                "scripts/shedding/block_hold_tables.py"
            )
            FIGS_PY=()
            FIGS_R=()
            ;;
        pocket)   # Fernández–Bornn pocket space control: rusher/blocker space, gain/gen/concede
            RUN=(
                "scripts/spacecontrol/pocket_build.py"
                "scripts/spacecontrol/pocket_metrics.py"
                "scripts/spacecontrol/pocket_space_gain.py"
                "scripts/spacecontrol/pocket_space_gen.py"
                "scripts/spacecontrol/pocket_space_concede.py"
            )
            TABLES=(
                "scripts/spacecontrol/pocket_rankings.py"
                "scripts/spacecontrol/pocket_sgg_tables.py"
                "scripts/spacecontrol/pocket_sog_tables.py"
                "scripts/spacecontrol/pocket_concede_tables.py"
            )
            FIGS_PY=(
                "scripts/spacecontrol/pocket_field_viz.py"
            )
            FIGS_R=()
            ;;
        openness) # all-22 downfield space control / receiver openness (8 weeks)
            RUN=(
                "scripts/openness/openness_spacecontrol.py"
            )
            TABLES=()
            FIGS_PY=(
                "scripts/openness/alltwentytwo_field_viz.py"
            )
            FIGS_R=()
            ;;
        qbforce)  # structural QB force-field milestone (go/no-go fit; no tables/figures)
            RUN=(
                "scripts/qb/run_qb_force_model.py"
            )
            TABLES=()
            FIGS_PY=()
            FIGS_R=()
            ;;
        *)
            echo "ERROR: unknown model '$1'." >&2
            echo "Known models: playpm shedding pocket openness qbforce" >&2
            return 1
            ;;
    esac
}

# ── Parse args ────────────────────────────────────────────────────────────────
# Usage:
#   ./run_model.sh                 # interactive: prompt for the model
#   ./run_model.sh <model>         # run the full pipeline for <model>
#   ./run_model.sh --list          # list known models
#   SKIP_HMM=1 ./run_model.sh playpm   # reuse existing assignments
#   GPU_ID=1  ./run_model.sh pocket
if [[ "${1:-}" == "--list" ]]; then
    echo "Known models: playpm shedding pocket openness qbforce"
    exit 0
fi

if [[ $# -ge 1 ]]; then
    MODEL="$1"
else
    echo "Model to run (playpm | shedding | pocket | openness | qbforce):"
    read -rp "? " MODEL
fi
[[ -z "${MODEL:-}" ]] && { echo "No model given — exiting."; exit 1; }

select_model "$MODEL"

echo ""
echo "Model    : $MODEL"
echo "GPU      : $GPU_ID  (container '$CTR_GPU')"
echo "R figures: container '$CTR_R'"
echo "HMM      : $([[ $SKIP_HMM == 1 ]] && echo 'SKIPPED (reusing assignments)' || echo 'run + export')"
echo ""

# ── Container health check ────────────────────────────────────────────────────
need_r=$([[ ${#FIGS_R[@]} -gt 0 ]] && echo "$CTR_R" || echo "")
for ctr in "$CTR_GPU" $need_r; do
    docker inspect --format='{{.State.Running}}' "$ctr" 2>/dev/null \
        | grep -q true \
        || { echo "ERROR: container '$ctr' is not running"; exit 1; }
done
echo "Containers running."
echo ""

# ── Stage runner ──────────────────────────────────────────────────────────────
# run_stage "<label>" <container> <interpreter> "<cmd>" "<cmd>" ...
run_stage() {
    local label="$1" ctr="$2" interp="$3"; shift 3
    (( $# == 0 )) && { echo "  ($label: nothing to do)"; return 0; }
    echo "=== [$(date '+%H:%M:%S')] $label ==="
    local cmd
    for cmd in "$@"; do
        echo "  [$ctr] $interp $cmd"
        if [[ "$interp" == "python3" ]]; then
            # word-split the command (script path + its args); CUDA pinned for the GPU container
            $EXEC -e "CUDA_VISIBLE_DEVICES=$GPU_ID" "$ctr" python3 $cmd
        else
            $EXEC "$ctr" $interp $cmd
        fi
    done
}

# ── Pipeline ──────────────────────────────────────────────────────────────────
if [[ $SKIP_HMM != 1 ]]; then
    run_stage "STEP 1: run HMM"           "$CTR_GPU" python3 "${HMM_RUN[@]}"
    run_stage "STEP 2: export HMM params" "$CTR_GPU" python3 "${HMM_EXPORT[@]}"
else
    echo "=== STEP 1–2: HMM skipped (SKIP_HMM=1) ==="
fi

run_stage "STEP 3: run model '$MODEL'"    "$CTR_GPU" python3 "${RUN[@]}"
run_stage "STEP 4: export model params"   "$CTR_GPU" python3 "$MODEL_EXPORT"
run_stage "STEP 5: produce tables"        "$CTR_GPU" python3 "${TABLES[@]}"
run_stage "STEP 6a: produce figures (py)" "$CTR_GPU" python3 "${FIGS_PY[@]}"
run_stage "STEP 6b: produce figures (R)"  "$CTR_R"   Rscript  "${FIGS_R[@]}"

echo ""
echo "=== [$(date '+%H:%M:%S')] Pipeline complete for '$MODEL' ==="
