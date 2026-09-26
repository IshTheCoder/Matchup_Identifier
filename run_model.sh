#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline:
#   1. run HMM            (fit emission + structured-null transition, phase 2.5)
#   2. export HMM params  (produce assignment_data_phase25.csv — what the models consume)
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
CTR_R="r-new-football"      # R container:   make_figures.R, make_ridge_figures.R
CONTAINER_WORKDIR="/home/joyvan/work"
EXEC="docker exec -w $CONTAINER_WORKDIR"

GPU_ID="${GPU_ID:-0}"               # GPU the fits run on (override: GPU_ID=1 ./run_model.sh ...)
SKIP_HMM="${SKIP_HMM:-0}"           # 1 → reuse existing assignments, skip steps 1–2 (HMM is shared/expensive)
# jax backend for the GPU-container fits. Default 'cuda' (the cuda12 plugin is aligned to jaxlib 0.10.1);
# this is passed through so it OVERRIDES each driver's `setdefault("JAX_PLATFORMS","cpu")` standalone
# default. Force CPU for the whole run with: JAX_PLATFORMS=cpu ./run_model.sh <model>
JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

# ── Fixed HMM stage (steps 1–2) ───────────────────────────────────────────────
# run HMM: phase-1 hierarchical emission fit → phase-2.5 structured-null transition fit
HMM_RUN=(
    "scripts/hmm/run_phase1_hmm.py"
    "scripts/hmm/run_phase25_fit.py"
)
# export HMM params: regenerate the assignments from the phase-2.5 params, then the
# HMM-level tables that every downstream model shares.
HMM_EXPORT=(
    "scripts/hmm/run_phase25_assignments.py"   # -> assignment_data_phase25.csv (smoothed posterior; the canonical production assignments)
    "scripts/hmm/run_phase25_filtered.py"      # -> assignment_data_phase25_filtered.csv (causal/forward-only; the continuous dose model needs this)
    "scripts/hmm/make_tables_figures.py"       # attention_rankings.tex, blocker_entropy.tex (+ tau figure) from assignment_data_phase25.csv
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
                # continuous-time dose model (Sec. 4.5) on the causal FILTERED assignments;
                # (+ rusher/QB random intercepts and the phi*STRAIN_t control); writes
                # blocker_dose_rankings_filtered_baseline_mcmc.csv + model_outputs/dose_*_baseline_*
                "scripts/plusminus/run_blocker_dose_model.py assignment_data_phase25_filtered.csv baseline mcmc"
            )
            TABLES=(
                "scripts/plusminus/regenerate_rankings.py"        # rusher/blocker plusminus (+ _bot), qb_suppression
                "scripts/plusminus/realized_impedance.py"         # blocker_value.tex (+ blocker_value.csv)
            )
            FIGS_PY=(
                "scripts/plusminus/rusher_2d.py"        # attention vs continuous R^Delta_j -> rusher_2d.csv (input to make_figures.R)
            )
            FIGS_R=(
                "src/make_figures.R"                    # tau_positions + rusher_2d figures
            )
            ;;
        shedding) # opponent-adjusted block-failure survival (hold / shed ratings)
                  # Run after playpm: the hold model's opponent covariate is the play-level R_j
                  # (model_outputs/playpm_rusher_phase25_summary.parquet).
            RUN=(
                "src/survival_metrics.py"                     # block_engagement.tex, rusher_shedding.tex (KM engagement/shed) + block_failure_events.csv
                "scripts/shedding/run_block_survival_model.py mcmc"   # exponential crossed hold/shed frailty via NUTS (not the paper's hold model)
                "scripts/shedding/run_block_hold_discrete.py" # the paper's discrete-time logistic hold hazard (NUTS 4x1000/1000) -> block_hold_discrete_{samples.pkl,ratings.csv}
            )
            TABLES=(
                "scripts/shedding/block_hold_tables.py"       # block_hold.tex (opponent-adjusted hold rating)
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
        robustness) # appendix robustness sweep: refit the play model across assignment phases,
                    # then compare. STEP 3 is handled specially (see run_phase_sweep) so it can
                    # capture each phase's stdout to phase_play_<phase>.log for compare_phases.py.
                    # Requires the per-phase assignment files (assignment_data_phase{0,1,2,25}.csv)
                    # and the committed phase-0 baseline (phase0_results/, play_model_samples.pkl).
            RUN=()                                            # STEP 3 uses run_phase_sweep, not RUN
            TABLES=(
                "scripts/hmm/compare_phases.py"               # phase_comparison.tex (+ .csv + figure)
            )
            FIGS_PY=()
            FIGS_R=()
            ;;
        validate) # appendix external validation vs PFF charting (no model fit of its own)
                  # Run after playpm + shedding so the ranking/survival CSVs it reads are present.
            RUN=()
            TABLES=(
                "scripts/validation/validate_pff.py"          # pff_validation.tex (+ printed correlations/agreement)
            )
            FIGS_PY=()
            FIGS_R=()
            ;;
        ridges)   # ridgeline (caterpillar) posterior figures for every CI-bearing paper table.
                  # Fits nothing: reads the play-level + continuous-time posteriors (playpm) AND the
                  # block-hold posterior (shedding), so it MUST run after both. STEP 4 refreshes the
                  # model_outputs/ parquet that the exporter reads.
            RUN=()
            TABLES=()
            FIGS_PY=(
                "scripts/export_ridge_draws.py"               # -> model_outputs/ridge_draws.parquet (owns player selection)
            )
            FIGS_R=(
                "src/make_ridge_figures.R"                    # -> figures/ridge_*.{png,pdf}, incl. the combined per-table panels
            )
            ;;
        *)
            echo "ERROR: unknown model '$1'." >&2
            echo "Known models: playpm shedding pocket openness qbforce robustness validate ridges" >&2
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
    echo "Known models: playpm shedding pocket openness qbforce robustness validate ridges"
    exit 0
fi

if [[ $# -ge 1 ]]; then
    MODEL="$1"
else
    echo "Model to run (playpm | shedding | pocket | openness | qbforce | robustness | validate | ridges):"
    read -rp "? " MODEL
fi
[[ -z "${MODEL:-}" ]] && { echo "No model given — exiting."; exit 1; }

select_model "$MODEL"

echo ""
echo "Model    : $MODEL"
echo "GPU      : $GPU_ID  (container '$CTR_GPU')"
echo "JAX      : $JAX_PLATFORMS  (fit backend; force CPU with JAX_PLATFORMS=cpu ./run_model.sh ...)"
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
            # word-split the command (script path + its args); CUDA + jax backend pinned for the GPU container
            $EXEC -e "CUDA_VISIBLE_DEVICES=$GPU_ID" -e "JAX_PLATFORMS=$JAX_PLATFORMS" "$ctr" python3 $cmd
        else
            $EXEC "$ctr" $interp $cmd
        fi
    done
}

# ── Robustness phase sweep (STEP 3 for the 'robustness' model) ─────────────────
# Refit the play-level model on each refinement phase's assignments, teeing each run's
# stdout (in-container, so it lands in $CONTAINER_WORKDIR) to phase_play_<phase>.log —
# compare_phases.py parses sigma from those logs. Phase 0 (raw) is the committed baseline
# in phase0_results/ and is not refit here.
run_phase_sweep() {
    echo "=== [$(date '+%H:%M:%S')] STEP 3: play-model phase sweep (robustness) ==="
    local phase asg
    for phase in phase1 phase2 phase25; do
        asg="assignment_data_${phase}.csv"
        echo "  [$CTR_GPU] python3 scripts/plusminus/run_phase_play_model.py $phase $asg  (-> phase_play_${phase}.log)"
        $EXEC -e "CUDA_VISIBLE_DEVICES=$GPU_ID" -e "JAX_PLATFORMS=$JAX_PLATFORMS" "$CTR_GPU" bash -c \
            "set -o pipefail; python3 scripts/plusminus/run_phase_play_model.py $phase $asg 2>&1 | tee phase_play_${phase}.log"
    done
}

# ── Pipeline ──────────────────────────────────────────────────────────────────
if [[ $SKIP_HMM != 1 ]]; then
    run_stage "STEP 1: run HMM"           "$CTR_GPU" python3 "${HMM_RUN[@]}"
    run_stage "STEP 2: export HMM params" "$CTR_GPU" python3 "${HMM_EXPORT[@]}"
else
    echo "=== STEP 1–2: HMM skipped (SKIP_HMM=1) ==="
fi

if [[ "$MODEL" == robustness ]]; then
    run_phase_sweep
else
    run_stage "STEP 3: run model '$MODEL'" "$CTR_GPU" python3 "${RUN[@]}"
fi

# STEP 4 re-exports the fitted-effect parquet from the play/dose pickles. The analysis-only
# models (robustness, validate) fit nothing new, so skip it for them.
if [[ "$MODEL" == robustness || "$MODEL" == validate ]]; then
    echo "  (STEP 4: export model params — skipped for analysis-only model '$MODEL')"
else
    run_stage "STEP 4: export model params" "$CTR_GPU" python3 "$MODEL_EXPORT"
fi

run_stage "STEP 5: produce tables"        "$CTR_GPU" python3 "${TABLES[@]}"
run_stage "STEP 6a: produce figures (py)" "$CTR_GPU" python3 "${FIGS_PY[@]}"
run_stage "STEP 6b: produce figures (R)"  "$CTR_R"   Rscript  "${FIGS_R[@]}"

echo ""
echo "=== [$(date '+%H:%M:%S')] Pipeline complete for '$MODEL' ==="
