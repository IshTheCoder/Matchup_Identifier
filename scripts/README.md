# scripts/

Driver/analysis scripts, grouped by functionality. Library modules live in `src/` and `model/`.

**Run everything from the repository root**, e.g. `python3 scripts/plusminus/run_play_model.py`.
The scripts use root-relative paths (`data/`, `tables/`, `figures/`, and `sys.path` insert of
`src`/`model`), so the working directory must be the repo root, not the script's own folder.

| folder | what's in it |
|---|---|
| `hmm/` | HMM fit + assignment production (phases 1/2/2.5, filtered), null-state + phase-comparison analysis, and the attention/entropy/tau table+figure generator |
| `plusminus/` | play-level and continuous-time STRAIN plus-minus models (incl. blocker delta/dose), ranking + realized-impedance + comparison table generators, two-dimensions figure |
| `shedding/` | block-shedding survival models and the hold/shed tables |
| `spacecontrol/` | Fernández–Bornn pocket space control: build, per-player metrics, field viz, space-won / generation / concession + their tables |
| `openness/` | all-22 downfield space control and receiver "openness" |
| `qb/` | QB force-field / escape / field / pressure models and the containment-break check (parked negative results) |
| `validation/` | external validation against PFF charting; descriptive attention |

Figures can also be produced in R via `src/make_figures.R` (reads `fitted_params_jax.csv` and
`rusher_2d.csv`).

## Parquet model outputs (`model_outputs/`)

For cross-language (R) access, fitted MCMC effects are exported as typed parquet under
`model_outputs/` (read in R with the `arrow` package). Posterior pickles stay for Python reuse; the
parquet is the portable surface. Each per-player effect writes two files:

- `{name}_summary.parquet` — one row per player: `nflId, name, pos, mean, lo, hi[, snaps]`
- `{name}_draws.parquet` — tidy posterior draws: `nflId, draw, value`

`scripts/export_parquet.py` produces these from the already-fit pickles (no re-run); the drivers
`plusminus/run_phase_play_model.py` and `plusminus/run_blocker_dose_model.py` also write them on each
fit (`playpm_{rusher,blocker,quarterback}_<phase>`, `dose_blocker[_baseline]`). Helper: `src/model_io.py`.
(`model_outputs/` is git-ignored like the other generated artifacts.)
