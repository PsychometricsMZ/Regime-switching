# Regime-Switching Simulation Study

## Overview

This repository contains Python code for the **simulation study** of a regime-switching state-space model estimated by an **integrated (marginal maximum likelihood) extended Kim filter** — the frequentist counterpart to the NDLC-SEM framework of Kelava et al. (2022). The person-specific random intercept residuals are carried in an augmented state vector and marginalized under their model-implied N(0, Q2) distribution; the variance component Q2 is estimated by maximum likelihood jointly with all other parameters (`sim_filtering_integrated.py`; Section 4.6 of the manuscript).

The simulation evaluates parameter recovery, latent state forecasting, and regime classification under a 2x2 factorial design (sample size x estimation length), plus a restricted-model comparison in which the within-person transition predictors are omitted.

The results reported in the manuscript come from the **integrated, uncentered** runs (`SIM_INTEGRATED=1`, `CENTER_MS_PREDICTOR=0`, `SIM_TAG=g35_uncent`): the transition predictor is the raw regime-1 conditional filtered mean of eta1, matching the data-generating process and the Kelava et al. (2022) parameterization.

The accompanying **empirical study** code is stored locally only and is not synced to GitHub.

## Repository Structure

```
Regime-switching/
├── README.md
├── .gitignore
└── Codes/
    └── Simulation/
        ├── sim_main.py                     # Main entry point (parallel simulation loop)
        ├── sim_filtering_integrated.py     # Integrated (marginal ML) extended Kim filter — estimator used in the manuscript
        ├── sim_filtering.py                # Earlier non-marginalized variant (retained for reference; not used in the manuscript)
        ├── sim_data_generation.py          # Synthetic data generation
        ├── sim_summarize.py                # Aggregation and summary statistics
        ├── sim_evaluation.py               # Evaluation metrics, tables, and figures
        ├── sim_utils.py                    # Utility functions
        ├── config.py                       # Simulation configuration
        ├── calibrate_gamma1.py             # Data-only grid search used to choose gamma1 = 3.50
        ├── make_appendix_tables.py         # Regenerates LaTeX bodies for the appendix tables
        ├── make_talk_score_overlay.py      # Score-function overlay figure for talks
        ├── run_test.py                     # Small smoke test (local or cluster)
        ├── parameter_estimates_kelava.csv  # Population (true) values from Kelava et al. (2022)
        ├── kelava_init_params_sim.csv      # Warm-start distribution for parameter initialization
        ├── requirements.txt                # Python dependencies
        ├── setup_env.sh                    # bwcluster venv setup (run once)
        ├── submit_sim_integrated.sh        # SLURM array job: integrated estimator (centered variant)
        ├── submit_sim_uncentered.sh        # SLURM array job: integrated + uncentered — the production run
        ├── submit_sim_reduced_integrated.sh# SLURM array job: restricted model (gamma3 = gamma4 = 0)
        ├── submit_sim_g35.sh               # SLURM array job: legacy non-marginalized run (superseded)
        ├── submit_test.sh                  # SLURM job: smoke test
        ├── submit_postprocess.sh           # SLURM job: postprocessing
        ├── comparison/                     # Full vs. restricted metrics, gamma1 calibration, appendix .tex bodies
        │   └── appendix_tex_uncent/        #   final appendix bodies used in the manuscript
        ├── sim_summary_integrated_g35_uncent/   # PRODUCTION summaries (Section 6 tables and figures)
        ├── sim_summary_reduced_integrated_g35/  # restricted-model summaries (Section 6.4)
        ├── sim_summary_integrated_g35/          # integrated, centered variant (verification run)
        └── sim_summary_g35/                     # legacy non-marginalized summaries (superseded)
```

## Simulation Design

2x2 factorial design, `N_SIM = 100` replications per cell (400 total):

| Factor | Levels |
|---|---|
| Sample size (N) | 50, 100 |
| Estimation length (T_est) | 25, 50 |

Model dimensions are a reduced version of the empirical study: O1 = 4, U1 = 2, O2 = 2, U2 = 1, Nt = 60 (50 dynamics + 10-point forecast window).

Population (true) values are the parameter estimates of Kelava et al. (2022), with two documented exceptions governing the regime process (both stated in Section 6.1 of the manuscript):

- The switch-back probability P12 is fixed at 1e-12 (near-absorbing regime 2, consistent with the estimator's assumption).
- The transition intercept gamma1 is set to 3.50 rather than the reported 1.48: under the reported value almost no regime-1 observations remain after the first few occasions and the classification metrics become uninformative. The value 3.50 (about 3% per-occasion switching) was chosen with `calibrate_gamma1.py` (`comparison/gamma1_calibration.csv`).

All transition coefficients (gamma1, gamma2, gamma3, gamma4) are freely estimated in the full model. The restricted model (`SIM_REDUCED=1`) holds gamma3 = gamma4 = 0 in the estimator while the data-generating process keeps the true non-zero values; this quantifies the cost of omitting an operative within-person predictor. (The restricted model does not involve the transition predictor, so it is invariant to the centering switch.)

Parameter starting values are drawn from a warm-start distribution centered at the Kelava et al. (2022) posterior means (`kelava_init_params_sim.csv`); 5 independent initializations are run per replication and the solution with the highest log-likelihood is retained.

## Environment Switches

All read by `sim_main.py` / `sim_filtering_integrated.py`; the submit scripts set them.

| Variable | Default | Meaning |
|---|---|---|
| `SIM_INTEGRATED` | 0 | 1 = integrated (marginal ML) estimator (`sim_filtering_integrated.py`); 0 = legacy non-marginalized variant |
| `CENTER_MS_PREDICTOR` | 1 | 0 = raw (uncentered) transition predictor — used for all manuscript results |
| `SIM_REDUCED` | 0 | 1 = restricted estimator (gamma3 = gamma4 = 0); appends `_reduced` to the output directory |
| `SIM_TAG` | (empty) | Output directory suffix, e.g. `g35_uncent` -> `output_integrated_g35_uncent/` |
| `SIM_COND_IDX` | (unset) | 0..3 selects a single (N, T_est) condition; on SLURM, `SLURM_ARRAY_TASK_ID` is used |

## How to Run

### On bwHelix (primary)

First-time setup (run once):
```bash
bash ~/Codes/Simulation/setup_env.sh
```

Production run (integrated, uncentered; edit `--array=0-3` to cover all four conditions):
```bash
cd ~/Codes/Simulation
mkdir -p logs output_integrated_g35_uncent
sbatch submit_sim_uncentered.sh
```

Restricted-model comparison:
```bash
sbatch submit_sim_reduced_integrated.sh
```

Summarize (login shell required):
```bash
bash -l
module load devel/python/3.13.1
source ~/venv_regime/bin/activate
cd ~/Codes/Simulation
python sim_summarize.py --output_dir output_integrated_g35_uncent
python sim_summarize.py --output_dir output_reduced_integrated_g35
```

### Locally

```bash
cd Codes/Simulation
pip install -r requirements.txt
SIM_INTEGRATED=1 CENTER_MS_PREDICTOR=0 SIM_TAG=g35_uncent python sim_main.py
python sim_summarize.py --output_dir output_integrated_g35_uncent
python sim_evaluation.py
```

## Output

Raw results (pkl; not synced to GitHub):

```
output_integrated_g35_uncent/       # production (full model)
output_reduced_integrated_g35/      # restricted model
└── sim_results_two_stage_N<N>_Ntrain<T>.pkl
```

Note: the `two_stage` prefix in output file names is a legacy naming artifact of the code history; the estimator actually used is selected by `SIM_INTEGRATED`, not by the file name.

Summary tables and figures (synced):

```
sim_summary_integrated_g35_uncent/  # production (restricted-run directory analogous)
├── sim_summary_all.csv
├── sim_summary_two_stage_N<N>_Ntrain<T>.csv
├── init_params_two_stage_N<N>_Ntrain<T>.csv
├── tables/
│   ├── param_table_ms.csv   # Markov-switching parameters
│   ├── param_table_sm.csv   # structural parameters
│   ├── param_table_mm.csv   # measurement parameters
│   └── metrics_table.csv    # sensitivity/specificity and power
└── plots/
    ├── sim_score_2x2.png
    └── sim_score_overlay.png
```

Comparison artifacts (synced):

```
comparison/
├── full_vs_reduced_metrics.csv   # sensitivity/specificity, full vs. restricted
├── gamma1_calibration.csv        # regime-2 proportions across the gamma1 grid
├── appendix_tex_uncent/          # final appendix table bodies (manuscript)
├── appendix_tex_integrated/      # centered-variant bodies (verification)
└── appendix_tex/                 # legacy bodies (superseded)
```

## Notes

- `Codes/Empirical/` is local only (not on GitHub) — contains empirical analysis scripts and data
- `psychometrika_revision/` and `IMPS2026_talk/` are local only — the manuscript is managed via Overleaf
- `output_*` (pkl) and `logs/` are excluded from git
- The centered integrated runs (`sim_summary_integrated_g35/`) and the legacy non-marginalized runs (`sim_summary_g35/`) are retained for provenance; all manuscript numbers come from `sim_summary_integrated_g35_uncent/` and `sim_summary_reduced_integrated_g35/`
- The venv on bwcluster uses CPU-only PyTorch for efficiency
