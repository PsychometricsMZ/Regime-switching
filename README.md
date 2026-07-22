# Regime-Switching Simulation Study

## Overview

This repository contains Python code for the **simulation study** of a regime-switching state-space model estimated via an extended Kim filter (frequentist counterpart to the NDLC-SEM framework). The simulation evaluates parameter recovery, latent state forecasting, and regime classification under a 2×2 factorial design (sample size × estimation length), plus a restricted-model comparison in which the within-person transition predictors are omitted.

The accompanying **empirical study** code is stored locally only and is not synced to GitHub.

## Repository Structure

```
Regime-switching/
├── README.md
├── .gitignore
└── Codes/
    └── Simulation/
        ├── sim_main.py                     # Main entry point (parallel simulation loop)
        ├── sim_filtering.py                # Extended Kim filter & parameter estimation
        ├── sim_data_generation.py          # Synthetic data generation
        ├── sim_summarize.py                # Aggregation and summary statistics
        ├── sim_evaluation.py               # Evaluation metrics, tables, and figures
        ├── sim_utils.py                    # Utility functions
        ├── config.py                       # Simulation configuration
        ├── calibrate_gamma1.py             # Data-only grid search used to choose gamma1 = 3.5
        ├── make_appendix_tables.py         # Regenerates LaTeX bodies for Appendix D (Tables D1-D4)
        ├── make_talk_score_overlay.py      # Score-function overlay figure for talks
        ├── run_test.py                     # Small smoke test (local or cluster)
        ├── parameter_estimates_kelava.csv  # Population (true) values from Kelava et al. (2022)
        ├── kelava_init_params_sim.csv      # Warm-start distribution for parameter initialization
        ├── requirements.txt                # Python dependencies
        ├── setup_env.sh                    # bwcluster venv setup (run once)
        ├── submit_sim_g35.sh               # SLURM array job: full model (gamma1 = 3.5)
        ├── submit_sim_reduced_g35.sh       # SLURM array job: restricted model (gamma3 = gamma4 = 0)
        ├── submit_test.sh                  # SLURM job: smoke test
        ├── submit_postprocess.sh           # SLURM job: postprocessing
        ├── comparison/                     # Full vs. restricted metrics, gamma1 calibration, Appendix D .tex
        ├── output_g35/                     # Raw results, full model (pkl; not synced)
        ├── output_reduced_g35/             # Raw results, restricted model (pkl; not synced)
        ├── sim_summary_g35/                # Summary tables and figures, full model
        └── sim_summary_reduced_g35/        # Summary tables and figures, restricted model
```

## Simulation Design

The simulation follows a 2×2 factorial design, each cell with `N_SIM = 100` replications (400 total):

| Factor | Levels |
|---|---|
| Sample size (N) | 50, 100 |
| Estimation length (T_est) | 25, 50 |

Model dimensions are a reduced version of the empirical study: O1 = 4, U1 = 2, O2 = 2, U2 = 1, Nt = 60 (50 dynamics + 10-point forecast window).

Population (true) values are the parameter estimates of Kelava et al. (2022), with two documented exceptions governing the regime process (both stated in Section 6.1 of the manuscript):

- The switch-back probability P12 is fixed at 1e-12 (near-absorbing regime 2, consistent with the estimator's assumption).
- The transition intercept gamma1 is set to 3.5 rather than the reported 1.48. Under the reported value essentially every person has switched within the first few occasions and the classification metrics are dominated by class composition. The value 3.5 (about 3% per-occasion switching) was chosen with `calibrate_gamma1.py`, which generates data over a grid of gamma1 values and reports the resulting regime-2 proportions (`comparison/gamma1_calibration.csv`).

All transition coefficients (gamma1, gamma2, gamma3, gamma4) are freely estimated in the full model. The restricted model (`SIM_REDUCED=1`) holds gamma3 = gamma4 = 0 in the estimator while the data-generating process keeps the true non-zero values; this quantifies the cost of omitting an operative within-person predictor.

Parameter starting values are drawn from a warm-start distribution centered at the Kelava et al. (2022) posterior means (`kelava_init_params_sim.csv`); 5 independent initializations are run per replication and the solution with the highest log-likelihood is retained.

Output directories are controlled by environment variables in `sim_main.py`: `SIM_TAG=g35` appends `_g35` to the output directory; `SIM_REDUCED=1` switches to the restricted estimator and appends `_reduced`. The current results were produced with the g35 tag; the submit scripts set these variables.

## How to Run

### On bwHelix (primary)

First-time setup (run once):
```bash
bash ~/Codes/Simulation/setup_env.sh
```

Run the full-model simulation (4 array tasks, one per condition):
```bash
cd ~/Codes/Simulation
mkdir -p logs output_g35
sbatch submit_sim_g35.sh
```

Run the restricted-model comparison:
```bash
sbatch submit_sim_reduced_g35.sh
```

Summarize (login bash shell required):
```bash
bash -l
module load devel/python/3.13.1
source ~/venv_regime/bin/activate
cd ~/Codes/Simulation
python sim_summarize.py --output_dir output_g35
python sim_summarize.py --output_dir output_reduced_g35
```

Monitor:
```bash
squeue -u $USER
tail -f logs/simg35_<JOBID>_<TASK>.out
```

### Locally

```bash
cd Codes/Simulation
pip install -r requirements.txt
SIM_TAG=g35 python sim_main.py
python sim_summarize.py --output_dir output_g35
python sim_evaluation.py
```

## Output

Raw results (not synced to GitHub):

```
output_g35/                 # full model
output_reduced_g35/         # restricted model (gamma3 = gamma4 = 0)
└── sim_results_two_stage_N<N>_Ntrain<T>.pkl   # per-replication estimates, metrics,
                                               # and score_function_history (delta_t)
```

Summary tables and figures (synced):

```
sim_summary_g35/            # full model (sim_summary_reduced_g35/ analogous)
├── sim_summary_all.csv
├── sim_summary_two_stage_N<N>_Ntrain<T>.csv
├── init_params_two_stage_N<N>_Ntrain<T>.csv
├── tables/
│   ├── param_table_ms.csv   # Markov-switching parameters (Table D1)
│   ├── param_table_sm.csv   # structural parameters (Table D2)
│   ├── param_table_mm.csv   # measurement parameters (Table D3)
│   └── metrics_table.csv    # sensitivity/specificity (Table 2) and power (Table D4)
└── plots/
    ├── sim_score_2x2.png
    └── sim_score_overlay.png
```

Comparison artifacts (synced):

```
comparison/
├── full_vs_reduced_metrics.csv   # sensitivity/specificity, full vs. restricted
├── gamma1_calibration.csv        # regime-2 proportions across gamma1 grid
└── appendix_tex/                 # D_ms.tex, D_sm.tex, D_mm.tex, D_power.tex
                                  # (generated by make_appendix_tables.py)
```

## Notes

- `Codes/Empirical/` is local only (not on GitHub) — contains empirical analysis scripts and data
- `psychometrika_revision/` and `IMPS2026_talk/` are local only — manuscript is managed via Overleaf
- `output_g35/`, `output_reduced_g35/` (pkl), and `logs/` are excluded from git
- Superseded pre-g35 runs (gamma1 = 1.48: `output/`, `output_reduced/`, `sim_summary/`, `sim_summary_reduced/`) were retired to `Codes/trash/simulation_pre_g35/` (local only)
- The venv on bwcluster uses CPU-only PyTorch for efficiency
