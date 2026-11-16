# Regime-Switching Simulation Study

## Reference

Okuyama, K., Schaffland, T. F., Kilian, P., Brandt, H., & Kelava, A. (in press). Frequentist forecasting in regime-switching models with extended Hamilton filter. *Psychometrika*.

The code in this repository corresponds to the accepted version of the manuscript. The arXiv preprint ([arXiv:2512.18149v1](https://arxiv.org/abs/2512.18149), December 2025) is the original submission and predates the revision; the section, figure, and table numbers cited in this README refer to the accepted version.

## Overview

This repository contains Python code for the **simulation study** of a regime-switching state-space model estimated by an **integrated (marginal maximum likelihood) extended Kim filter** — the frequentist counterpart to the NDLC-SEM framework of Kelava et al. (2022). The person-specific random intercept residuals are carried in an augmented state vector and marginalized under their model-implied N(0, Q2) distribution; the variance component Q2 is estimated by maximum likelihood jointly with all other parameters (`sim_filtering_integrated.py`; Section 4.6 of the manuscript).

The simulation evaluates parameter recovery, latent state forecasting, and regime classification under a 2x2 factorial design (sample size x estimation length), plus a restricted-model comparison in which the within-person transition predictors are omitted.

The full-model results reported in the manuscript come from the **integrated, uncentered** runs (`SIM_INTEGRATED=1`, `CENTER_MS_PREDICTOR=0`, `SIM_TAG=g35_uncent`): the transition predictor is the raw regime-1 conditional filtered mean of eta1, matching the data-generating process and the Kelava et al. (2022) parameterization. The restricted-model results (Section 6.4) come from the integrated restricted run (`SIM_REDUCED=1`, `SIM_INTEGRATED=1`, `SIM_TAG=g35`), for which the centering switch has no effect.

This repository covers the simulation study only. The **empirical study** (Section 5 of the manuscript) is not included: its data cannot be made publicly available because of the terms of the data disclosure agreement and privacy concerns (see the Data Availability Statement of the paper).

## Repository Structure

```
Regime-switching/
├── README.md
├── .gitignore
└── Codes/
    └── Simulation/
        ├── sim_main.py                     # Main entry point (parallel simulation loop)
        ├── sim_filtering_integrated.py     # Integrated (marginal ML) extended Kim filter — estimator used in the manuscript
        ├── sim_filtering.py                # Shared filter building blocks (imported by sim_filtering_integrated.py); also holds the earlier non-marginalized estimator, which the manuscript does not use
        ├── sim_data_generation.py          # Synthetic data generation
        ├── sim_summarize.py                # Aggregation and summary statistics
        ├── sim_evaluation.py               # Evaluation metrics, tables, and figures
        ├── sim_utils.py                    # Utility functions
        ├── config.py                       # Simulation configuration
        ├── calibrate_gamma1.py             # Data-only grid search behind the choice gamma1 = 3.50 (which lies between the grid values 3.3 and 3.6)
        ├── make_appendix_tables.py         # Regenerates the LaTeX bodies of Tables C1-C4 (defaults: production run -> comparison/appendix_tex_uncent/)
        ├── make_full_vs_reduced.py         # Regenerates comparison/full_vs_reduced_metrics.csv from the production summaries
        ├── make_talk_score_overlay.py      # Talk-only variant of Figure 4 (not used in the manuscript; needs the raw pkl files)
        ├── run_test.py                     # Small smoke test (2 replications, N = 50, T_est = 25); set SIM_INTEGRATED=1 CENTER_MS_PREDICTOR=0 to test the manuscript estimator
        ├── parameter_estimates_kelava.csv  # Population (true) values: Kelava et al. (2022) estimates, with gamma1 = 3.50 and P12 = 1e-12 (see Simulation Design)
        ├── kelava_init_params_sim.csv      # Means and SDs of the starting-value distributions (see Simulation Design)
        ├── requirements.txt                # Python dependencies
        ├── setup_env.sh                    # bwHelix venv setup (run once)
        ├── submit_sim_integrated.sh        # SLURM array job: integrated estimator (centered variant)
        ├── submit_sim_uncentered.sh        # SLURM array job: integrated + uncentered — the production run (the file is set to one condition; see How to Run)
        ├── submit_sim_reduced_integrated.sh # SLURM array job: restricted model (gamma3 = gamma4 = 0)
        ├── submit_test.sh                  # SLURM job: smoke test
        ├── submit_postprocess.sh           # SLURM job: postprocessing with the script defaults (output/ -> sim_summary/); edit its two python lines for the production directories
        ├── comparison/                     # Full vs. restricted metrics, gamma1 calibration, appendix .tex bodies
        │   ├── appendix_tex_uncent/        #   final bodies of Tables C1-C4 (D_ms = C1, D_sm = C2, D_mm = C3, D_power = C4; the D_ prefix is a legacy name)
        │   └── appendix_tex_integrated/    #   centered-variant bodies (verification)
        ├── sim_summary_integrated_g35_uncent/   # PRODUCTION summaries (Table 2 and Figure 4 in Section 6; Tables C1-C4 in Appendix C)
        ├── sim_summary_reduced_integrated_g35/  # restricted-model summaries (Section 6.4; summary CSVs only)
        └── sim_summary_integrated_g35/          # integrated, centered variant (verification run)
```

## Simulation Design

2x2 factorial design, `N_SIM = 100` replications per cell (400 total):

| Factor | Levels |
|---|---|
| Sample size (N) | 50, 100 |
| Estimation length (T_est) | 25, 50 |

In every condition 99 of the 100 replications converged; one generated dataset gave a singular innovation covariance under all five starts and was excluded. All reported results therefore use 99 replications per condition (`ConvergenceRate` = 0.99; `N_Valid` = 99 on the parameter rows of the summary CSVs).

Model dimensions are a reduced version of the empirical study: O1 = 4, U1 = 2, O2 = 2, U2 = 1. T = 60 time points are generated per replication (config key `Nt`). With T_est = 25, t = 1-25 are used for estimation and t = 26-35 form the 10-point forecast window (later time points do not enter the results). With T_est = 50, the windows are t = 1-50 and t = 51-60. File names use `Ntrain` for T_est.

Parameters are estimated on t = 1, ..., T_est only. Over the 10-point forecast window the filter runs with these parameters held fixed and keeps updating on the incoming observations. The forecast-window results (the Forecast columns of Table 2 and the score function in Figure 4) are computed from the resulting filtered regime probabilities and filtered latent-state means.

Population (true) values are the parameter estimates of Kelava et al. (2022), with two documented exceptions governing the regime process (both stated in Section 6.1 of the manuscript):

- The switch-back probability P12 is fixed at 1e-12 (near-absorbing regime 2, consistent with the estimator's assumption).
- The transition intercept gamma1 is set to 3.50 rather than the reported 1.48: under the reported value almost no regime-1 observations remain after the first few occasions and the classification metrics become uninformative. The value 3.50 (about 3% per-occasion switching) was chosen from the data-only calibration in `calibrate_gamma1.py` (`comparison/gamma1_calibration.csv`), whose grid values 3.3 and 3.6 bracket it.

All transition coefficients (gamma1, gamma2, gamma3, gamma4) are freely estimated in the full model. The restricted model (`SIM_REDUCED=1`) holds gamma3 = gamma4 = 0 in the estimator while the data-generating process keeps the true non-zero values; this quantifies the cost of omitting an operative within-person predictor. (The restricted model does not involve the transition predictor, so it is invariant to the centering switch.)

Factor loadings are not estimated. All within-level loadings equal 1 in both the data-generating process and the estimator. The two between-level loadings are (1.0, 1.2) in the data-generating process (the Kelava et al. (2022) values) and are fixed at (1, 1) in the estimator, which uses them to compute the Bartlett score of eta2.

Parameter starting values are drawn from the distributions in `kelava_init_params_sim.csv`: normal distributions centered at the population values, with standard deviations based on the Kelava et al. (2022) posterior standard deviations (variances are started on the log scale), with three exceptions. The gamma1 start is drawn from a half-normal distribution above 3.50 (scale 0.55), so every start is at least 3.50; gamma3 and gamma4 start from N(0, 0.1^2); and Q2 starts from U(0.05, 0.30). 5 independent initializations are run per replication and the solution with the highest log-likelihood is retained.

## Environment Switches

All read by `sim_main.py` / `sim_filtering_integrated.py`; the submit scripts set them.

| Variable | Default | Meaning |
|---|---|---|
| `SIM_INTEGRATED` | 0 | 1 = integrated (marginal ML) estimator (`sim_filtering_integrated.py`), used for all manuscript results; 0 = earlier non-marginalized variant in `sim_filtering.py` (not used in the manuscript) |
| `CENTER_MS_PREDICTOR` | 1 | 0 = raw (uncentered) transition predictor — used for the full-model results in the manuscript; it has no effect on the restricted model, whose run keeps the default |
| `SIM_REDUCED` | 0 | 1 = restricted estimator (gamma3 = gamma4 = 0); appends `_reduced` to the output directory |
| `SIM_TAG` | (empty) | Output directory suffix, appended after `_reduced` and `_integrated`; e.g. `SIM_INTEGRATED=1 SIM_TAG=g35_uncent` -> `output_integrated_g35_uncent/` |
| `SIM_COND_IDX` | (unset) | Selects a single (N, T_est) condition: 0 = (50, 25), 1 = (50, 50), 2 = (100, 25), 3 = (100, 50). If unset, `SLURM_ARRAY_TASK_ID` is used; if both are unset, all four conditions run |

## How to Run

### On bwHelix (primary)

The submit scripts run `cd $HOME/Codes/Simulation`, so place the repository's `Codes/` folder in your home directory or edit that line. Before submitting, change or remove the `#SBATCH --mail-user` line in each submit script.

First-time setup (run once):
```bash
bash ~/Codes/Simulation/setup_env.sh
```

Production run (integrated, uncentered). The script is set to `#SBATCH --array=2` (one condition), so pass the array range to cover all four conditions:
```bash
cd ~/Codes/Simulation
mkdir -p logs output_integrated_g35_uncent
sbatch --array=0-3 submit_sim_uncentered.sh
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
python sim_summarize.py --output_dir output_integrated_g35_uncent --save_dir sim_summary_integrated_g35_uncent
python sim_summarize.py --output_dir output_reduced_integrated_g35 --save_dir sim_summary_reduced_integrated_g35
```

Without `--save_dir`, `sim_summarize.py` writes to `sim_summary/`, and a second call overwrites the first.

### Locally

```bash
cd Codes/Simulation
pip install -r requirements.txt
# full model (long run: 4 conditions x 100 replications x 5 starts; set SIM_COND_IDX=0..3 to run one condition at a time)
SIM_INTEGRATED=1 CENTER_MS_PREDICTOR=0 SIM_TAG=g35_uncent python sim_main.py
python sim_summarize.py --output_dir output_integrated_g35_uncent --save_dir sim_summary_integrated_g35_uncent
python sim_evaluation.py --summary_dir sim_summary_integrated_g35_uncent --output_dir output_integrated_g35_uncent --no-show
# restricted model
SIM_REDUCED=1 SIM_INTEGRATED=1 SIM_TAG=g35 python sim_main.py
python sim_summarize.py --output_dir output_reduced_integrated_g35 --save_dir sim_summary_reduced_integrated_g35
# tables
python make_appendix_tables.py      # appendix table bodies -> comparison/appendix_tex_uncent/
python make_full_vs_reduced.py      # full vs. restricted metrics -> comparison/full_vs_reduced_metrics.csv
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
sim_summary_integrated_g35_uncent/  # production (the restricted-run directory holds the CSVs only)
├── sim_summary_all.csv
├── sim_summary_two_stage_N<N>_Ntrain<T>.csv
├── init_params_two_stage_N<N>_Ntrain<T>.csv
├── tables/
│   ├── param_table_ms.csv   # Markov-switching parameters (bias, RMSE, power per condition)
│   ├── param_table_sm.csv   # structural parameters (bias, RMSE, power per condition)
│   ├── param_table_mm.csv   # measurement parameters (bias, RMSE, power per condition)
│   └── metrics_table.csv    # accuracy, sensitivity, specificity and regime-2 share, observed and forecast (Table 2)
└── plots/
    ├── sim_score_2x2.png
    ├── sim_score_overlay.png
    └── sim_score_overlay.pdf   # manuscript Figure 4 (vector version used in the manuscript)
```

The plots are drawn from the raw pkl files. From the synced CSVs alone, `sim_evaluation.py` rebuilds `tables/` only.

Comparison artifacts (synced):

```
comparison/
├── full_vs_reduced_metrics.csv   # accuracy/sensitivity/specificity, full vs. restricted (make_full_vs_reduced.py)
├── gamma1_calibration.csv        # regime-2 proportions across the gamma1 grid
├── appendix_tex_uncent/          # final bodies of Tables C1-C4 (D_ms = C1, D_sm = C2, D_mm = C3, D_power = C4; make_appendix_tables.py)
└── appendix_tex_integrated/      # centered-variant bodies (verification)
```

## Notes

- Raw results (`*.pkl` in `output_*/`) and `logs/` are excluded from git
- The centered integrated run (`sim_summary_integrated_g35/`) is retained as a verification run; all manuscript numbers come from `sim_summary_integrated_g35_uncent/` and `sim_summary_reduced_integrated_g35/`
- Summaries of the earlier non-marginalized estimator were removed from the repository on 2026-10-26 (they are not referenced in the manuscript); they remain in the git history
- The venv on bwHelix uses CPU-only PyTorch for efficiency
