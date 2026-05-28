# Regime-Switching Simulation Study

## Overview

This repository contains Python code for a **simulation study** of a regime-switching state-space model estimated via an extended Kim filter. The simulation evaluates parameter recovery, latent state estimation, and regime classification accuracy under a 2×2 factorial design (sample size × training time points).

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
        ├── sim_evaluation.py               # Evaluation metrics and figures
        ├── sim_utils.py                    # Utility functions
        ├── config.py                       # Simulation configuration
        ├── parameter_estimates_loaded.csv  # True parameters from empirical study
        ├── kelava_init_params_sim.csv      # Kelava warm-start initial values
        ├── requirements.txt                # Python dependencies
        ├── setup_env.sh                    # bwcluster venv setup (run once)
        ├── submit_sim.sh                   # SLURM job submission (simulation)
        └── submit_postprocess.sh           # SLURM job submission (postprocessing)
```

## Simulation Design

The simulation follows a 2×2 factorial design:

| Factor | Levels |
|---|---|
| Sample size (N) | 50, 100 |
| Training time points (N_train) | 25, 50 |

Each cell runs `N_SIM = 100` replications (`400` total). The model dimensions match a reduced version of the empirical study: O1=4, U1=2, O2=2, Nt=60 (50 dynamics + 10 forecast window).

Parameter starting values are sampled from a Kelava warm-start distribution (`kelava_init_params_sim.csv`) to improve convergence; 5 independent initializations are run per replication and the best solution (highest log-likelihood) is retained.

## How to Run

### On bwcluster (primary)

First-time setup (run once):
```bash
bash ~/Codes/Simulation/setup_env.sh
```

Run simulation:
```bash
cd ~/Codes/Simulation
mkdir -p logs output
sbatch submit_sim.sh
```

Run postprocessing (after simulation completes):
```bash
sbatch submit_postprocess.sh
```

Monitor:
```bash
squeue -u $USER
tail -f logs/sim_<JOBID>.out
```

### Locally

```bash
cd Codes/Simulation
pip install -r requirements.txt
python sim_main.py
python sim_summarize.py
python sim_evaluation.py
```

## Output

Raw results are saved to `Codes/Simulation/output/` (not synced to GitHub):

```
output/
├── sim_results_two_stage_N50_Ntrain25.pkl
├── sim_results_two_stage_N50_Ntrain50.pkl
├── sim_results_two_stage_N100_Ntrain25.pkl
└── sim_results_two_stage_N100_Ntrain50.pkl
```

Summary tables and figures are saved to `Codes/Simulation/sim_summary/`:

```
sim_summary/
├── sim_summary_all.csv
├── sim_summary_two_stage_N<N>_Ntrain<T>.csv
├── tables/
│   ├── param_table_ms.csv
│   ├── param_table_sm.csv
│   ├── param_table_mm.csv
│   └── metrics_table.csv
└── plots/
    └── sim_score_2x2.png
```

## Notes

- `Codes/Empirical/` is local only (not on GitHub) — contains empirical analysis scripts and data
- `psychometrika_revision/` is local only — manuscript files are managed via Overleaf
- `Codes/Simulation/output/` and `logs/` are excluded from git
- The venv on bwcluster uses CPU-only PyTorch for efficiency
