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
        ├── sim_main.py              # Main entry point (parallel simulation loop)
        ├── sim_filtering.py         # Extended Kim filter & parameter estimation
        ├── sim_data_generation.py   # Synthetic data generation
        ├── sim_summarize.py         # Aggregation and summary statistics
        ├── sim_eval.py              # Evaluation metrics
        ├── config.py                # Simulation configuration
        ├── supp_functs.py           # Utility functions
        ├── parameter_estimates_loaded.csv  # True parameters from empirical study
        ├── requirements.txt         # Python dependencies
        ├── setup_env.sh             # bwcluster venv setup (run once)
        └── submit_sim.sh            # SLURM job submission script
```

## Simulation Design

The simulation follows a 2×2 factorial design:

| Factor | Levels |
|---|---|
| Sample size (N) | 50, 100 |
| Training time points (N_train) | 25, 50 |

Each cell runs `N_SIM = 100` replications. The model dimensions match a reduced version of the empirical study: O1=4, U1=2, O2=2, Nt=60 (50 dynamics + 10 forecast window).

## How to Run

### On bwcluster (primary)

First-time setup (run once):
```bash
bash ~/Codes/Simulation/setup_env.sh
```

Job submission:
```bash
cd ~/Codes/Simulation
mkdir -p logs output
sbatch submit_sim.sh
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
```

## Output

Results are saved to `Codes/Simulation/output/` (not synced to GitHub):

```
output/
├── results_N50_train25.csv
├── results_N50_train50.csv
├── results_N100_train25.csv
└── results_N100_train50.csv
```

Each CSV contains per-simulation parameter estimates, bias, RMSE, and classification metrics (accuracy, sensitivity, specificity, F1).

## Notes

- `Codes/Empirical/` is local only (not on GitHub) — contains empirical analysis scripts and data
- `Codes/Simulation/output/` and `logs/` are excluded from git
- The venv on bwcluster uses CPU-only PyTorch for efficiency
