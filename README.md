# Regime-Switching Simulation and Empirical Study

## 1. Project Overview

This project contains R scripts for running a **simulation study** and an **empirical study** on a regime-switching model.

### Simulation Study
The simulation study aims to:
1.  Load "true parameters" based on a preceding empirical study and remap them to a reduced dimensionality (e.g., U1=7 -> U1=2).
2.  Generate simulation data based on these true parameters across different sample size conditions (e.g., N=75, N=100).
3.  Evaluate how well our extended Kim filter (implemented in `torch`) can perform the estimation of parameters, (dynamic) latent states, and regimes across multiple simulation runs.

The code set generates aggregated results (CSVs) and visualizations concerning parameter bias, classification metrics, and score function trajectories.

### Empirical Study
The empirical study directory (`Empirical/`) contains analysis scripts and data for the empirical validation study. This folder is **stored locally only** and is not synced to GitHub.

## 2. Project Structure

```
Regime-switching/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
│
├── Codes/                             # All analysis code
│   ├── Simulation/                    # Simulation study (synced to GitHub)
│   │   ├── sim_workflow.R             # Main workflow script
│   │   ├── sim_config.R               # Configuration settings
│   │   ├── sim_import_parameters.R    # Parameter loading and remapping
│   │   ├── sim_main.R                 # Core simulation engine
│   │   ├── sim_evaluation.R           # Post-processing and evaluation
│   │   │
│   │   ├── R/                         # Helper R scripts
│   │   │   ├── 00_sim_load_libraries.R   # Package loading
│   │   │   ├── 01_sim_data_generation.R  # Data generation function
│   │   │   └── 02_sim_filtering.R        # Extended Kim Filter implementation
│   │   │
│   │   ├── data/                      # Input data
│   │   │   └── parameter_estimates_loaded.csv  # Empirical parameter estimates
│   │   │
│   │   └── output/                    # Output results (generated)
│   │       └── results/
│   │           ├── N_75/              # Results for N=75 condition
│   │           ├── N_100/             # Results for N=100 condition
│   │           └── combined_score_function_plot.png
│   │
│   └── Empirical/                     # Empirical study (LOCAL ONLY - not on GitHub)
│       ├── emp_workflow.R             # Main workflow script
│       ├── emp_config.R               # Configuration settings
│       ├── emp_evaluation.R           # Evaluation script
│       ├── emp_main.R                 # Main empirical analysis
│       │
│       ├── R/                         # Helper R scripts
│       │   ├── 00_emp_load_libraries.R   # Package loading
│       │   ├── 01_emp_preprocessing.R    # Data preprocessing
│       │   └── 02_emp_filtering.R        # Filtering implementation
│       │
│       ├── data/                      # Input data
│       │   ├── sam_1718_t_C.csv       # Sample data
│       │   ├── cov_matrix_L2.csv      # Covariance matrix
│       │   └── kelava_2022_estimates.csv # Reference estimates
│       │
│       └── output/                    # Output results
│           └── results/
│               └── evaluation/        # Evaluation results
│
└── review_response_tracking.md        # LOCAL ONLY - canonical local response-letter tracker
```

## 3. Prerequisites

### Software

* R
* RStudio

### R Packages

This project uses the `pacman` package to automatically install and load required packages (`Codes/Simulation/R/00_sim_load_libraries.R`).

Key packages include:
* `here`
* `dplyr`, `tidyr`
* `ggplot2`
* `torch`
* `progress` (for simulation progress bars)
* `truncnorm`

**⚠️ Note on `torch`:**
The `torch` package, after being installed as an R library, may require additional setup via the `install_torch()` command. If `Codes/Simulation/R/02_sim_filtering.R` fails, first check that `torch` is installed correctly.

## 4. Mandatory Pre-Execution Setup (Simulation Study)

This simulation requires a source parameter file to exist.

Ensure that the file `parameter_estimates_loaded.csv` (or the file specified in `Codes/Simulation/sim_config.R`) exists in the `Codes/Simulation/data/` directory.

```
Regime-switching/
└── Codes/
    └── Simulation/
        ├── data/
        │   └── parameter_estimates_loaded.csv <-- Source data (required)
        ├── sim_workflow.R
        ├── sim_import_parameters.R
        ...
```

The script `Codes/Simulation/sim_import_parameters.R` will load this file and remap the parameters for the simulation context.

## 5. How to Run (Simulation Study)

The entire simulation and evaluation pipeline is designed to be run from a single file.

**Simply source the `Codes/Simulation/sim_workflow.R` script.**

### Execution Flow

`Codes/Simulation/sim_workflow.R` acts as the main controller. It executes scripts in the following order:

1.  **`sim_workflow.R` -> `sim_config.R`**
    * Loads global variables (paths, `N_SIM`, `N_CONDITIONS`, `MAX_ITER`, `PATIENCE`) into a `config` list.
2.  **`sim_workflow.R` -> `sim_import_parameters.R`**
    * Defines the `load_true_parameters()` function.
    * Reads the CSV from `Codes/Simulation/data/` and performs dimensionality reduction mapping (e.g., mapping specific parameters from the empirical study to the simulation model structure).
3.  **`sim_workflow.R` -> `sim_main.R`**
    * This is the core simulation engine.
    * **Conditions Loop**: Iterates through each sample size condition defined in `config$N_CONDITIONS` (e.g., 75, 100).
    * **Simulation Loop**: Runs `N_SIM` iterations for each condition.
        * **Data Generation**: Generates synthetic data using `generate_sim_data()`.
        * **Model Fitting**: Runs the `filtering()` function (Extended Kim Filter) with retry logic for initialization (`MAX_INIT_ATTEMPTS`).
    * **Saving**: Saves a single `.rds` file per condition containing the list of all simulation runs (e.g., `sim_results_N75.rds`).
4.  **`sim_workflow.R` -> `sim_evaluation.R`**
    * Reads the generated `.rds` files.
    * Creates specific output directories (e.g., `Simulation/output/results/N_75`).
    * Aggregates parameter estimates, calculates Bias/RMSE, and summarizes classification metrics.
    * Computes and plots the time-varying score function (forecast period).
    * Moves the raw `.rds` file into its specific result folder.

## 6. Output Folder Structure

Upon successful completion of the simulation, the output directory will be organized by sample size condition:

```
Regime-switching/
└── Codes/
    └── Simulation/
        └── output/
            └── results/
                ├── combined_score_function_plot.png <-- Visualization comparing conditions
                ├── N_75/                             <-- Results for N=75 condition
                │   ├── sim_results_N75.rds          <-- Raw list of all simulation objects
                │   ├── param_summary_N75.csv        <-- Parameter Bias, RMSE, Coverage
                │   ├── metrics_summary_N75.csv      <-- Accuracy, Sensitivity, Specificity
                │   └── score_function_trajectory_N75.csv
                │
                └── N_100/                            <-- Results for N=100 condition
                    ├── sim_results_N100.rds
                    ├── param_summary_N100.csv
                    ├── metrics_summary_N100.csv
                    └── score_function_trajectory_N100.csv
```

## 7. Script Details (Simulation Study)

* **`Codes/Simulation/sim_workflow.R`**: **(Main)** The primary script that orchestrates the entire workflow.
* **`Codes/Simulation/sim_config.R`**: Defines global configuration including paths, simulation iterations, and model dimensions.
* **`Codes/Simulation/sim_import_parameters.R`**: Logic to load and remap the empirical parameter estimates into the "True Parameters" used for data generation.
* **`Codes/Simulation/sim_main.R`**: **(Core)** Runs the simulation loops. Handles data generation, optimization (via `torch` Rprop), and computes base metrics.
* **`Codes/Simulation/sim_evaluation.R`**: Post-processing script. Generates summary statistics tables and plots for parameter recovery and model performance.
* **`Codes/Simulation/R/00_sim_load_libraries.R`**: Loads required R packages.
* **`Codes/Simulation/R/01_sim_data_generation.R`**: Contains the `generate_sim_data()` function.
* **`Codes/Simulation/R/02_sim_filtering.R`**: Contains the `filtering()` function which implements the Extended Kim Filter and the objective function (Log-Likelihood).

## 8. Empirical Study Scripts

The `Codes/Empirical/` folder contains scripts for the empirical analysis. This folder is **NOT synced to GitHub** for privacy and project confidentiality.

* **`Codes/Empirical/emp_workflow.R`**: Main workflow for empirical analysis.
* **`Codes/Empirical/emp_config.R`**: Configuration settings for empirical analysis.
* **`Codes/Empirical/emp_main.R`**: Core empirical analysis engine.
* **`Codes/Empirical/emp_evaluation.R`**: Evaluation and post-processing of empirical results.
* **`Codes/Empirical/R/00_emp_load_libraries.R`**: Loads required packages for empirical analysis.
* **`Codes/Empirical/R/01_emp_preprocessing.R`**: Data preprocessing functions.
* **`Codes/Empirical/R/02_emp_filtering.R`**: Extended Kim Filter implementation for empirical data.

## 9. Notes on Local-Only Files

The following files/folders are stored locally and **NOT synced to GitHub**:

- **`Codes/Empirical/`**: Complete empirical study folder with scripts and results
- **`review_response_tracking.md`**: Canonical local working file for Psychometrika review responses

These are listed in `.gitignore` to prevent accidental uploads to the remote repository.

## 10. Psychometrika Revision Workflow

For manuscript `PSY-2025-0213`, the current revision workflow is:

- **TeX manuscript source**: maintained in Overleaf as the canonical copy
- **Local response letter / review tracker**: maintained in `review_response_tracking.md`
