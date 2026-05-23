# sim_config.py

from pathlib import Path

_HERE = Path(__file__).resolve().parent

config = {}

# --- File Paths ---
config["true_params_file"] = _HERE / "parameter_estimates_loaded.csv"
config["output_dir"] = _HERE / "output"

# --- Simulation Settings ---
# Psychometrika revision: 2x2 factorial design
# N in {50, 100} x N_train in {25, 50} -> 4 conditions (D_{50,25}, D_{100,25}, D_{50,50}, D_{100,50})
config["N_CONDITIONS"] = [50, 100]   # Sample size conditions (revised from [75, 100])
config["N_SIM"] = 100                # Number of simulations for each condition
config["MAX_INIT_ATTEMPTS"] = 5      # Maximum attempts to try different initial values (Best of N)
config["FILTER_METHODS"] = ["two_stage"]
config["TWO_STAGE_OUTER_LOOPS"] = 50  # Hard cap; actual stopping by 3-consecutive-no-improvement criterion
config["TWO_STAGE_DAMPING"] = 0.5

# --- Data Generation Dimensions ---
# Nt=60: 50 dynamics time points + 10 forecast window (revised from 50)
config["Nt"] = 60                    # Time points (Generation)
config["O1"] = 4                     # Observed variables for eta1 (Reduced)
config["O2"] = 2                     # Observed variables for eta2
config["U1"] = 2                     # Latent factors for eta1 (Reduced)
config["N_IMPUTE"] = 1               # (Placeholder)

# --- Filtering/Estimation Settings ---
config["MAX_ITER"] = float("inf")    # Optimization max iterations
config["NT_TRAIN"] = [25, 50]        # Training time points (revised from 25; now a list for factorial loop)
config["PATIENCE"] = 20              # Early stopping patience
config["COMPUTE_SE"] = True          # True: Compute SE, False: Do not compute
config["SE_SAMPLE_SIZE"] = None      # Number of individuals to use for SE calculation (None uses all)

config["seed"] = 123

# Create directory (just in case)
config["output_dir"].mkdir(parents=True, exist_ok=True)

print("Configuration loaded completely.")
print(" - Params file: " + str(config["true_params_file"]))
print(" - Output dir:  " + str(config["output_dir"]))
