# config_test.py
# Local test config: small scale to verify code runs correctly

from pathlib import Path

config = {}

# --- File Paths ---
config["true_params_file"] = Path("parameter_estimates_loaded.csv")
config["output_dir"] = Path("output_test")

# --- Simulation Settings (reduced for local test) ---
config["N_CONDITIONS"] = [50]            # 1 condition only
config["N_SIM"] = 2                      # 2 replications only
config["MAX_INIT_ATTEMPTS"] = 2
config["FILTER_METHODS"] = ["two_stage"]
config["TWO_STAGE_OUTER_LOOPS"] = 1      # 1 loop only
config["TWO_STAGE_DAMPING"] = 0.5

# --- Data Generation Dimensions ---
config["Nt"] = 60
config["O1"] = 4
config["O2"] = 2
config["U1"] = 2
config["N_IMPUTE"] = 1

# --- Filtering/Estimation Settings ---
config["MAX_ITER"] = float("inf")
config["NT_TRAIN"] = [25]                # 1 condition only
config["PATIENCE"] = 20
config["COMPUTE_SE"] = False             # Skip SE computation
config["SE_SAMPLE_SIZE"] = None

config["seed"] = 123

# Create directory
config["output_dir"].mkdir(parents=True, exist_ok=True)

print("Test configuration loaded.")
print(" - Params file: " + str(config["true_params_file"]))
print(" - Output dir:  " + str(config["output_dir"]))
