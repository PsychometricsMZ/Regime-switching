"""
run_test.py
-----------
End-to-end pipeline test (small scale) -- no external config file needed.

Runs:
  1. sim_main.py       -- 2 simulations, N=50, NT_TRAIN=25 only  (~2-5 min)
  2. sim_summarize.py  -- summarise results into sim_summary_test/
  3. sim_evaluation.py -- produce tables + score function plot

All output goes to output_test/ and sim_summary_test/ so it never
touches the real output/ directory.

Usage
-----
    python run_test.py
"""

import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Inline test configuration (replaces the old config_test.py)
# ---------------------------------------------------------------------------
_cfg = types.ModuleType("config")
_cfg.config = {}

_cfg.config["true_params_file"] = Path("parameter_estimates_kelava.csv")
_cfg.config["sim_init_path"]    = Path("kelava_init_params_sim.csv")
_cfg.config["output_dir"]       = Path("output_test")

_cfg.config["N_CONDITIONS"]          = [50]
_cfg.config["N_SIM"]                 = 2
_cfg.config["MAX_INIT_ATTEMPTS"]     = 2
_cfg.config["FILTER_METHODS"]        = ["two_stage"]
_cfg.config["TWO_STAGE_OUTER_LOOPS"] = 1
_cfg.config["TWO_STAGE_DAMPING"]     = 0.5

_cfg.config["Nt"]        = 60
_cfg.config["O1"]        = 4
_cfg.config["O2"]        = 2
_cfg.config["U1"]        = 2
_cfg.config["N_IMPUTE"]  = 1

_cfg.config["MAX_ITER"]       = float("inf")
_cfg.config["NT_TRAIN"]       = [25]
_cfg.config["PATIENCE"]       = 20
_cfg.config["COMPUTE_SE"]     = False
_cfg.config["SE_SAMPLE_SIZE"] = None
_cfg.config["seed"]           = 123

_cfg.config["output_dir"].mkdir(parents=True, exist_ok=True)

# Must inject before any other local import so worker processes also get it
sys.modules["config"] = _cfg

# ---------------------------------------------------------------------------
# Windows multiprocessing requires this guard at the TOP LEVEL
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    test_output_dir  = _cfg.config["output_dir"]
    test_params_file = _cfg.config["true_params_file"]

    # Step 1: sim_main
    print("\n" + "=" * 60)
    print("STEP 1: Running sim_main (2 sims x 1 condition)")
    print("=" * 60)
    import sim_main
    sim_main.main()

    # Step 2: sim_summarize
    print("\n" + "=" * 60)
    print("STEP 2: Running sim_summarize")
    print("=" * 60)
    _orig_argv = sys.argv[:]
    sys.argv = [
        "sim_summarize.py",
        "--output_dir",       str(test_output_dir),
        "--save_dir",         "sim_summary_test",
        "--true_params_file", str(test_params_file),
    ]
    import sim_summarize
    sim_summarize.main()
    sys.argv = _orig_argv

    # Step 3: sim_evaluation
    print("\n" + "=" * 60)
    print("STEP 3: Running sim_evaluation")
    print("=" * 60)
    _orig_argv = sys.argv[:]
    sys.argv = [
        "sim_evaluation.py",
        "--summary_dir", "sim_summary_test",
        "--output_dir",  str(test_output_dir),
        "--no-show",
    ]
    import sim_evaluation
    sim_evaluation.main()
    sys.argv = _orig_argv

    # Done
    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETE - pipeline is healthy.")
    print(f"  pkl output : {test_output_dir}/")
    print( "  summaries  : sim_summary_test/")
    print( "  tables     : sim_summary_test/tables/")
    print( "  plots      : sim_summary_test/plots/")
    print("=" * 60)
