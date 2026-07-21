"""
Choose gamma1 for the simulation DGP.

With the reported value (gamma1 = 1.48) and a near-absorbing regime 2, essentially
every person has switched by t = 25, so regime 2 accounts for ~79-89% of
person-time points in the estimation window and ~97-99% in the forecast window.
Under that composition the classification metrics are uninformative: a constant
"always regime 2" predictor beats the model on accuracy.

This script generates data only (no estimation) for a grid of gamma1 values and
reports the resulting regime-2 proportion, so the value can be chosen on evidence
rather than on the closed-form approximation.

Run from Codes/Simulation:
    python calibrate_gamma1.py
"""
import numpy as np
import pandas as pd

from sim_data_generation import generate_sim_data
from sim_utils import load_true_parameters2

TRUTH_CSV = "parameter_estimates_kelava.csv"
GAMMA1_GRID = [1.48, 2.5, 3.0, 3.3, 3.6, 4.0, 4.5]
N_REPS = 20               # enough to pin the proportion down; generation is cheap
CONDITIONS = [(50, 25), (100, 25), (50, 50), (100, 50)]
NT = 60                   # full generated length, as in the main design
FORECAST_LEN = 10


def load_truth():
    """Same remapping (empirical U1=7 -> simulation U1=2) that sim_main.py uses."""
    return load_true_parameters2(TRUTH_CSV)


def main():
    truth = load_truth()
    if "gamma1" not in truth:
        raise KeyError(f"'gamma1' not in truth file. Keys: {list(truth)[:10]} ...")
    print(f"reported gamma1 = {truth['gamma1']}\n")

    rows = []
    for g1 in GAMMA1_GRID:
        tp = dict(truth)
        tp["gamma1"] = g1
        for N, ntrain in CONDITIONS:
            obs, fore = [], []
            for i in range(N_REPS):
                d = generate_sim_data(N=N, Nt=NT, U1=2, O1=4, O2=2,
                                      true_params=tp, seed=123 + i)
                S = np.asarray(d["S_true"]) - 1          # 0 = regime 1, 1 = regime 2
                obs.append(S[:, :ntrain].mean())
                fore.append(S[:, ntrain:ntrain + FORECAST_LEN].mean())
            rows.append({"gamma1": g1, "N": N, "Ntrain": ntrain,
                         "regime2_obs": np.mean(obs),
                         "regime2_fore": np.mean(fore)})
            print(f"  gamma1={g1:4.2f}  N={N:3d} Ntrain={ntrain:2d}   "
                  f"obs {np.mean(obs):.3f}   fore {np.mean(fore):.3f}")
        print()

    out = pd.DataFrame(rows)
    out.to_csv("comparison/gamma1_calibration.csv", index=False)
    print("Saved: comparison/gamma1_calibration.csv\n")

    print("=== summary: worst-case imbalance across the four conditions ===")
    for g1, grp in out.groupby("gamma1"):
        worst = max(grp.regime2_fore.max(), 1 - grp.regime2_obs.min())
        print(f"  gamma1={g1:4.2f}  obs range {grp.regime2_obs.min():.3f}-{grp.regime2_obs.max():.3f}   "
              f"fore range {grp.regime2_fore.min():.3f}-{grp.regime2_fore.max():.3f}   "
              f"worst class share {worst:.3f}")


if __name__ == "__main__":
    main()
