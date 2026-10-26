"""
Regenerate comparison/full_vs_reduced_metrics.csv from the two production summary
directories (full model vs. restricted model with gamma3 = gamma4 = 0).

The classification metrics are the condition-level means already stored in
sim_summary_all.csv (identical across the parameter rows of a condition), so this
script only reads them side by side and takes the difference in percentage points.

Run from Codes/Simulation:
    python make_full_vs_reduced.py
    python make_full_vs_reduced.py sim_summary_integrated_g35_uncent sim_summary_reduced_integrated_g35
"""
import sys
from pathlib import Path

import pandas as pd

FULL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "sim_summary_integrated_g35_uncent")
REDUCED_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "sim_summary_reduced_integrated_g35")
OUT_PATH = Path("comparison/full_vs_reduced_metrics.csv")

# (N, T_est) in manuscript order
CONDS = [(50, 25), (100, 25), (50, 50), (100, 50)]
# (column prefix in the output, column in sim_summary_all.csv)
METRICS = [("Obs_Accuracy", "Mean_Observed_Accuracy"),
           ("Fore_Accuracy", "Mean_Forecast_Accuracy"),
           ("Obs_Sensitivity", "Mean_Observed_Sensitivity"),
           ("Fore_Sensitivity", "Mean_Forecast_Sensitivity"),
           ("Obs_Specificity", "Mean_Observed_Specificity"),
           ("Fore_Specificity", "Mean_Forecast_Specificity")]


def load(summary_dir: Path) -> dict:
    p = summary_dir / "sim_summary_all.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    out = {}
    for n, t in CONDS:
        sub = df[(df["N"] == n) & (df["NT_TRAIN"] == t)]
        if sub.empty:
            raise ValueError(f"no rows for N={n}, T_est={t} in {p}")
        # condition-level metrics are repeated on every parameter row
        out[(n, t)] = {src: float(sub[src].iloc[0]) for _, src in METRICS}
    return out


def main():
    full = load(FULL_DIR)
    reduced = load(REDUCED_DIR)
    rows = []
    for n, t in CONDS:
        row = {"Condition": f"D{n},{t}"}
        for name, src in METRICS:
            f, r = full[(n, t)][src], reduced[(n, t)][src]
            row[f"{name}_full"] = round(f, 4)
            row[f"{name}_reduced"] = round(r, 4)
            row[f"{name}_diff_pp"] = round(100 * (f - r), 2)
        rows.append(row)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}  (full = {FULL_DIR}, reduced = {REDUCED_DIR})")


if __name__ == "__main__":
    main()
