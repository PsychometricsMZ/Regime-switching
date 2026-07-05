# check_results.py
import os
import pandas as pd
import numpy as np

output_base = "output/results"

conditions = [
    ("N50_Ntrain25",  50,  25),
    ("N50_Ntrain50",  50,  50),
    ("N100_Ntrain25", 100, 25),
    ("N100_Ntrain50", 100, 50),
]

param_dfs = {}
for cond_label, N, ntrain in conditions:
    path = os.path.join(output_base, f"N_{N}_Ntrain_{ntrain}", f"param_summary_{cond_label}.csv")
    if os.path.exists(path):
        param_dfs[cond_label] = pd.read_csv(path).set_index("Parameter")

metrics_dfs = {}
for cond_label, N, ntrain in conditions:
    path = os.path.join(output_base, f"N_{N}_Ntrain_{ntrain}", f"metrics_summary_{cond_label}.csv")
    if os.path.exists(path):
        metrics_dfs[cond_label] = pd.read_csv(path).set_index("Metric")

all_params = sorted(set().union(*[df.index for df in param_dfs.values()])) if param_dfs else []
cond_labels = [c[0] for c in conditions]

print("\n" + "="*70)
print("PARAMETER RECOVERY across 4 conditions")
print("="*70)
for metric in ["Bias", "RMSE", "Coverage_Rate"]:
    print(f"\n--- {metric} ---")
    rows = []
    for param in all_params:
        row = {"Parameter": param}
        for cond_label in cond_labels:
            if cond_label in param_dfs and param in param_dfs[cond_label].index:
                row[cond_label] = round(param_dfs[cond_label].loc[param, metric], 4)
            else:
                row[cond_label] = None
        rows.append(row)
    print(pd.DataFrame(rows).set_index("Parameter").to_string())

print("\n" + "="*70)
print("FORECAST METRICS across 4 conditions")
print("="*70)
if metrics_dfs:
    all_metrics = sorted(set().union(*[df.index for df in metrics_dfs.values()]))
    for stat in ["Mean", "SD"]:
        print(f"\n--- {stat} ---")
        rows = []
        for metric in all_metrics:
            row = {"Metric": metric}
            for cond_label in cond_labels:
                if cond_label in metrics_dfs and metric in metrics_dfs[cond_label].index:
                    row[cond_label] = round(metrics_dfs[cond_label].loc[metric, stat], 4)
                else:
                    row[cond_label] = None
            rows.append(row)
        print(pd.DataFrame(rows).set_index("Metric").to_string())

print("\n" + "="*70)
print("FLAGS: Coverage < 0.70 or |Bias| > 0.5")
print("="*70)
for cond_label, N, ntrain in conditions:
    if cond_label not in param_dfs:
        continue
    df = param_dfs[cond_label]
    flags = df[(df["Coverage_Rate"] < 0.70) | (df["Bias"].abs() > 0.5)][["True_Value","Mean_Est","Bias","RMSE","Coverage_Rate"]]
    print(f"\nN={N}, NT_TRAIN={ntrain} ({len(flags)} flagged):")
    print(flags.to_string() if len(flags) > 0 else "  None")
