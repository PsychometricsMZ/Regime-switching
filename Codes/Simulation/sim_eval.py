# sim_eval.py
# Updated for 2x2 factorial design: N in {50, 100} x NT_TRAIN in {25, 50}

import os
import glob
import pickle
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import config


# ---------------------------------------------------------------------------
# Helper: parameter summary statistics
# ---------------------------------------------------------------------------
def calc_summary_stats(estimates_df, true_params):
    true_df = pd.DataFrame({
        "Parameter": list(true_params.keys()),
        "TrueValue": pd.to_numeric(list(true_params.values()), errors="coerce")
    })

    merged = estimates_df.merge(true_df, on="Parameter", how="left")

    merged["Bias"] = merged["Estimate"] - merged["TrueValue"]
    merged["SquaredError"] = merged["Bias"] ** 2
    merged["CI_Lower"] = merged["Estimate"] - 1.96 * merged["SE"]
    merged["CI_Upper"] = merged["Estimate"] + 1.96 * merged["SE"]
    merged["Coverage"] = (
        (merged["TrueValue"] >= merged["CI_Lower"]) &
        (merged["TrueValue"] <= merged["CI_Upper"])
    )

    summary = (
        merged.groupby("Parameter", dropna=False)
        .agg(
            True_Value=("TrueValue", lambda x: np.nanmean(x)),
            Mean_Est=("Estimate", lambda x: np.nanmean(x)),
            Bias=("Bias", lambda x: np.nanmean(x)),
            RMSE=("SquaredError", lambda x: np.sqrt(np.nanmean(x))),
            SE_Avg=("SE", lambda x: np.nanmean(x)),
            SD_Est=("Estimate", lambda x: np.nanstd(x, ddof=1)),
            Coverage_Rate=("Coverage", lambda x: np.nanmean(x.astype(float))),
            Valid_N=("Estimate", lambda x: np.sum(~pd.isna(x)))
        )
        .reset_index()
    )

    return summary


# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------
# Match new file naming: sim_results_{method}_N{N}_Ntrain{NT_TRAIN}.pkl
result_files = glob.glob(
    os.path.join(str(config["output_dir"]), "sim_results_*_N*_Ntrain*.pkl"),
    recursive=False
)
# Also search in subdirectories in case files were already moved
result_files += glob.glob(
    os.path.join(str(config["output_dir"]), "**", "sim_results_*_N*_Ntrain*.pkl"),
    recursive=True
)
result_files = sorted(set(result_files))

if len(result_files) == 0:
    warnings.warn("No result files found in the output directory.")
else:
    all_score_function_data = {}

    for f in result_files:
        with open(f, "rb") as file:
            results = pickle.load(file)

        if len(results) == 0:
            continue

        # Extract condition info (N and NT_TRAIN)
        valid_indices = [
            i for i, x in enumerate(results)
            if x is not None and x.get("condition") is not None
        ]
        if len(valid_indices) == 0:
            continue

        first_valid = results[valid_indices[0]]
        current_method = first_valid["condition"].get("METHOD", first_valid.get("method", "augmented"))
        current_N      = first_valid["condition"]["N"]
        current_ntrain = first_valid["condition"]["NT_TRAIN"]
        cond_label     = f"{current_method}_N{current_N}_Ntrain{current_ntrain}"

        n_total   = len(results)
        n_errors  = sum(1 for r in results if r is not None and r.get("error") is not None)
        n_none    = sum(1 for r in results if r is None)
        n_success = n_total - n_errors - n_none

        print(
            f"\n>>> Evaluating Condition: method={current_method}, "
            f"N={current_N}, NT_TRAIN={current_ntrain} <<<"
        )
        print(f"    Total={n_total}, Success={n_success}, Error={n_errors}, None={n_none}")

        # Output directory: output/results/N_{N}_Ntrain_{NT_TRAIN}
        specific_out_dir = os.path.join(
            str(config["output_dir"]), "results", current_method, f"N_{current_N}_Ntrain_{current_ntrain}"
        )
        os.makedirs(specific_out_dir, exist_ok=True)

        # --- 1. Parameter Estimation Summary ---
        all_estimates_list = []
        for x in results:
            if x is None or x.get("estimates") is None or x.get("error") is not None:
                continue

            df = x["estimates"]
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)

            if x.get("P2_estimated") is not None:
                p2_row = pd.DataFrame({
                    "Parameter": ["P2"],
                    "Estimate": [float(x["P2_estimated"])],
                    "SE": [np.nan]
                })
                if all(col in df.columns for col in p2_row.columns):
                    df = pd.concat([df, p2_row[df.columns]], ignore_index=True)

            all_estimates_list.append(df)

        all_estimates = pd.concat(all_estimates_list, ignore_index=True) if all_estimates_list else None

        if all_estimates is not None and len(all_estimates) > 0:
            sim_true_params = first_valid["true_params"]
            summary_df = calc_summary_stats(all_estimates, sim_true_params)
            csv_path = os.path.join(specific_out_dir, f"param_summary_{cond_label}.csv")
            summary_df.to_csv(csv_path, index=False)
            print(f"  Saved parameter summary: {csv_path}")
        else:
            print("  No valid parameter estimates found.")

        # --- 2. Performance Metrics Summary ---
        metrics_list = []
        for x in results:
            if x is None or x.get("error") is not None or x.get("metrics") is None:
                continue
            metrics = x["metrics"]
            if isinstance(metrics, pd.DataFrame):
                metrics_list.append(metrics)
            elif isinstance(metrics, dict):
                metrics_list.append(pd.DataFrame([metrics]))
            else:
                metrics_list.append(pd.DataFrame(metrics))

        metrics_df = pd.concat(metrics_list, ignore_index=True) if metrics_list else None

        if metrics_df is not None and len(metrics_df) > 0:
            perf_summary = pd.DataFrame({
                "Metric": metrics_df.columns,
                "Mean": [pd.to_numeric(metrics_df[col], errors="coerce").mean() for col in metrics_df.columns],
                "SD":   [pd.to_numeric(metrics_df[col], errors="coerce").std(ddof=1) for col in metrics_df.columns],
                "Min":  [pd.to_numeric(metrics_df[col], errors="coerce").min() for col in metrics_df.columns],
                "Max":  [pd.to_numeric(metrics_df[col], errors="coerce").max() for col in metrics_df.columns],
            })
            csv_path = os.path.join(specific_out_dir, f"metrics_summary_{cond_label}.csv")
            perf_summary.to_csv(csv_path, index=False)
            print(f"  Saved metrics summary:   {csv_path}")
        else:
            print("  No performance metrics found.")

        # --- 3. Time-Varying Score Function Summary (Forecast Period Only) ---
        score_hist_list = [
            x["score_function_history"]
            for x in results
            if x is not None and x.get("error") is None and x.get("score_function_history") is not None
        ]

        if score_hist_list:
            processed = []
            for arr in score_hist_list:
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                processed.append(arr)

            score_mat = np.vstack(processed)

            # NT_TRAIN comes from the condition dict (config["NT_TRAIN"] is now a list)
            score_summary = pd.DataFrame({
                "Time": np.arange(current_ntrain + 1, current_ntrain + score_mat.shape[1] + 1),
                "Mean_Score_Function": np.nanmean(score_mat, axis=0),
                "SD_Score_Function":   np.nanstd(score_mat, axis=0, ddof=1),
                "Condition": cond_label,
            })

            all_score_function_data[cond_label] = score_summary

            csv_path = os.path.join(specific_out_dir, f"score_function_trajectory_{cond_label}.csv")
            score_summary.to_csv(csv_path, index=False)
            print(
                f"  Saved score function trajectory ({score_mat.shape[1]} forecast points, "
                f"t={int(score_summary['Time'].min())}..{int(score_summary['Time'].max())}): {csv_path}"
            )
        else:
            print("  No score function history data found.")

        # Move pkl to condition-specific subdirectory
        new_path = os.path.join(specific_out_dir, os.path.basename(f))
        if os.path.abspath(f) != os.path.abspath(new_path):
            shutil.move(f, new_path)
            print(f"  Moved pkl to: {new_path}")

    # --- 4. Combined Score Function Plot (2x2 panel) ---
    if all_score_function_data:
        print("\n>>> Generating Combined Score Function Plot <<<")

        # Canonical order for 2x2 layout
        ordered_keys = [
            "N50_Ntrain25", "N50_Ntrain50",
            "N100_Ntrain25", "N100_Ntrain50",
        ]
        available = [k for k in ordered_keys if k in all_score_function_data]
        # Append any extra conditions not in the canonical list
        available += [k for k in all_score_function_data if k not in ordered_keys]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
        axes_flat = axes.flatten()

        panel_titles = {
            "N50_Ntrain25":  "N=50, $N_\\mathrm{train}$=25",
            "N50_Ntrain50":  "N=50, $N_\\mathrm{train}$=50",
            "N100_Ntrain25": "N=100, $N_\\mathrm{train}$=25",
            "N100_Ntrain50": "N=100, $N_\\mathrm{train}$=50",
        }

        for idx, key in enumerate(available):
            if idx >= 4:
                break
            ax  = axes_flat[idx]
            df  = all_score_function_data[key].sort_values("Time")
            ax.plot(df["Time"], df["Mean_Score_Function"], linewidth=1.2, color="steelblue")
            ax.fill_between(
                df["Time"],
                df["Mean_Score_Function"] - df["SD_Score_Function"],
                df["Mean_Score_Function"] + df["SD_Score_Function"],
                alpha=0.25, color="steelblue"
            )
            ax.set_title(panel_titles.get(key, key))
            ax.set_xlabel("Time")
            ax.set_ylabel("Score function")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Hide unused panels
        for idx in range(len(available), 4):
            axes_flat[idx].set_visible(False)

        fig.suptitle("Score Function (Forecast Interval) by Condition", fontsize=13)
        plt.tight_layout()

        plot_path = os.path.join(str(config["output_dir"]), "results", "score_function_plot_2x2.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved 2x2 score function plot to: {plot_path}")


# ---------------------------------------------------------------------------
# Display Summaries for All 4 Conditions
# ---------------------------------------------------------------------------
output_base_dir = os.path.join(str(config["output_dir"]), "results")

conditions = [
    ("N50_Ntrain25",  50,  25),
    ("N50_Ntrain50",  50,  50),
    ("N100_Ntrain25", 100, 25),
    ("N100_Ntrain50", 100, 50),
]

for cond_label, N, ntrain in conditions:
    subdir = os.path.join(output_base_dir, f"N_{N}_Ntrain_{ntrain}")

    print(f"\n{'='*60}")
    print(f"Condition: N={N}, NT_TRAIN={ntrain}")
    print(f"{'='*60}")

    param_path = os.path.join(subdir, f"param_summary_{cond_label}.csv")
    print(f"\n--- Parameter Estimation Summary ---")
    if os.path.exists(param_path):
        print(pd.read_csv(param_path).to_markdown(index=False))
    else:
        print(f"  Not found: {param_path}")

    metrics_path = os.path.join(subdir, f"metrics_summary_{cond_label}.csv")
    print(f"\n--- Performance Metrics Summary ---")
    if os.path.exists(metrics_path):
        print(pd.read_csv(metrics_path).to_markdown(index=False))
    else:
        print(f"  Not found: {metrics_path}")

    score_path = os.path.join(subdir, f"score_function_trajectory_{cond_label}.csv")
    print(f"\n--- Score Function Trajectory ---")
    if os.path.exists(score_path):
        print(pd.read_csv(score_path).head(10).to_markdown(index=False))
    else:
        print(f"  Not found: {score_path}")

print("\n--- Score Function Plot ---")
plot_path = os.path.join(output_base_dir, "score_function_plot_2x2.png")
if os.path.exists(plot_path):
    print(f"  Saved to: {plot_path}")
else:
    print(f"  Not found: {plot_path}")
