import os
import pandas as pd
import os
import glob
import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt
try:
    import pyreadr
except ImportError:
    pyreadr = None

try:
    config
except NameError:
    if os.path.exists("config.py"):
        from config import config
    else:
        raise FileNotFoundError("config.py not found. Please set working directory correctly.")



def load_true_parameters2(param_file_path):
    # File existence check
    if not os.path.exists(param_file_path):
        raise FileNotFoundError(f"Parameter file not found at: {param_file_path}")

    params_df = pd.read_csv(param_file_path)

    # --- Mapping Logic (U1=7 -> U1=2 conversion) ---
    param_map = pd.DataFrame({
        "Original": [
            # Indicators (R1)
            "R1d_4", "R1d_5", "R1d_6", "R1d_7",
            "log_R1d_4", "log_R1d_5", "log_R1d_6", "log_R1d_7",
            # Transition Matrix (B) - Selecting columns/rows 2 and 3
            "B11_2", "B11_3", "B12_2", "B12_3",
            "B21_2", "B21_3", "B22_2", "B22_3",
            "B31d_2", "B31d_3", "B32d_2", "B32d_3",
            "B41d_2", "B41d_3", "B42d_2", "B42d_3",
            # Variances (Q)
            "Q1d_2", "Q1d_3", "log_Q1d_2", "log_Q1d_3",
            "Q2d_2", "Q2d_3", "log_Q2d_2", "log_Q2d_3",
            # Switching Parameters (gamma)
            "gamma3_2", "gamma3_3", "gamma4_2", "gamma4_3"
        ],
        "New": [
            "R1d_1", "R1d_2", "R1d_3", "R1d_4",
            "log_R1d_1", "log_R1d_2", "log_R1d_3", "log_R1d_4",
            "B11_1", "B11_2", "B12_1", "B12_2",
            "B21_1", "B21_2", "B22_1", "B22_2",
            "B31d_1", "B31d_2", "B32d_1", "B32d_2",
            "B41d_1", "B41d_2", "B42d_1", "B42d_2",
            "Q1d_1", "Q1d_2", "log_Q1d_1", "log_Q1d_2",
            "Q2d_1", "Q2d_2", "log_Q2d_1", "log_Q2d_2",
            "gamma3_1", "gamma3_2", "gamma4_1", "gamma4_2"
        ]
    })

    other_params = params_df[
        (~params_df["Parameter"].str.contains("_", regex=False)) |
        (params_df["Parameter"].str.contains(r"^(?:Lmd2_|R2d_|log_R2d_)", regex=True))
    ].copy()

    remapped_params = params_df.merge(
        param_map,
        left_on="Parameter",
        right_on="Original",
        how="inner"
    )[["New", "Estimate"]].copy()
    remapped_params.columns = ["Parameter", "Value"]

    other_params_selected = other_params[["Parameter", "Estimate"]].copy()
    other_params_selected.columns = ["Parameter", "Value"]

    final_params_df = pd.concat(
        [other_params_selected, remapped_params],
        ignore_index=True
    )

    true_params_list = dict(zip(final_params_df["Parameter"], final_params_df["Value"]))

    return true_params_list

# sim_evaluation.py


# Function to calculate parameter summary stats
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


# --- Main Evaluation Loop ---
result_files = glob.glob(
    os.path.join(str(config["output_dir"]), "**", "sim_results_N*.rds"),
    recursive=True
)

if len(result_files) == 0:
    warnings.warn("No result files found in the output directory.")
else:
    # List to temporarily store score function results per N
    all_score_function_data = {}

    for f in result_files:
        rds_result = pyreadr.read_r(f)

        # pyreadr returns a dictionary-like object
        # For .rds, it is often under key None
        results = rds_result[None] if None in rds_result else list(rds_result.values())[0]

        if results is None or len(results) == 0:
            continue

        # If results is a DataFrame rather than a Python list-like structure,
        # keep it as-is; otherwise attempt to iterate similarly to the R code.
        if isinstance(results, pd.DataFrame):
            results_list = results.to_dict(orient="records")
        else:
            results_list = results

        # Extract Condition info (N)
        valid_indices = [
            i for i, x in enumerate(results_list)
            if x is not None and isinstance(x, dict) and x.get("condition") is not None
        ]
        if len(valid_indices) == 0:
            continue

        first_valid = results_list[valid_indices[0]]
        current_N = first_valid["condition"]["N"]

        print(f"\n>>> Evaluating Condition: N = {current_N} <<<")

        # Create output directory (e.g., output/results/N_75)
        specific_out_dir = os.path.join(str(config["output_dir"]), "results", f"N_{current_N}")
        if not os.path.exists(specific_out_dir):
            os.makedirs(specific_out_dir, exist_ok=True)
            print(f"  Created directory: {specific_out_dir}")

        # --- 1. Parameter Estimation Summary ---
        all_estimates_list = []
        for x in results_list:
            if x is None:
                continue
            if not isinstance(x, dict):
                continue
            if x.get("estimates") is None:
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

                if all(col in p2_row.columns for col in df.columns):
                    df = pd.concat([df, p2_row[df.columns]], ignore_index=True)

            all_estimates_list.append(df)

        all_estimates = pd.concat(all_estimates_list, ignore_index=True) if len(all_estimates_list) > 0 else None

        if all_estimates is not None and len(all_estimates) > 0:
            sim_true_params = results_list[valid_indices[0]]["true_params"]
            summary_df = calc_summary_stats(all_estimates, sim_true_params)

            summary_df.to_csv(
                os.path.join(specific_out_dir, f"param_summary_N{current_N}.csv"),
                index=False
            )
        else:
            print("  No valid parameter estimates found.")

        # --- 2. Performance Metrics Summary (Scalar metrics including split metrics) ---
        metrics_list = []
        for x in results_list:
            if x is not None and isinstance(x, dict) and x.get("metrics") is not None:
                metrics = x["metrics"]
                if isinstance(metrics, pd.DataFrame):
                    metrics_list.append(metrics)
                else:
                    metrics_list.append(pd.DataFrame([metrics]) if isinstance(metrics, dict) else pd.DataFrame(metrics))

        metrics_df = pd.concat(metrics_list, ignore_index=True) if len(metrics_list) > 0 else None

        if metrics_df is not None and len(metrics_df) > 0:
            print("\n  [Performance Metrics Summary]")

            perf_summary = pd.DataFrame({
                "Metric": metrics_df.columns,
                "Mean": [pd.to_numeric(metrics_df[col], errors="coerce").mean() for col in metrics_df.columns],
                "SD": [pd.to_numeric(metrics_df[col], errors="coerce").std(ddof=1) for col in metrics_df.columns],
                "Min": [pd.to_numeric(metrics_df[col], errors="coerce").min() for col in metrics_df.columns],
                "Max": [pd.to_numeric(metrics_df[col], errors="coerce").max() for col in metrics_df.columns],
            })

            perf_summary.to_csv(
                os.path.join(specific_out_dir, f"metrics_summary_N{current_N}.csv"),
                index=False
            )
        else:
            print("  No performance metrics found.")

        # --- 3. Time-Varying Score Function Summary (Forecast Period Only) ---
        score_function_hist_list = []
        for x in results_list:
            if x is not None and isinstance(x, dict) and x.get("score_function_history") is not None:
                score_function_history = x["score_function_history"]
                score_function_hist_list.append(score_function_history)

        score_function_hist_list = [x for x in score_function_hist_list if x is not None]

        if len(score_function_hist_list) > 0:
            score_function_hist_list_processed = []
            for x in score_function_hist_list:
                arr = np.asarray(x)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                score_function_hist_list_processed.append(arr)

            score_function_mat = np.vstack(score_function_hist_list_processed)

            # The length of the training period is assumed to be config["NT_TRAIN"]
            n_train_time = 25

            score_function_time_summary = pd.DataFrame({
                "Time": np.arange(n_train_time + 1, n_train_time + score_function_mat.shape[1] + 1),
                "Mean_Score_Function": np.nanmean(score_function_mat, axis=0),
                "SD_Score_Function": np.nanstd(score_function_mat, axis=0, ddof=1),
                "Dataset": f"N{current_N}"
            })

            # Save for combining all data
            all_score_function_data[f"N{current_N}"] = score_function_time_summary

            print("\n  [Time-Varying Score Function Output (Forecast Only)]")
            print(
                f"  Processing score function history for {score_function_mat.shape[1]} "
                f"forecast time points (t={score_function_time_summary['Time'].min()} "
                f"to {score_function_time_summary['Time'].max()})."
            )

            csv_path = os.path.join(specific_out_dir, f"score_function_trajectory_N{current_N}.csv")
            score_function_time_summary.to_csv(csv_path, index=False)
            print(f"  Saved score function trajectory to: {csv_path}")
        else:
            print("  No score function history data found.")

        # ----------------------------------------------------------------
        # Move original .rds file to the specific directory
        # ----------------------------------------------------------------
        old_path = f
        new_path = os.path.join(specific_out_dir, os.path.basename(f))

        if os.path.abspath(old_path) != os.path.abspath(new_path):
            shutil.move(old_path, new_path)
            print(f"  Moved result file to: {new_path}")

    # --- 4. Plot Combined Score Function Trajectory ---
    if len(all_score_function_data) > 0:
        print("\n>>> Generating Combined Score Function Plot <<<")

        combined_data = pd.concat(all_score_function_data.values(), ignore_index=True)

        # Convert Dataset to ordered categorical, setting legend order to N75 -> N100
        combined_data["Dataset"] = pd.Categorical(
            combined_data["Dataset"],
            categories=["N75", "N100"],
            ordered=True
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        for dataset_name, dataset_df in combined_data.groupby("Dataset", observed=False):
            if pd.isna(dataset_name):
                continue

            dataset_df = dataset_df.sort_values("Time")

            ax.plot(
                dataset_df["Time"],
                dataset_df["Mean_Score_Function"],
                label={"N75": "75", "N100": "100"}.get(dataset_name, str(dataset_name))
            )

            ax.errorbar(
                dataset_df["Time"],
                dataset_df["Mean_Score_Function"],
                yerr=dataset_df["SD_Score_Function"],
                fmt="none",
                alpha=0.5,
                capsize=2
            )

        ax.set_title("Score function (forecast interval) with Standard Deviation")
        ax.set_xlabel("Time")
        ax.set_ylabel("Score function")
        ax.legend(title="N", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2)

        # Theme settings similar to theme_classic()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        plot_save_path = os.path.join(str(config["output_dir"]), "results", "combined_score_function_plot.png")
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved combined score function plot to: {plot_save_path}")   
