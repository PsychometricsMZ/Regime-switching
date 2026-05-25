"""
sim_summarize.py
----------------
Summarize simulation results from pkl files produced by sim_main.py.

Outputs
-------
1. sim_summary_<method>_N<N>_Ntrain<NT>.csv
   Per-parameter recovery table: mean, SD, SE, bias, RMSE, relative bias (%)
   + classification metrics (accuracy, sensitivity, etc.) averaged across runs.

2. init_params_<method>_N<N>_Ntrain<NT>.csv
   Initialization CSV in the same format as parameter_estimates_loaded.csv.
   Use this file to seed the empirical estimation:
       Estimate = mean across simulation runs
       SE       = SD / sqrt(n_converged)
       CI_Lower = Estimate - 1.96 * SE
       CI_Upper = Estimate + 1.96 * SE

Usage
-----
Run from the Python/ directory (same place as sim_main.py):

    python sim_summarize.py                          # all pkl files in output/
    python sim_summarize.py --output_dir my_output/  # custom output dir
"""

import os
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sim_utils import load_true_parameters2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Summarize simulation pkl results.")
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Directory containing sim_results_*.pkl files (default: output/)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="sim_summary",
        help="Directory to write summary CSVs (default: sim_summary/)"
    )
    parser.add_argument(
        "--true_params_file", type=str, default="parameter_estimates_loaded.csv",
        help="CSV with true parameter values used as DGP (default: parameter_estimates_loaded.csv)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load true params
# ---------------------------------------------------------------------------
def load_true_params(path):
    """Return {parameter_name: true_value} from the DGP CSV."""
    if not os.path.exists(path):
        warnings.warn(f"True params file not found: {path}. Bias/RMSE will not be computed.")
        return {}
    df = pd.read_csv(path)
    # Support both 'Estimate' (old) and 'Kelava_Value' column names
    val_col = "Estimate" if "Estimate" in df.columns else df.columns[1]
    return dict(zip(df["Parameter"], df[val_col]))


# ---------------------------------------------------------------------------
# Parse one pkl file
# ---------------------------------------------------------------------------
def parse_pkl(pkl_path):
    """
    Returns a dict with keys:
        condition   : {"METHOD": ..., "N": ..., "NT_TRAIN": ...}
        estimates   : list of DataFrames (one per successful run)
        metrics     : list of DataFrames (one per successful run)
        ll_list     : list of floats
        n_total     : int
        n_converged : int
    """
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    estimates = []
    metrics   = []
    ll_list   = []
    condition = None
    n_total   = 0
    n_converged = 0

    for res in results:
        if res is None:
            continue
        n_total += 1

        if condition is None and "condition" in res:
            condition = res["condition"]

        if res.get("error") is not None:
            continue  # failed run

        est_df = res.get("estimates")
        if est_df is None or not isinstance(est_df, pd.DataFrame):
            continue

        n_converged += 1
        est_series = est_df.set_index("Parameter")["Estimate"]

        # P2 is estimated via a moment estimator outside the main optimisation
        # and stored separately — append it manually so it appears in the
        # parameter recovery table (cf. Manuscript Table A5).
        p2_val = res.get("P2_estimated")
        if p2_val is not None and np.isfinite(float(p2_val)):
            est_series = pd.concat([est_series, pd.Series({"P2": float(p2_val)})])

        estimates.append(est_series)
        ll_list.append(res.get("ll", np.nan))

        met = res.get("metrics")
        if met is not None and isinstance(met, pd.DataFrame) and len(met) > 0:
            metrics.append(met.iloc[0])  # one row per run

    return {
        "condition":    condition,
        "estimates":    estimates,
        "metrics":      metrics,
        "ll_list":      ll_list,
        "n_total":      n_total,
        "n_converged":  n_converged,
    }


# ---------------------------------------------------------------------------
# Summarize parameter recovery
# ---------------------------------------------------------------------------
def summarize_estimates(estimates_list, true_params, n_converged):
    """
    estimates_list : list of pd.Series indexed by parameter name
    Returns a DataFrame with columns:
        Parameter, True_Value, Mean, SD, SE, Bias, Relative_Bias_pct, RMSE
    """
    if not estimates_list:
        return pd.DataFrame()

    # Stack into (n_runs × n_params) DataFrame; align columns
    df = pd.DataFrame(estimates_list).reset_index(drop=True)

    rows = []
    for param in df.columns:
        vals = df[param].dropna().values
        if len(vals) == 0:
            continue
        mean_val = float(np.mean(vals))
        sd_val   = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        se_val   = sd_val / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
        true_val = true_params.get(param, np.nan)
        bias     = mean_val - true_val if not np.isnan(true_val) else np.nan
        rel_bias = 100 * bias / abs(true_val) if (not np.isnan(true_val) and true_val != 0) else np.nan
        rmse     = float(np.sqrt(np.mean((vals - true_val) ** 2))) if not np.isnan(true_val) else np.nan
        rows.append({
            "Parameter":        param,
            "True_Value":       true_val,
            "Mean":             round(mean_val, 6),
            "SD":               round(sd_val, 6),
            "SE":               round(se_val, 6),
            "Bias":             round(bias, 6)     if not np.isnan(bias)     else np.nan,
            "Relative_Bias_pct": round(rel_bias, 2) if not np.isnan(rel_bias) else np.nan,
            "RMSE":             round(rmse, 6)     if not np.isnan(rmse)     else np.nan,
            "N_Valid":          len(vals),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summarize classification metrics
# ---------------------------------------------------------------------------
def summarize_metrics(metrics_list):
    """Returns a single-row DataFrame with mean of each metric column."""
    if not metrics_list:
        return pd.DataFrame()
    df = pd.DataFrame(metrics_list).reset_index(drop=True)
    # Numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    summary = numeric_df.mean().to_frame().T
    summary.columns = [f"Mean_{c}" for c in summary.columns]
    return summary.round(4)


# ---------------------------------------------------------------------------
# Build initialization CSV
# ---------------------------------------------------------------------------
def build_init_csv(summary_df):
    """
    Convert parameter recovery summary into an initialization CSV.
    Format mirrors parameter_estimates_loaded.csv:
        Parameter, Estimate (=Mean), SE, CI_Lower, CI_Upper, ...
    """
    if summary_df.empty:
        return pd.DataFrame()

    init = summary_df[["Parameter", "Mean", "SD", "SE"]].copy()
    init = init.rename(columns={"Mean": "Estimate"})
    init["CI_Lower"] = (init["Estimate"] - 1.96 * init["SE"]).round(6)
    init["CI_Upper"] = (init["Estimate"] + 1.96 * init["SE"]).round(6)
    init["P_Value"]  = np.nan
    init["Z_Score"]  = np.nan
    init["Type"]     = "SimulationMean"
    init["Pretty_Parameter"] = init["Parameter"]
    init["Parameter_Label"]  = init["Parameter"]
    return init


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    true_params = load_true_parameters2(args.true_params_file)

    pkl_files = sorted(output_dir.glob("sim_results_*.pkl"))
    if not pkl_files:
        print(f"No pkl files found in '{output_dir}'. Exiting.")
        return

    print(f"Found {len(pkl_files)} pkl file(s) in '{output_dir}'.\n")

    all_summary_rows = []

    for pkl_path in pkl_files:
        print(f"Processing: {pkl_path.name}")
        data = parse_pkl(pkl_path)

        cond = data["condition"] or {}
        method   = cond.get("METHOD", "unknown")
        N        = cond.get("N", "?")
        NT_TRAIN = cond.get("NT_TRAIN", "?")

        n_total     = data["n_total"]
        n_converged = data["n_converged"]
        conv_rate   = n_converged / n_total if n_total > 0 else np.nan

        print(f"  Condition: method={method}, N={N}, NT_TRAIN={NT_TRAIN}")
        print(f"  Converged: {n_converged}/{n_total} ({100*conv_rate:.1f}%)")

        # --- Parameter recovery ---
        est_summary = summarize_estimates(data["estimates"], true_params, n_converged)

        # --- Metrics ---
        met_summary = summarize_metrics(data["metrics"])

        # --- LL ---
        ll_arr = np.array([x for x in data["ll_list"] if not np.isnan(x)])
        ll_mean = float(np.mean(ll_arr)) if len(ll_arr) > 0 else np.nan
        ll_sd   = float(np.std(ll_arr, ddof=1)) if len(ll_arr) > 1 else np.nan

        tag = f"{method}_N{N}_Ntrain{NT_TRAIN}"

        # --- Save full summary ---
        if not est_summary.empty:
            # Add condition info
            est_summary.insert(0, "NT_TRAIN", NT_TRAIN)
            est_summary.insert(0, "N", N)
            est_summary.insert(0, "Method", method)

            # Append LL stats as special rows
            ll_rows = pd.DataFrame([{
                "Method": method, "N": N, "NT_TRAIN": NT_TRAIN,
                "Parameter": "LogLikelihood",
                "True_Value": np.nan, "Mean": round(ll_mean, 4),
                "SD": round(ll_sd, 4), "SE": np.nan,
                "Bias": np.nan, "Relative_Bias_pct": np.nan, "RMSE": np.nan,
                "N_Valid": len(ll_arr),
            }])
            conv_rows = pd.DataFrame([{
                "Method": method, "N": N, "NT_TRAIN": NT_TRAIN,
                "Parameter": "ConvergenceRate",
                "True_Value": np.nan, "Mean": round(conv_rate, 4),
                "SD": np.nan, "SE": np.nan,
                "Bias": np.nan, "Relative_Bias_pct": np.nan, "RMSE": np.nan,
                "N_Valid": n_total,
            }])
            full_summary = pd.concat([est_summary, ll_rows, conv_rows], ignore_index=True)

            # Merge metrics
            if not met_summary.empty:
                for col in met_summary.columns:
                    full_summary[col] = met_summary[col].values[0]

            out_path = save_dir / f"sim_summary_{tag}.csv"
            full_summary.to_csv(out_path, index=False)
            print(f"  -> Saved summary: {out_path}")
            all_summary_rows.append(full_summary)

        # --- Save initialization CSV ---
        if not est_summary.empty:
            init_df = build_init_csv(
                est_summary[["Parameter", "Mean", "SD", "SE"]]
            )
            init_path = save_dir / f"init_params_{tag}.csv"
            init_df.to_csv(init_path, index=False)
            print(f"  -> Saved init CSV:  {init_path}")

        print()

    # --- Combined summary across all conditions ---
    if all_summary_rows:
        combined = pd.concat(all_summary_rows, ignore_index=True)
        combined_path = save_dir / "sim_summary_all.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Combined summary saved: {combined_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
