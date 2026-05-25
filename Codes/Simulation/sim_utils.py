"""
sim_utils.py
------------
Utility functions for the simulation pipeline.

Functions
---------
load_true_parameters2 : Load and remap empirical parameters to simulation
                        dimensions (U1=7 -> U1=2).
calc_summary_stats    : Compute bias, RMSE, SE, and CI coverage for a
                        set of parameter estimates vs. true values.
"""

import os
import numpy as np
import pandas as pd


def load_true_parameters2(param_file_path):
    """
    Load true parameter values from a CSV and remap empirical (U1=7)
    parameters to simulation dimensions (U1=2) by selecting factors 2 and 3.

    Returns
    -------
    dict  {parameter_name: true_value}
    """
    if not os.path.exists(param_file_path):
        raise FileNotFoundError(f"Parameter file not found at: {param_file_path}")

    params_df = pd.read_csv(param_file_path)

    # Mapping: empirical U1=7 indices -> simulation U1=2 indices
    # Factors 2 and 3 of the empirical model become factors 1 and 2.
    param_map = pd.DataFrame({
        "Original": [
            "R1d_4",  "R1d_5",  "R1d_6",  "R1d_7",
            "log_R1d_4", "log_R1d_5", "log_R1d_6", "log_R1d_7",
            "B11_2", "B11_3", "B12_2", "B12_3",
            "B21_2", "B21_3", "B22_2", "B22_3",
            "B31d_2", "B31d_3", "B32d_2", "B32d_3",
            "B41d_2", "B41d_3", "B42d_2", "B42d_3",
            "Q1d_2", "Q1d_3", "log_Q1d_2", "log_Q1d_3",
            "Q2d_2", "Q2d_3", "log_Q2d_2", "log_Q2d_3",
            "gamma3_2", "gamma3_3", "gamma4_2", "gamma4_3",
        ],
        "New": [
            "R1d_1",  "R1d_2",  "R1d_3",  "R1d_4",
            "log_R1d_1", "log_R1d_2", "log_R1d_3", "log_R1d_4",
            "B11_1", "B11_2", "B12_1", "B12_2",
            "B21_1", "B21_2", "B22_1", "B22_2",
            "B31d_1", "B31d_2", "B32d_1", "B32d_2",
            "B41d_1", "B41d_2", "B42d_1", "B42d_2",
            "Q1d_1", "Q1d_2", "log_Q1d_1", "log_Q1d_2",
            "Q2d_1", "Q2d_2", "log_Q2d_1", "log_Q2d_2",
            "gamma3_1", "gamma3_2", "gamma4_1", "gamma4_2",
        ],
    })

    # Parameters not subject to remapping (scalars and between-level params)
    other_params = params_df[
        (~params_df["Parameter"].str.contains("_", regex=False)) |
        (params_df["Parameter"].str.contains(r"^(?:Lmd2_|R2d_|log_R2d_)", regex=True))
    ].copy()

    remapped = (
        params_df
        .merge(param_map, left_on="Parameter", right_on="Original", how="inner")
        [["New", "Estimate"]]
        .rename(columns={"New": "Parameter", "Estimate": "Value"})
    )

    others = other_params[["Parameter", "Estimate"]].rename(columns={"Estimate": "Value"})

    final_df = pd.concat([others, remapped], ignore_index=True)
    return dict(zip(final_df["Parameter"], final_df["Value"]))


def calc_summary_stats(estimates_df, true_params):
    """
    Compute per-parameter summary statistics across simulation replications.

    Parameters
    ----------
    estimates_df : DataFrame with columns [Parameter, Estimate, SE]
    true_params  : dict {parameter_name: true_value}

    Returns
    -------
    DataFrame with columns:
        Parameter, True_Value, Mean_Est, Bias, RMSE,
        SE_Avg, SD_Est, Coverage_Rate, Valid_N
    """
    true_df = pd.DataFrame({
        "Parameter": list(true_params.keys()),
        "TrueValue": pd.to_numeric(list(true_params.values()), errors="coerce"),
    })

    merged = estimates_df.merge(true_df, on="Parameter", how="left")
    merged["Bias"]         = merged["Estimate"] - merged["TrueValue"]
    merged["SquaredError"] = merged["Bias"] ** 2
    merged["CI_Lower"]     = merged["Estimate"] - 1.96 * merged["SE"]
    merged["CI_Upper"]     = merged["Estimate"] + 1.96 * merged["SE"]
    merged["Coverage"]     = (
        (merged["TrueValue"] >= merged["CI_Lower"]) &
        (merged["TrueValue"] <= merged["CI_Upper"])
    )

    summary = (
        merged.groupby("Parameter", dropna=False)
        .agg(
            True_Value    =("TrueValue",    lambda x: np.nanmean(x)),
            Mean_Est      =("Estimate",     lambda x: np.nanmean(x)),
            Bias          =("Bias",         lambda x: np.nanmean(x)),
            RMSE          =("SquaredError", lambda x: np.sqrt(np.nanmean(x))),
            SE_Avg        =("SE",           lambda x: np.nanmean(x)),
            SD_Est        =("Estimate",     lambda x: np.nanstd(x, ddof=1)),
            Coverage_Rate =("Coverage",     lambda x: np.nanmean(x.astype(float))),
            Valid_N       =("Estimate",     lambda x: np.sum(~pd.isna(x))),
        )
        .reset_index()
    )
    return summary
