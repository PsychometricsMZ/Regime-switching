"""
sim_evaluation.py
-----------------
Post-simulation evaluation script. Reads sim_summary CSVs produced by
sim_summarize.py and pkl files to produce manuscript-ready outputs.

Tables  → sim_summary/tables/
  1. param_table_ms.csv    Markov-switching params: True | (Bias, RMSE) × 4 conditions
  2. param_table_sm.csv    Structural model params: same format
  3. param_table_mm.csv    Measurement model params: same format
  4. metrics_table.csv     Accuracy / Sensitivity / Specificity: Observed + Forecast × 4 cond.

Plots   → sim_summary/plots/
  5. sim_score_2x2.png        δ_t across 10-pt forecast window, 2×2 panel (±1 SD)

Usage
-----
    python sim_evaluation.py
    python sim_evaluation.py --summary_dir sim_summary --output_dir output --no-show
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical 2×2 factorial conditions  (method assumed "two_stage")
CONDITIONS = [
    {"tag": "two_stage_N50_Ntrain25",   "label": r"$\mathcal{D}_{50,25}$",  "N": 50,  "Ntrain": 25},
    {"tag": "two_stage_N100_Ntrain25",  "label": r"$\mathcal{D}_{100,25}$", "N": 100, "Ntrain": 25},
    {"tag": "two_stage_N50_Ntrain50",   "label": r"$\mathcal{D}_{50,50}$",  "N": 50,  "Ntrain": 50},
    {"tag": "two_stage_N100_Ntrain50",  "label": r"$\mathcal{D}_{100,50}$", "N": 100, "Ntrain": 50},
]

# Parameter group membership (regex patterns).
# Order within each group controls row order in tables.
PARAM_GROUPS = {
    "ms": [                                   # Markov-switching
        r"^gamma1$",
        r"^gamma2$",
        r"^gamma3_\d+$",
        r"^gamma4_active_\d+$",
        r"^gamma4_\d+$",
        r"^P12$",
    ],
    "sm": [                                   # Structural model
        r"^B11_\d+$",
        r"^B12_\d+$",
        r"^B21_\d+$",
        r"^B22_\d+$",
        r"^B31d_\d+$",
        r"^B32d_\d+$",
        r"^B41d_\d+$",
        r"^B42d_\d+$",
        r"^Q1d_\d+$",
        r"^Q2d_\d+$",
        r"^P2$",
    ],
    "mm": [                                   # Measurement model
        r"^Lmd1f\d+$",
        r"^R1d_\d+$",
        r"^R2d_\d+$",
    ],
}

# Mapping from Python parameter name to display name + idx for manuscript tables
def _parse_param(name: str):
    """Return (display_name, idx_str) for a parameter like 'B31d_2' → ('diag(B31)', '2')."""
    mapping = [
        (r"^gamma1$",              r"$\gamma_1$",         "1"),
        (r"^gamma2$",              r"$\gamma_2$",         "1"),
        (r"^gamma3_(\d+)$",        r"$\bm{\gamma}_3$",    None),
        (r"^gamma4_active_(\d+)$", r"$\bm{\gamma}_4$",    None),
        (r"^gamma4_(\d+)$",        r"$\bm{\gamma}_4$",    None),
        (r"^P12$",                 r"$P_{12}$",           "1"),
        (r"^B11_(\d+)$",           r"$\bm{b}_{11}$",      None),
        (r"^B12_(\d+)$",           r"$\bm{b}_{12}$",      None),
        (r"^B21_(\d+)$",           r"$\bm{b}_{21}$",      None),
        (r"^B22_(\d+)$",           r"$\bm{b}_{22}$",      None),
        (r"^B31d_(\d+)$",          r"$\mathrm{diag}(\bm{B}_{31})$", None),
        (r"^B32d_(\d+)$",          r"$\mathrm{diag}(\bm{B}_{32})$", None),
        (r"^B41d_(\d+)$",          r"$\mathrm{diag}(\bm{B}_{41})$", None),
        (r"^B42d_(\d+)$",          r"$\mathrm{diag}(\bm{B}_{42})$", None),
        (r"^Q1d_(\d+)$",           r"$\bm{Q}_1$",         None),
        (r"^Q2d_(\d+)$",           r"$\bm{Q}_2$",         None),
        (r"^P2$",                  r"$P_2$",              "1"),
        (r"^Lmd1f(\d+)$",          r"$\bm{\Lambda}_1$",   None),
        (r"^R1d_(\d+)$",           r"$\bm{R}_1$",         None),
        (r"^R2d_(\d+)$",           r"$\bm{R}_2$",         None),
    ]
    for pattern, display, fixed_idx in mapping:
        m = re.match(pattern, name)
        if m:
            idx = fixed_idx if fixed_idx is not None else m.group(1)
            return display, idx
    return name, "1"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_dir", default="sim_summary",
                   help="Directory with sim_summary_*.csv files (default: sim_summary/)")
    p.add_argument("--output_dir", default="output",
                   help="Directory with sim_results_*.pkl files (default: output/)")
    p.add_argument("--no-show", action="store_true",
                   help="Suppress interactive plot windows")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loader: sim_summary CSVs
# ---------------------------------------------------------------------------
def load_summaries(summary_dir: Path) -> dict[str, pd.DataFrame]:
    """Return {tag: DataFrame} for each condition found."""
    loaded = {}
    for cond in CONDITIONS:
        path = summary_dir / f"sim_summary_{cond['tag']}.csv"
        if path.exists():
            df = pd.read_csv(path)
            loaded[cond["tag"]] = df
            print(f"  Loaded: {path.name}  ({len(df)} rows)")
        else:
            print(f"  NOT FOUND: {path.name}")
    return loaded


# ---------------------------------------------------------------------------
# Loader: score function from pkl
# ---------------------------------------------------------------------------
def load_score_functions(output_dir: Path) -> dict[str, np.ndarray]:
    """Return {tag: (n_runs × n_forecast) array} from pkl files."""
    sf = {}
    for cond in CONDITIONS:
        pkl_path = output_dir / f"sim_results_{cond['tag']}.pkl"
        if not pkl_path.exists():
            print(f"  pkl NOT FOUND: {pkl_path.name}")
            continue
        with open(pkl_path, "rb") as fh:
            results = pickle.load(fh)
        arrays = []
        for res in results:
            if res is None or res.get("error") is not None:
                continue
            sfh = res.get("score_function_history")
            if sfh is not None:
                arr = np.asarray(sfh, dtype=float)
                if arr.ndim == 1:
                    arrays.append(arr)
        if arrays:
            sf[cond["tag"]] = np.vstack(arrays)   # (n_runs × n_forecast)
            print(f"  Score fn loaded: {cond['tag']}  ({len(arrays)} runs, {arrays[0].shape[0]} pts)")
    return sf


# ---------------------------------------------------------------------------
# Parameter table builder
# ---------------------------------------------------------------------------
def _match_group(param: str, group_patterns: list[str]) -> bool:
    return any(re.match(p, param) for p in group_patterns)


def build_param_table(summaries: dict[str, pd.DataFrame],
                      group_key: str) -> pd.DataFrame:
    """
    Build wide-format parameter recovery table for one group.
    Columns: Parameter | Idx | True_Value | (Bias, RMSE, Power) × 4 conditions
    Power = proportion of replications where 0 lies outside the 95% CI
    (rejection rate; for true_value = 0 parameters this is the Type-I error rate).
    """
    patterns = PARAM_GROUPS[group_key]

    # Collect all parameter names that belong to this group, in order
    all_params = []
    seen = set()
    for cond in CONDITIONS:
        tag = cond["tag"]
        if tag not in summaries:
            continue
        df = summaries[tag]
        for param in df["Parameter"].tolist():
            if param not in seen and _match_group(param, patterns):
                seen.add(param)
                all_params.append(param)

    # Sort by pattern order then by index within pattern
    def sort_key(p):
        for i, pat in enumerate(patterns):
            if re.match(pat, p):
                m = re.search(r"(\d+)$", p)
                idx = int(m.group(1)) if m else 0
                return (i, idx)
        return (999, 0)
    all_params.sort(key=sort_key)

    rows = []
    for param in all_params:
        display, idx = _parse_param(param)
        row = {"Parameter": display, "Idx": idx}

        # True value from first available condition
        true_val = np.nan
        for cond in CONDITIONS:
            df = summaries.get(cond["tag"])
            if df is None:
                continue
            sub = df[df["Parameter"] == param]
            if len(sub) > 0:
                tv = sub["True_Value"].values[0]
                if not pd.isna(tv):
                    true_val = tv
                    break
        row["True_Value"] = round(true_val, 4) if not np.isnan(true_val) else "NA"

        for cond in CONDITIONS:
            tag   = cond["tag"]
            label = cond["label"]
            df = summaries.get(tag)
            if df is None:
                row[f"{label} Bias"] = "NA"
                row[f"{label} RMSE"] = "NA"
                row[f"{label} Power"] = "NA"
                continue
            sub = df[df["Parameter"] == param]
            if len(sub) == 0:
                row[f"{label} Bias"] = "NA"
                row[f"{label} RMSE"] = "NA"
                row[f"{label} Power"] = "NA"
            else:
                bias = sub["Bias"].values[0]
                rmse = sub["RMSE"].values[0]
                row[f"{label} Bias"] = round(float(bias), 4) if not pd.isna(bias) else "NA"
                row[f"{label} RMSE"] = round(float(rmse), 4) if not pd.isna(rmse) else "NA"
                power = sub["Power"].values[0] if "Power" in sub.columns else np.nan
                row[f"{label} Power"] = round(float(power), 4) if not pd.isna(power) else "NA"
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics table builder
# ---------------------------------------------------------------------------
def build_metrics_table(summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build manuscript metrics table.
    Columns: Condition | Obs.Accuracy | Fore.Accuracy | Obs.Sensitivity |
             Fore.Sensitivity | Obs.Specificity | Fore.Specificity
    """
    metric_cols = {
        "Obs_Accuracy":     "Mean_Observed_Accuracy",
        "Fore_Accuracy":    "Mean_Forecast_Accuracy",
        "Obs_Sensitivity":  "Mean_Observed_Sensitivity",
        "Fore_Sensitivity": "Mean_Forecast_Sensitivity",
        "Obs_Specificity":  "Mean_Observed_Specificity",
        "Fore_Specificity": "Mean_Forecast_Specificity",
        # Class composition and the constant-predictor benchmark. Accuracy cannot
        # be read without these: regime 2 accumulates under the near-absorbing
        # structure, so the majority-class rate differs across conditions.
        "Obs_Regime2_Share":   "Mean_Observed_Regime2_Share",
        "Fore_Regime2_Share":  "Mean_Forecast_Regime2_Share",
        "Obs_Majority_Acc":    "Mean_Observed_Majority_Accuracy",
        "Fore_Majority_Acc":   "Mean_Forecast_Majority_Accuracy",
    }
    rows = []
    for cond in CONDITIONS:
        tag = cond["tag"]
        row = {"Condition": cond["label"], "N": cond["N"], "Ntrain": cond["Ntrain"]}
        df = summaries.get(tag)
        if df is None:
            for col in metric_cols:
                row[col] = np.nan
        else:
            # Metrics are stored as extra columns; grab first finite row
            param_rows = df[~df["Parameter"].isin(["LogLikelihood", "ConvergenceRate"])]
            for display_col, src_col in metric_cols.items():
                if src_col in param_rows.columns:
                    vals = pd.to_numeric(param_rows[src_col], errors="coerce").dropna()
                    row[display_col] = round(float(vals.iloc[0]), 4) if len(vals) > 0 else np.nan
                else:
                    row[display_col] = np.nan

        # Balanced accuracy = (sensitivity + specificity) / 2. Unlike plain
        # accuracy it is invariant to the class composition, and a constant
        # predictor scores exactly 0.50 whatever the base rate. It is therefore
        # the metric that can be compared across conditions here.
        for pre in ("Obs", "Fore"):
            se, sp = row.get(f"{pre}_Sensitivity"), row.get(f"{pre}_Specificity")
            row[f"{pre}_Balanced_Acc"] = (
                round((se + sp) / 2, 4)
                if se is not None and sp is not None
                and np.isfinite(se) and np.isfinite(sp) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Score function plot (2×2 panel)
# ---------------------------------------------------------------------------
def plot_score_function(score_fns: dict[str, np.ndarray],
                        summaries: dict[str, pd.DataFrame],
                        save_path: Path, show: bool = False):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    axes_flat = axes.flatten()

    panel_titles = {
        "two_stage_N50_Ntrain25":  r"$N=50,\ T_\mathrm{train}=25$",
        "two_stage_N100_Ntrain25": r"$N=100,\ T_\mathrm{train}=25$",
        "two_stage_N50_Ntrain50":  r"$N=50,\ T_\mathrm{train}=50$",
        "two_stage_N100_Ntrain50": r"$N=100,\ T_\mathrm{train}=50$",
    }

    for idx, cond in enumerate(CONDITIONS):
        tag    = cond["tag"]
        ax     = axes_flat[idx]
        ntrain = cond["Ntrain"]

        if tag not in score_fns:
            ax.set_title(panel_titles.get(tag, tag))
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            continue

        mat  = score_fns[tag]            # (n_runs × n_forecast)
        mean = np.nanmean(mat, axis=0)
        sd   = np.nanstd(mat, axis=0, ddof=1)
        t    = np.arange(ntrain + 1, ntrain + len(mean) + 1)

        ax.plot(t, mean, color="steelblue", linewidth=1.5)
        ax.fill_between(t, mean - sd, mean + sd, color="steelblue", alpha=0.25)
        ax.set_title(panel_titles.get(tag, tag), fontsize=11)
        ax.set_xlabel("Time point", fontsize=9)
        ax.set_ylabel(r"Score function $\delta_t$", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(r"Score function $\delta_t$ across forecast window by condition",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_score_function_overlay(score_fns: dict[str, np.ndarray],
                                save_path: Path, show: bool = False):
    """All 4 conditions overlaid in a single panel for direct comparison.
    Mean as solid line; SD bounds as thin dotted lines of the same colour.
    Legend placed below the axes so it never overlaps the plot area.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels = {
        "two_stage_N50_Ntrain25":  r"$N=50,\ T_{\rm est}=25$",
        "two_stage_N100_Ntrain25": r"$N=100,\ T_{\rm est}=25$",
        "two_stage_N50_Ntrain50":  r"$N=50,\ T_{\rm est}=50$",
        "two_stage_N100_Ntrain50": r"$N=100,\ T_{\rm est}=50$",
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for (cond, color) in zip(CONDITIONS, colors):
        tag = cond["tag"]
        if tag not in score_fns:
            continue
        mat  = score_fns[tag]
        mean = np.nanmean(mat, axis=0)
        sd   = np.nanstd(mat, axis=0, ddof=1)
        t    = np.arange(1, len(mean) + 1)   # relative forecast step 1..10
        # Mean line: solid, same style for all conditions, distinguished by color
        ax.plot(t, mean, color=color, linewidth=2.0, linestyle="-",
                label=labels.get(tag, tag))
        # SD bounds: dotted lines, same color
        ax.plot(t, mean + sd, color=color, linewidth=0.8, linestyle=":",
                alpha=0.7)
        ax.plot(t, mean - sd, color=color, linewidth=0.8, linestyle=":",
                alpha=0.7)

    # Truncate the vertical axis for legibility (the +1 SD line of
    # N=100, T_est=25 reaches 0.077 at the final step; noted in the caption).
    ax.set_ylim(0.042, 0.066)
    ax.set_xlabel("Forecast step", fontsize=11)
    ax.set_ylabel(r"Score function $\delta_t$", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend below axes, outside the plot area
    ax.legend(fontsize=9, framealpha=0.9, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args    = parse_args()
    show    = not args.no_show
    summary_dir = Path(args.summary_dir)
    output_dir  = Path(args.output_dir)
    tables_dir  = summary_dir / "tables"
    plots_dir   = summary_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("sim_evaluation.py")
    print("=" * 60)

    # --- Load summaries ---
    print(f"\nLoading sim_summary CSVs from '{summary_dir}' ...")
    summaries = load_summaries(summary_dir)
    n_loaded  = sum(1 for c in CONDITIONS if c["tag"] in summaries)
    print(f"  {n_loaded}/4 conditions loaded.")

    if n_loaded == 0:
        print("No summary files found. Run sim_summarize.py first.")
        return

    # --- [1-3] Parameter tables ---
    group_meta = {
        "ms": ("Markov-switching parameters",  "param_table_ms.csv"),
        "sm": ("Structural model parameters",  "param_table_sm.csv"),
        "mm": ("Measurement model parameters", "param_table_mm.csv"),
    }
    for gkey, (title, fname) in group_meta.items():
        print(f"\n[{gkey.upper()}] {title}")
        tbl = build_param_table(summaries, gkey)
        if tbl.empty:
            print("  No parameters found for this group.")
            continue
        path = tables_dir / fname
        tbl.to_csv(path, index=False)
        print(f"  Rows: {len(tbl)}  ->  {path}")
        print(tbl.to_string(index=False))

    # --- [4] Metrics table ---
    print("\n[METRICS] Accuracy / Sensitivity / Specificity")
    met_tbl = build_metrics_table(summaries)
    met_path = tables_dir / "metrics_table.csv"
    met_tbl.to_csv(met_path, index=False)
    print(f"  -> {met_path}")
    print(met_tbl.to_string(index=False))

    # --- [5] Score function plot ---
    print(f"\n[PLOT] Score function 2×2 panel — loading pkl files from '{output_dir}' ...")
    score_fns = load_score_functions(output_dir)
    if score_fns:
        plot_score_function(score_fns, summaries,
                            plots_dir / "sim_score_2x2.png", show=show)
        plot_score_function_overlay(score_fns,
                                    plots_dir / "sim_score_overlay.png", show=show)
    else:
        print("  No score function data found; skipping plot.")

    # --- Convergence rate summary ---
    print("\n[CONVERGENCE]")
    for cond in CONDITIONS:
        tag = cond["tag"]
        df  = summaries.get(tag)
        if df is None:
            continue
        cr_row = df[df["Parameter"] == "ConvergenceRate"]
        if len(cr_row) > 0:
            rate = cr_row["Mean"].values[0]
            n    = cr_row["N_Valid"].values[0]
            print(f"  {cond['label']:25s}  {rate*100:.1f}%  ({int(n)} total)")

    print("\nDone.")


if __name__ == "__main__":
    main()
