"""
Talk-only variant of the score-function overlay figure.

Same data and styling as sim_evaluation.plot_score_function_overlay, but the
legend sits inside the axes so the plot itself can be larger on the slide.
The manuscript figure (sim_summary/plots/sim_score_overlay.png) is left alone.

Run from Codes/Simulation:
    python make_talk_score_overlay.py
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

OUTPUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "output_g35")
OUT_PNG = Path("../../IMPS2026_talk/beamer/figs/sim_score_overlay.png")

CONDITIONS = [
    ("two_stage_N50_Ntrain25",  r"$N=50,\ T_{\rm est}=25$",  "#1f77b4"),
    ("two_stage_N100_Ntrain25", r"$N=100,\ T_{\rm est}=25$", "#ff7f0e"),
    ("two_stage_N50_Ntrain50",  r"$N=50,\ T_{\rm est}=50$",  "#2ca02c"),
    ("two_stage_N100_Ntrain50", r"$N=100,\ T_{\rm est}=50$", "#d62728"),
]


def load_score_fns():
    out = {}
    for tag, _, _ in CONDITIONS:
        p = OUTPUT_DIR / f"sim_results_{tag}.pkl"
        if not p.exists():
            print(f"  missing: {p}")
            continue
        with open(p, "rb") as f:
            results = pickle.load(f)
        runs = []
        for res in results:
            if res is None or res.get("error") is not None:
                continue
            sfh = res.get("score_function_history")
            if sfh is not None:
                arr = np.asarray(sfh, dtype=float)
                if arr.ndim == 1:
                    runs.append(arr)
        if runs:
            out[tag] = np.vstack(runs)
            print(f"  {tag}: {out[tag].shape[0]} runs x {out[tag].shape[1]} pts")
    return out


def main():
    score_fns = load_score_fns()
    if not score_fns:
        raise RuntimeError("no score functions found in output/")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tag, label, color in CONDITIONS:
        if tag not in score_fns:
            continue
        mat = score_fns[tag]
        mean = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0, ddof=1)
        t = np.arange(1, len(mean) + 1)
        ax.plot(t, mean, color=color, linewidth=2.0, label=label)
        ax.plot(t, mean + sd, color=color, linewidth=0.8, linestyle=":", alpha=0.7)
        ax.plot(t, mean - sd, color=color, linewidth=0.8, linestyle=":", alpha=0.7)

    ax.set_xlabel("Forecast step", fontsize=12)
    ax.set_ylabel(r"Score function $\delta_t$", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend inside, upper left, two columns so it stays flat
    ax.legend(fontsize=9, framealpha=0.85, loc="upper left", ncol=2,
              borderaxespad=0.4)
    # headroom so the legend does not sit on the curves
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.32 * (hi - lo))

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
