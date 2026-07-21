"""
Regenerate the LaTeX bodies for Appendix D (Tables D1-D4) from a summary directory.

Writes four .tex fragments containing only the tabular rows, so they can be pasted
into A4. simulation.tex without touching the surrounding table environments.

Run from Codes/Simulation:
    python make_appendix_tables.py sim_summary_g35
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "sim_summary_g35")
OUT_DIR = Path("comparison/appendix_tex")
CONDS = ["N50_Ntrain25", "N100_Ntrain25", "N50_Ntrain50", "N100_Ntrain50"]

# (regex on Parameter, pretty LaTeX name, index shown)
GROUPS_MS = [("^gamma1$", r"$\gamma_1$"), ("^gamma2$", r"$\gamma_2$"),
             ("^gamma3_", r"$\bm{\gamma}_3$"), ("^gamma4_", r"$\bm{\gamma}_4$"),
             ("^P12$", r"$P_{12}^{*}$")]
GROUPS_SM = [("^B11_", r"$\bm{b}_{11}$"), ("^B12_", r"$\bm{b}_{12}$"),
             ("^B21_", r"$\bm{b}_{21}$"), ("^B22_", r"$\bm{b}_{22}$"),
             ("^B31d_", r"$\mathrm{diag}(\bm{B}_{31})$"),
             ("^B32d_", r"$\mathrm{diag}(\bm{B}_{32})$"),
             ("^B41d_", r"$\mathrm{diag}(\bm{B}_{41})$"),
             ("^B42d_", r"$\mathrm{diag}(\bm{B}_{42})$"),
             ("^Q1d_", r"$\bm{Q}_1$"), ("^Q2d_", r"$\bm{Q}_2$"), ("^P2$", r"$P_2$")]
GROUPS_MM = [("^R1d_", r"$\bm{R}_1$"), ("^R2d_", r"$\bm{R}_2$")]


def load():
    out = {}
    for c in CONDS:
        p = SUMMARY_DIR / f"sim_summary_two_stage_{c}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        out[c] = pd.read_csv(p)
    return out


def fmt(x, nd=2, math=True):
    """Match the existing Appendix D style: 2 decimals, $...$, --- for missing."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    s = f"{x:.{nd}f}"
    return f"${s}$" if math else s


def rows_for(F, groups, with_power=False):
    """Yield LaTeX rows. Parameters are matched per group and ordered by suffix."""
    lines = []
    for pat, pretty in groups:
        names = sorted(
            {n for n in F[CONDS[0]].Parameter if pd.Series([n]).str.match(pat).iloc[0]},
            key=lambda s: (len(s), s),
        )
        for k, name in enumerate(names):
            idx = name.rsplit("_", 1)[-1] if "_" in name and name.rsplit("_", 1)[-1].isdigit() else "1"
            base = F[CONDS[0]]
            true_v = base.loc[base.Parameter == name, "True_Value"]
            cells = []
            for c in CONDS:
                r = F[c].loc[F[c].Parameter == name]
                if len(r) == 0:
                    cells += ["NA"] * (1 if with_power else 3)
                    continue
                r = r.iloc[0]
                if with_power:
                    cells.append(fmt(r.get("Power"), math=False))
                else:
                    cells += [fmt(r.get("Bias")),
                              fmt(r.get("RMSE"), math=False),
                              fmt(r.get("Coverage_Rate"), math=False)]
            if k == 0:
                lines.append(f"        {pretty}")
            lines.append(f"         & {idx} & {fmt(float(true_v.iloc[0]))} & "
                         + " & ".join(cells) + r" \\")
    return lines


def main():
    F = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tag, groups in [("ms", GROUPS_MS), ("sm", GROUPS_SM), ("mm", GROUPS_MM)]:
        body = "\n".join(rows_for(F, groups))
        (OUT_DIR / f"D_{tag}.tex").write_text(body + "\n", encoding="utf-8", newline="")
        print(f"  wrote comparison/appendix_tex/D_{tag}.tex  ({len(body.splitlines())} rows)")
    # Power table: different layout (one row per parameter, index folded into the
    # symbol, section headers, no Idx column, parameters with undefined power omitted)
    POWER_SECTIONS = [
        ("Markov-switching parameters",
         [("gamma1", r"$\gamma_{1}$"), ("gamma2", r"$\gamma_{2}$"),
          ("gamma3_1", r"$\bm{\gamma}_{3,1}$"), ("gamma3_2", r"$\bm{\gamma}_{3,2}$"),
          ("gamma4_1", r"$\bm{\gamma}_{4,1}$"), ("gamma4_2", r"$\bm{\gamma}_{4,2}$")]),
        ("Structural model parameters",
         [("B11_1", r"$\bm{b}_{11,1}$"), ("B11_2", r"$\bm{b}_{11,2}$"),
          ("B21_1", r"$\bm{b}_{21,1}$"), ("B21_2", r"$\bm{b}_{21,2}$"),
          ("B22_1", r"$\bm{b}_{22,1}$"), ("B22_2", r"$\bm{b}_{22,2}$"),
          ("B31d_1", r"$\mathrm{diag}(\bm{B}_{31})_1$"), ("B31d_2", r"$\mathrm{diag}(\bm{B}_{31})_2$"),
          ("B32d_1", r"$\mathrm{diag}(\bm{B}_{32})_1$"), ("B32d_2", r"$\mathrm{diag}(\bm{B}_{32})_2$"),
          ("B41d_1", r"$\mathrm{diag}(\bm{B}_{41})_1$"), ("B41d_2", r"$\mathrm{diag}(\bm{B}_{41})_2$"),
          ("B42d_1", r"$\mathrm{diag}(\bm{B}_{42})_1$"), ("B42d_2", r"$\mathrm{diag}(\bm{B}_{42})_2$"),
          ("Q1d_1", r"$\bm{Q}_{1,1}$"), ("Q1d_2", r"$\bm{Q}_{1,2}$")]),
        ("Measurement model parameters",
         [("R1d_1", r"$\bm{R}_{1,1}$"), ("R1d_2", r"$\bm{R}_{1,2}$"),
          ("R1d_3", r"$\bm{R}_{1,3}$"), ("R1d_4", r"$\bm{R}_{1,4}$"),
          ("R2d_1", r"$\bm{R}_{2,1}$"), ("R2d_2", r"$\bm{R}_{2,2}$")]),
    ]
    lines = []
    for k, (title, items) in enumerate(POWER_SECTIONS):
        if k:
            lines.append(r"        \midrule")
        lines.append(r"        \multicolumn{6}{@{}l}{\textit{" + title + r"}} \\")
        for name, pretty in items:
            base = F[CONDS[0]]
            tv = base.loc[base.Parameter == name, "True_Value"]
            if len(tv) == 0:
                continue
            cells = []
            for c in CONDS:
                r = F[c].loc[F[c].Parameter == name]
                cells.append(fmt(r.iloc[0].get("Power"), math=False) if len(r) else "---")
            lines.append(f"        {pretty} & {fmt(float(tv.iloc[0]))} & " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    (OUT_DIR / "D_power.tex").write_text(body + "\n", encoding="utf-8", newline="")
    print(f"  wrote comparison/appendix_tex/D_power.tex  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
