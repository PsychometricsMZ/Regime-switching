"""
parameter_estimates_loaded.csv を正しく再構築する。
実行: python fix_params_csv.py  (Codes/Simulation/ から)
"""
import pickle, numpy as np, pandas as pd
from pathlib import Path

_HERE      = Path(__file__).resolve().parent
bak_path   = _HERE / "parameter_estimates_loaded.csv.bak"
csv_path   = _HERE / "parameter_estimates_loaded.csv"
results_dir = _HERE.parent / "Empirical" / "output" / "results"

# 1. バックアップから全113行を復元
df = pd.read_csv(bak_path)
print(f"Restored from backup: {len(df)} rows")

# 2. 最良 emp pkl を読み込む
best_ll, best = -np.inf, None
for p in sorted(results_dir.glob("emp_result_run*.pkl")):
    res = pickle.load(open(p, "rb"))
    if res and np.isfinite(res.get("sumLL_best", -np.inf)) and res["sumLL_best"] > best_ll:
        best_ll, best = res["sumLL_best"], res

df_emp      = best["final_estimates"].copy()
emp_dict    = dict(zip(df_emp["Parameter"], df_emp["Estimate"]))
emp_se_dict = dict(zip(df_emp["Parameter"], df_emp["SE"]))
print(f"Emp estimates: {len(emp_dict)} params, sumLL={best_ll:.4f}")

# 3. emp MLE で上書き
n_updated = 0
for i, row in df.iterrows():
    pname = row["Parameter"]
    if pname in emp_dict:
        df.at[i, "Estimate"] = emp_dict[pname]
        df.at[i, "SE"]       = emp_se_dict.get(pname, np.nan)
        df.at[i, "Type"]     = "EmpMLE"
        n_updated += 1
print(f"EmpMLE updated: {n_updated} / {len(df)}")

# 4. gamma3/gamma4 を 0 に固定（DGP と推定モデルの一致）
g34 = df['Parameter'].str.startswith('gamma3_') | df['Parameter'].str.startswith('gamma4_')
df.loc[g34, 'Estimate'] = 0.0
df.loc[g34, 'Type']     = 'Fixed'
print(f"gamma3/gamma4 set to 0: {g34.sum()} rows")

# 5. 保存
df.to_csv(csv_path, index=False)
print(f"Saved: {len(df)} rows")
print("Done.")
