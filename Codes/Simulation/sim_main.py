# sim_main_multi.py

import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import config
from sim_utils import load_true_parameters2
from sim_data_generation import generate_sim_data
from sim_filtering import filtering


def load_sim_prior(csv_path):
    """Load Kelava warm-start priors from CSV.

    Returns dict: param_base -> {'mean': scalar_or_array, 'sd': scalar_or_array}
    Vector params (e.g. B11_1, B11_2) are assembled in index order.
    """
    df = pd.read_csv(csv_path)
    prior = {}
    for _, row in df.iterrows():
        pname = str(row["Parameter"])
        mean_val = float(row["Mean"])
        sd_val   = float(row["SD"])
        parts = pname.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            base = parts[0]
            if base not in prior:
                prior[base] = {"mean": [], "sd": []}
            prior[base]["mean"].append(mean_val)
            prior[base]["sd"].append(sd_val)
        else:
            prior[pname] = {"mean": mean_val, "sd": sd_val}
    for v in prior.values():
        if isinstance(v["mean"], list):
            v["mean"] = np.array(v["mean"], dtype=np.float32)
            v["sd"]   = np.array(v["sd"],   dtype=np.float32)
    return prior


def calculate_metrics(true_vec, pred_vec, prefix):
    if len(true_vec) == 0:
        return pd.DataFrame()

    true_vec = np.asarray(true_vec)
    pred_vec = np.asarray(pred_vec)

    TP = np.sum((pred_vec == 1) & (true_vec == 1))
    TN = np.sum((pred_vec == 0) & (true_vec == 0))
    FP = np.sum((pred_vec == 1) & (true_vec == 0))
    FN = np.sum((pred_vec == 0) & (true_vec == 1))

    accuracy = (TP + TN) / len(true_vec)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    f1_score = (
        2 * (precision * sensitivity) / (precision + sensitivity)
        if (not np.isnan(precision) and not np.isnan(sensitivity) and (precision + sensitivity) > 0)
        else np.nan
    )

    df = pd.DataFrame({
        "Accuracy": [accuracy],
        "Sensitivity": [sensitivity],
        "Specificity": [specificity],
        "Precision": [precision],
        "F1_Score": [f1_score]
    })
    df.columns = [f"{prefix}{col}" for col in df.columns]
    return df


def run_one_simulation(args):
    i, method, N_val, n_train_val, true_params, config, device, sim_prior = args

    try:
        sim_data_list = generate_sim_data(
            N=N_val,
            Nt=config["Nt"],
            U1=2,
            O1=4,
            O2=2,
            true_params=true_params,
            seed=123 + i
        )

        best_ll = -np.inf
        best_res = None
        MAX_INIT_ATTEMPTS = config["MAX_INIT_ATTEMPTS"]

        for init_attempt in range(1, MAX_INIT_ATTEMPTS + 1):
            try:
                current_res = filtering(
                    seed=10 * i,
                    N=sim_data_list["N"],
                    Nt=sim_data_list["Nt"],
                    O1=sim_data_list["O1"],
                    O2=sim_data_list["O2"],
                    U1=sim_data_list["U1"],
                    y1=sim_data_list["y1"],
                    y2=sim_data_list["y2"],
                    DO=sim_data_list["DO"],
                    init=init_attempt,
                    maxIter=config["MAX_ITER"],
                    n_train=n_train_val,
                    patience=config["PATIENCE"],
                    min_delta=1e-4,
                    compute_se=config["COMPUTE_SE"],
                    se_sample_size=config["SE_SAMPLE_SIZE"],
                    verbose=False,
                    show_progress=True,
                    device=device,
                    method=method,
                    two_stage_outer_loops=config["TWO_STAGE_OUTER_LOOPS"],
                    two_stage_damping=config["TWO_STAGE_DAMPING"],
                    fix_gamma3=True,
                    fix_gamma4=True,
                    fix_gamma1=True,
                    fix_p12=True,
                    p12_fixed_value=1e-12,
                    sim_prior=sim_prior,
                )
            except Exception as e:
                print(f"     !!! Sim {i} Attempt {init_attempt} Error in filtering: {e}")
                current_res = None

            if (
                current_res is not None
                and current_res.get("sumLL_best") is not None
                and np.isfinite(current_res["sumLL_best"])
            ):
                if current_res["sumLL_best"] > best_ll:
                    best_ll = current_res["sumLL_best"]
                    best_res = current_res

            gc.collect()

        metrics_res = None
        score_function_over_time = None

        if best_res is not None and best_res.get("trajectories") is not None:
            true_eta = sim_data_list["eta1_true"]
            true_regime_binary = sim_data_list["S_true"] - 1

            est_eta = best_res["trajectories"]["eta_est"]
            est_prob_regime2 = best_res["trajectories"]["regime_prob"]
            pred_regime_binary = (est_prob_regime2 > 0.5).astype(int)

            mse_tensor = (true_eta - est_eta) ** 2

            Nt = sim_data_list["Nt"]
            n_train = n_train_val

            t_train = np.arange(0, n_train)
            t_forecast = np.arange(n_train, n_train + 10)  # fixed 10-point forecast window

            mse_tensor_fc = mse_tensor[:, t_forecast, :]
            score_function_over_time = np.mean(mse_tensor_fc, axis=(0, 2))

            flat_true_train = true_regime_binary[:, t_train].reshape(-1)
            flat_pred_train = pred_regime_binary[:, t_train].reshape(-1)
            metrics_train = calculate_metrics(flat_true_train, flat_pred_train, "Observed_")

            flat_true_fc = true_regime_binary[:, t_forecast].reshape(-1)
            flat_pred_fc = pred_regime_binary[:, t_forecast].reshape(-1)
            metrics_fc = calculate_metrics(flat_true_fc, flat_pred_fc, "Forecast_")

            flat_true_all = true_regime_binary.reshape(-1)
            flat_pred_all = pred_regime_binary.reshape(-1)
            metrics_all = calculate_metrics(flat_true_all, flat_pred_all, "All_")

            metrics_res = pd.concat(
                [
                    pd.DataFrame({"All_LL_Best": [best_ll]}),
                    metrics_all,
                    metrics_train,
                    metrics_fc
                ],
                axis=1
            )

            cols_to_drop = [
                col for col in metrics_res.columns
                if col in [
                    "All_Accuracy",
                    "All_Sensitivity",
                    "All_Specificity",
                    "All_Precision",
                    "All_F1_Score"
                ]
            ]
            metrics_res = metrics_res.drop(columns=cols_to_drop)

        if best_res is not None:
            return i, {
                "condition": {"METHOD": method, "N": N_val, "NT_TRAIN": n_train_val},
                "method": method,
                "true_params": true_params,
                "estimates": best_res["final_estimates"],
                "metrics": metrics_res,
                "score_function_history": score_function_over_time,
                "P2_estimated": best_res["P2_estimated"],
                "Q2_estimated_diag": best_res.get("Q2_estimated_diag"),
                "conv_status": "Converged",
                "ll": best_ll
            }

        return i, {
            "condition": {"METHOD": method, "N": N_val, "NT_TRAIN": n_train_val},
            "method": method,
            "error": "Optimization Failed"
        }

    except Exception as e:
        return i, {
            "condition": {"METHOD": method, "N": N_val, "NT_TRAIN": n_train_val},
            "method": method,
            "error": f"Worker Failed: {e}"
        }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if device == "cuda":
        print("CUDA detected, but multiprocessing across many workers is usually best kept on CPU.")
        device = "cpu"
        print("Using device for parallel workers:", device)

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    true_params = load_true_parameters2(config["true_params_file"])
    print(config["true_params_file"])

    sim_prior = load_sim_prior(config["sim_init_path"])
    print(f"Kelava sim_prior loaded: {len(sim_prior)} parameter groups")

    max_workers = min(16, os.cpu_count() or 1)
    print("Using workers:", max_workers)

    # Psychometrika revision: compare filtering methods over the 2x2 factorial.
    for method in config["FILTER_METHODS"]:
        for N_val in config["N_CONDITIONS"]:
            for n_train_val in config["NT_TRAIN"]:
                print(
                    f"\n### STARTING CONDITION: method = {method}, N = {N_val}, "
                    f"NT_TRAIN = {n_train_val} ###\n"
                )
                print(
                    f"===== Starting {config['N_SIM']} simulations for method={method}, "
                    f"N={N_val}, NT_TRAIN={n_train_val} ====="
                )

                results_list = [None] * config["N_SIM"]

                task_args = [
                    (i, method, N_val, n_train_val, true_params, config, device, sim_prior)
                    for i in range(1, config["N_SIM"] + 1)
                ]

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_i = {
                        executor.submit(run_one_simulation, args): args[0]
                        for args in task_args
                    }

                    for future in as_completed(future_to_i):
                        i = future_to_i[future]
                        try:
                            sim_index, result = future.result()
                            results_list[sim_index - 1] = result

                            if result.get("error") is not None:
                                print(f"{method} sim {sim_index}: failed -> {result['error']}")
                            else:
                                print(f"{method} sim {sim_index}: done, ll={result['ll']:.4f}")
                        except Exception as e:
                            print(f"{method} sim {i}: crashed -> {e}")
                            results_list[i - 1] = {
                                "condition": {"METHOD": method, "N": N_val, "NT_TRAIN": n_train_val},
                                "method": method,
                                "error": f"Future Failed: {e}"
                            }

                file_name = f"sim_results_{method}_N{N_val}_Ntrain{n_train_val}.pkl"
                file_path = os.path.join(str(config["output_dir"]), file_name)

                with open(file_path, "wb") as f:
                    pickle.dump(results_list, f)

                print(f"Saved results to: {file_name}")

    print("\n===== All Simulations Complete =====")


if __name__ == "__main__":
    main()
