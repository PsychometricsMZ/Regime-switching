import os
import numpy as np
import pandas as pd


# ==============================================================================
# 1. Load and Remap Parameters
# ==============================================================================

#def load_true_parameters(param_file_path):
#    if not os.path.exists(param_file_path):
#        raise FileNotFoundError(f"Parameter file not found: {param_file_path}")
#
#    params_df = pd.read_csv(param_file_path)
#
#    # --------------------------------------------------------------------------
#    # A. Dimensionality Reduction Mapping (U1=7 -> U1=2, O1=17 -> O1=4)
#    # --------------------------------------------------------------------------
#    param_map = pd.DataFrame({
#        "Original": [
#            # Indicators (R1)
#            "R1d_4", "R1d_5", "R1d_6", "R1d_7",
#            "log_R1d_4", "log_R1d_5", "log_R1d_6", "log_R1d_7",
#            # Transition Matrix (B)
#            "B11_2", "B11_3", "B12_2", "B12_3",
#            "B21_2", "B21_3", "B22_2", "B22_3",
#            "B31d_2", "B31d_3", "B32d_2", "B32d_3",
#            "B41d_2", "B41d_3", "B42d_2", "B42d_3",
#            # Variances (Q)
#            "Q1d_2", "Q1d_3", "log_Q1d_2", "log_Q1d_3",
#            "Q2d_2", "Q2d_3", "log_Q2d_2", "log_Q2d_3",
#            # Switching Parameters (gamma)
#            "gamma3_2", "gamma3_3", "gamma4_2", "gamma4_3"
#        ],
#        "New": [
#            "R1d_1", "R1d_2", "R1d_3", "R1d_4",
#            "log_R1d_1", "log_R1d_2", "log_R1d_3", "log_R1d_4",
#            "B11_1", "B11_2", "B12_1", "B12_2",
#            "B21_1", "B21_2", "B22_1", "B22_2",
#            "B31d_1", "B31d_2", "B32d_1", "B32d_2",
#            "B41d_1", "B41d_2", "B42d_1", "B42d_2",
#            "Q1d_1", "Q1d_2", "log_Q1d_1", "log_Q1d_2",
#            "Q2d_1", "Q2d_2", "log_Q2d_1", "log_Q2d_2",
#            "gamma3_1", "gamma3_2", "gamma4_1", "gamma4_2"
#        ]
#    })
#
#    # --------------------------------------------------------------------------
#    # B. Capture "Other" Parameters
#    # --------------------------------------------------------------------------
#    other_params = params_df[
#        (~params_df["Parameter"].str.contains("_", regex=False)) |
#        (params_df["Parameter"].str.contains(r"^(Lmd2_|R2d_|log_R2d_)", regex=True))
#    ].copy()
#
#    # --------------------------------------------------------------------------
#    # C. Merge and Format
#    # --------------------------------------------------------------------------
#    remapped_params = params_df.merge(
#        param_map,
#        left_on="Parameter",
#        right_on="Original",
#        how="inner"
#    )[["New", "Estimate"]].copy()
#    remapped_params.columns = ["Parameter", "Value"]
#
#    other_params_selected = other_params[["Parameter", "Estimate"]].copy()
#    other_params_selected.columns = ["Parameter", "Value"]
#
#    final_params_df = pd.concat(
#        [other_params_selected, remapped_params],
#        axis=0,
#        ignore_index=True
#    )
#
#    true_params_list = dict(zip(final_params_df["Parameter"], final_params_df["Value"]))
#
#    return true_params_list
#

# ==============================================================================
# 2. Data Generation Function (Scenario B Adjusted)
# ==============================================================================

def generate_sim_data(seed, N, Nt, O1, O2, U1, true_params, N_IMPUTE=None):
    np.random.seed(seed)
    epsilon = 1e-12

    # --- 1. Extract Parameters (Using Remapped Names) ---

    # Transition Matrices (2x2 for U1=2)
    B1_s = np.array([
        [true_params["B11_1"], true_params["B11_2"]],
        [true_params["B12_1"], true_params["B12_2"]]
    ], dtype=float)

    B2_s = np.array([
        [true_params["B21_1"], true_params["B21_2"]],
        [true_params["B22_1"], true_params["B22_2"]]
    ], dtype=float)

    B3_s1 = np.diag([true_params["B31d_1"], true_params["B31d_2"]])
    B3_s2 = np.diag([true_params["B32d_1"], true_params["B32d_2"]])
    B3_s_diag = np.stack([B3_s1, B3_s2], axis=2)

    B4_s1 = np.diag([true_params["B41d_1"], true_params["B41d_2"]])
    B4_s2 = np.diag([true_params["B42d_1"], true_params["B42d_2"]])
    B4_s_diag = np.stack([B4_s1, B4_s2], axis=2)

    # --- Scenario B Parameters ---
    gamma1 = true_params["gamma1"]
    P2 = true_params["P2"]
    P12 = float(true_params.get("P12", 1e-12))

    gamma2 = true_params["gamma2"]
    gamma3 = np.array([true_params["gamma3_1"], true_params["gamma3_2"]], dtype=float)
    gamma4 = np.array([true_params["gamma4_1"], true_params["gamma4_2"]], dtype=float)

    # Measurement Model Parameters
    Q1 = np.diag([true_params["Q1d_1"], true_params["Q1d_2"]])
    Q2 = np.diag([true_params["Q2d_1"], true_params["Q2d_2"]])
    R1 = np.diag([
        true_params["R1d_1"], true_params["R1d_2"],
        true_params["R1d_3"], true_params["R1d_4"]
    ])

    # Lmd2
    Lmd2 = np.array([[true_params["Lmd2_1"]],
                     [true_params["Lmd2_2"]]], dtype=float)
    R2 = np.diag([true_params["R2d_1"], true_params["R2d_2"]])

    # Lmd1
    Lmd1 = np.zeros((O1, U1), dtype=float)
    Lmd1[0, 0] = 1
    Lmd1[1, 0] = 1
    Lmd1[2, 1] = 1
    Lmd1[3, 1] = 1

    Lmd1T = Lmd1.T
    Lmd2T = Lmd2.T

    # --- 2. Initialize Arrays ---
    y1 = np.full((N, Nt, O1), np.nan, dtype=float)
    y2 = np.full((N, O2), np.nan, dtype=float)
    eta1_true = np.full((N, Nt + 1, U1), np.nan, dtype=float)
    S_true = np.full((N, Nt + 1), np.nan, dtype=float)
    eta2_true = np.full((N, 1), np.nan, dtype=float)
    zeta2_true = np.full((N, U1), np.nan, dtype=float)

    # --- 3. Generation Process ---

    # Generate eta2 using P2 (Variance)
    eta2_true_i = np.random.normal(loc=0.0, scale=np.sqrt(P2), size=N)
    eta2_true[:, 0] = eta2_true_i

    # Generate y2
    eps2 = np.random.normal(size=(N, O2)) @ np.linalg.cholesky(R2)
    y2 = eta2_true_i.reshape(N, 1) @ Lmd2T + eps2

    # Initial State for eta1
    # Sample subject-specific random intercepts from Q2 so the simulation can
    # actually test whether alternative handling of zeta2_i improves recovery.
    zeta2_true_i = np.random.normal(size=(N, U1)) @ np.linalg.cholesky(Q2)
    zeta2_true[:, :] = zeta2_true_i

    eta1_true[:, 0, :] = np.random.normal(size=(N, U1))
    S_true[:, 0] = 1

    def sigmoid_clip(x):
        p = 1.0 / (1.0 + np.exp(-x))
        return np.minimum(np.maximum(p, epsilon), 1 - epsilon)

    # Time Loop
    for t in range(Nt):
        eta1_prev = eta1_true[:, t, :]
        S_prev = S_true[:, t]

        # State Transition Probabilities
        interaction_term = (eta1_prev @ gamma4.reshape(-1, 1)) * eta2_true_i.reshape(-1, 1)
        prob_S1_to_S1 = sigmoid_clip(
            gamma1
            + (eta2_true_i * gamma2).reshape(-1, 1)
            + (eta1_prev @ gamma3.reshape(-1, 1))
            + interaction_term
        ).flatten()

        prob_S_t = np.zeros(N, dtype=float)
        prob_S_t[S_prev == 1] = prob_S1_to_S1[S_prev == 1]
        prob_S_t[S_prev == 2] = P12

        # Sample State
        S_t = (np.random.uniform(size=N) > prob_S_t).astype(int) + 1
        S_true[:, t + 1] = S_t

        # Update Continuous State
        B1_is_t = np.full((N, U1), np.nan, dtype=float)
        B3_is_t = np.full((N, U1, U1), np.nan, dtype=float)
        eta1_t_ar_part = np.full((N, U1), np.nan, dtype=float)

        idx_s1 = np.where(S_t == 1)[0]
        idx_s2 = np.where(S_t == 2)[0]

        if len(idx_s1) > 0:
            B1_is_t[idx_s1, :] = (
                np.tile(B1_s[0, :], (len(idx_s1), 1))
                + eta2_true_i[idx_s1].reshape(-1, 1) @ B2_s[0, :].reshape(1, -1)
            )

        if len(idx_s2) > 0:
            B1_is_t[idx_s2, :] = (
                np.tile(B1_s[1, :], (len(idx_s2), 1))
                + eta2_true_i[idx_s2].reshape(-1, 1) @ B2_s[1, :].reshape(1, -1)
            )

        for i in idx_s1:
            B3_is_t[i, :, :] = B3_s_diag[:, :, 0] + (B4_s_diag[:, :, 0] * eta2_true_i[i])

        for i in idx_s2:
            B3_is_t[i, :, :] = B3_s_diag[:, :, 1] + (B4_s_diag[:, :, 1] * eta2_true_i[i])

        for i in range(N):
            eta1_t_ar_part[i, :] = eta1_prev[i, :].reshape(1, -1) @ B3_is_t[i, :, :]

        zeta1 = np.random.normal(size=(N, U1)) @ np.linalg.cholesky(Q1)
        eta1_t = B1_is_t + eta1_t_ar_part + zeta2_true_i + zeta1
        eta1_true[:, t + 1, :] = eta1_t

        # Generate Observations y1
        eps1 = np.random.normal(size=(N, O1)) @ np.linalg.cholesky(R1)
        y1[:, t, :] = eta1_t @ Lmd1T + eps1

    # No Dropout
    DO = np.zeros((N, Nt), dtype=int)

    return {
        "y1": y1,
        "y2": y2,
        "DO": DO,
        "S_true": S_true[:, 1:(Nt + 1)],
        "eta1_true": eta1_true[:, 1:(Nt + 1), :],
        "eta2_true": eta2_true,
        "zeta2_true": zeta2_true,
        "O1": O1,
        "O2": O2,
        "U1": U1,
        "N": N,
        "Nt": Nt
    }
