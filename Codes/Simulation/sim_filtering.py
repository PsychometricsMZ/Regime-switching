import gc
import math
import numpy as np
import pandas as pd
import torch
from torch.optim import Rprop
from scipy.stats import truncnorm


# ==============================================================================
# Helper utilities
# ==============================================================================

def torch_diag_from_vector(x):
    return torch.diag(x)


def safe_eye(n, device=None, dtype=None):
    return torch.eye(n, device=device, dtype=dtype)


def rtruncnorm_positive(size, mean=0.0, sd=0.5):
    a, b = (0 - mean) / sd, np.inf
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)


def clip_prob(x, eps=1e-12):
    return torch.clamp(x, min=eps, max=1.0 - eps)


def which(condition_np):
    return np.where(condition_np)[0]


# ==============================================================================
# Two-stage forward filter pass (non-augmented, U1-dimensional state only)
# zeta_offsets: (N, U1) person-level random intercept residuals estimated
# externally via _update_two_stage_offsets and passed in each outer iteration.
# ==============================================================================

def _forward_filter_pass_two_stage(
    theta,
    y1_tensor,
    y2_tensor,
    DO_matrix,
    N,
    Nt,
    O1,
    O2,
    U1,
    missing,
    non_missing,
    zeta_offsets,
    device,
    dtype,
):
    epsilon = 1e-12
    const = (2 * math.pi) ** (-O1 / 2)

    B11 = theta["B11"]
    log_B12_delta = theta["log_B12_delta"]
    B21 = theta["B21"]
    B22 = theta["B22"]
    B31d = theta["B31d"]
    B32d = theta["B32d"]
    B41d = theta["B41d"]
    B42d = theta["B42d"]

    gamma1 = theta["gamma1"]
    gamma2 = theta["gamma2"]
    gamma3 = theta["gamma3"]
    gamma4 = theta["gamma4"]

    log_Q1d = theta["log_Q1d"]
    log_R1d = theta["log_R1d"]
    log_R2d = theta["log_R2d"]

    # ------------------------------------------------------------------
    # Step 1: CFA for eta2 (Bartlett factor score)
    # ------------------------------------------------------------------
    Lmd2 = torch.ones((O2, 1), device=device, dtype=dtype)
    R2 = torch_diag_from_vector(
        torch.exp(torch.clamp(log_R2d, min=math.log(1e-6), max=math.log(1e1)))
    )
    R2_inv = torch.linalg.inv(R2)
    term1 = Lmd2.T @ R2_inv @ Lmd2
    term1_stab = term1 + 1e-6 * safe_eye(term1.shape[0], device=device, dtype=dtype)
    W_bartlett_term = torch.linalg.inv(term1_stab) @ Lmd2.T @ R2_inv
    eta2 = (y2_tensor @ W_bartlett_term.T).squeeze(-1)

    # ------------------------------------------------------------------
    # Initialization (U1-dimensional state, no augmentation)
    # ------------------------------------------------------------------
    mEta_0 = torch.zeros((N, 2, U1), device=device, dtype=dtype)
    mP_0 = safe_eye(U1, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(N, 2, 1, 1)

    mPr_pred_0 = torch.full((N, 2), float("nan"), device=device, dtype=dtype)
    mPr_filtered_0 = torch.full((N, 2), float("nan"), device=device, dtype=dtype)
    mPr_pred_0[:, 0] = 1 - epsilon
    mPr_pred_0[:, 1] = epsilon
    mPr_filtered_0[:, 0] = 1 - epsilon
    mPr_filtered_0[:, 1] = epsilon

    # ------------------------------------------------------------------
    # Measurement and process noise matrices
    # ------------------------------------------------------------------
    Lmd1 = torch.zeros((O1, U1), device=device, dtype=dtype)
    Lmd1[0, 0] = 1
    Lmd1[1, 0] = 1
    Lmd1[2, 1] = 1
    Lmd1[3, 1] = 1
    Lmd1T = Lmd1.T

    Q1 = torch_diag_from_vector(
        torch.exp(torch.clamp(log_Q1d, min=math.log(1e-6), max=math.log(1e1)))
    )
    R1 = torch_diag_from_vector(
        torch.exp(torch.clamp(log_R1d, min=math.log(1e-6), max=math.log(1e1)))
    )

    # ------------------------------------------------------------------
    # Transition matrices (person- and regime-specific)
    # zeta_offsets absorbed into B1_is as additive shift
    # ------------------------------------------------------------------
    B12_val = B11 + torch.exp(torch.clamp(log_B12_delta, min=-20))
    B1_s = torch.cat([B11, B12_val], dim=0).reshape(2, U1)
    B2_s = torch.cat([B21, B22], dim=0).reshape(2, U1)
    B3_s_diag = torch.cat(
        [torch.diag(B31d).reshape(1, U1, U1), torch.diag(B32d).reshape(1, U1, U1)],
        dim=0,
    )
    B4_s_diag = torch.cat(
        [torch.diag(B41d).reshape(1, U1, U1), torch.diag(B42d).reshape(1, U1, U1)],
        dim=0,
    )

    eta2_expanded_B1B2 = eta2.unsqueeze(-1).unsqueeze(-1)
    eta2_expanded_B3B4 = eta2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    B1_is = B1_s.unsqueeze(0) + B2_s.unsqueeze(0) * eta2_expanded_B1B2
    B1_is = B1_is + zeta_offsets.unsqueeze(1)   # add person-specific intercept offset
    B3_is = B3_s_diag.unsqueeze(0) + B4_s_diag.unsqueeze(0) * eta2_expanded_B3B4
    B3_is_T = B3_is.transpose(-1, -2)

    P12 = 0.1 * torch.sigmoid(theta["logit_P12_b"])

    # ------------------------------------------------------------------
    # History lists
    # ------------------------------------------------------------------
    mEta_list = [mEta_0]
    mP_list = [mP_0]
    mPr_filtered_list = [mPr_filtered_0]
    mPr_pred_list = [mPr_pred_0]
    mLL_list = []
    tPr_list = []   # transition probability matrices (N, S, S) per time step

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    for t in range(Nt):
        mEta_t = mEta_list[-1]
        mP_t = mP_list[-1]
        mPr_filtered_t = mPr_filtered_list[-1]

        dropout_indices_now = which(DO_matrix[:, t] == 1)
        no_dropout_indices_now = which(DO_matrix[:, t] == 0)
        missing_indices_now = which(missing[:, t] == 1)
        no_missing_indices_now = which(non_missing[:, t] == 1)

        # --- Prediction step ---
        jEta_t = (
            B1_is.unsqueeze(2)
            + torch.matmul(
                mEta_t.unsqueeze(1).unsqueeze(-2),
                B3_is.unsqueeze(2),
            ).squeeze(-2)
        )

        jP_t = (
            torch.matmul(
                torch.matmul(B3_is.unsqueeze(2), mP_t.unsqueeze(1)),
                B3_is_T.unsqueeze(2),
            )
            + Q1.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
        jP_t = jP_t + 1e-6 * safe_eye(U1, device=device, dtype=dtype)

        jEta2_t = jEta_t.clone()
        jP2_t = jP_t.clone()
        jLL_t = torch.zeros((N, 2, 2), device=device, dtype=dtype)

        # --- Update step ---
        if len(no_missing_indices_now) > 0:
            idx = torch.tensor(no_missing_indices_now, device=device, dtype=torch.long)

            jV_nm = (
                y1_tensor[idx, t, :].unsqueeze(1).unsqueeze(1)
                - torch.matmul(jEta_t[idx], Lmd1T.unsqueeze(0).unsqueeze(0))
            )

            jF_nm = (
                torch.matmul(
                    Lmd1.unsqueeze(0).unsqueeze(0),
                    torch.matmul(jP_t[idx], Lmd1T.unsqueeze(0).unsqueeze(0)),
                )
                + R1.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            jF_nm = jF_nm + 1e-6 * safe_eye(O1, device=device, dtype=dtype)

            C = torch.matmul(jP_t[idx], Lmd1T.unsqueeze(0).unsqueeze(0))
            XT = torch.linalg.solve(jF_nm.transpose(-1, -2), C.transpose(-1, -2))
            KG_nm = XT.transpose(-1, -2)

            jEta2_nm = jEta_t[idx] + torch.matmul(KG_nm, jV_nm.unsqueeze(-1)).squeeze(-1)

            I_KGLmd_nm = (
                safe_eye(U1, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                - torch.matmul(KG_nm, Lmd1.unsqueeze(0).unsqueeze(0))
            )

            jP2_nm = (
                torch.matmul(
                    torch.matmul(I_KGLmd_nm, jP_t[idx]),
                    I_KGLmd_nm.transpose(-1, -2),
                )
                + torch.matmul(
                    torch.matmul(KG_nm, R1.unsqueeze(0).unsqueeze(0).unsqueeze(0)),
                    KG_nm.transpose(-1, -2),
                )
            )

            _, logabsdet = torch.linalg.slogdet(jF_nm)
            v_unsqueezed = jV_nm.unsqueeze(-1)
            solved_x = torch.linalg.solve(jF_nm, v_unsqueezed)
            quadratic_term = -0.5 * torch.matmul(
                v_unsqueezed.transpose(-1, -2), solved_x
            ).squeeze(-1).squeeze(-1)
            jLL_nm = math.log(epsilon + const) - 0.5 * logabsdet + quadratic_term

            jEta2_t[idx] = jEta2_nm
            jP2_t[idx] = jP2_nm
            jLL_t[idx] = jLL_nm

        # --- Transition probabilities ---
        eta1_pred_t = (
            mPr_filtered_t[:, 0].unsqueeze(-1) * mEta_t[:, 0, :]
            + mPr_filtered_t[:, 1].unsqueeze(-1) * mEta_t[:, 1, :]
        )

        tPr_t = torch.full((N, 2, 2), float("nan"), device=device, dtype=dtype)
        tPr_t[:, 0, 1] = P12
        tPr_t[:, 1, 1] = 1 - P12

        if t == 0:
            interaction_term = (eta1_pred_t * gamma4).sum(dim=1) * eta2
            tPr_t[:, 0, 0] = clip_prob(
                torch.sigmoid(gamma1 + eta2 * gamma2 + eta1_pred_t @ gamma3 + interaction_term)
            )
            tPr_t[:, 1, 0] = 1 - tPr_t[:, 0, 0]
        else:
            dropout_indices_yesterday = which(DO_matrix[:, t - 1] == 1)
            no_dropout_indices_yesterday = which(DO_matrix[:, t - 1] == 0)

            if len(no_dropout_indices_yesterday) > 0:
                idx = torch.tensor(no_dropout_indices_yesterday, device=device, dtype=torch.long)
                interaction_term = (eta1_pred_t[idx] * gamma4).sum(dim=1) * eta2[idx]
                tPr_t[idx, 0, 0] = clip_prob(
                    torch.sigmoid(
                        gamma1 + eta2[idx] * gamma2 + eta1_pred_t[idx] @ gamma3 + interaction_term
                    )
                )
                tPr_t[idx, 1, 0] = 1 - tPr_t[idx, 0, 0]

            if len(dropout_indices_yesterday) > 0:
                idx = torch.tensor(dropout_indices_yesterday, device=device, dtype=torch.long)
                tPr_t[idx, 0, :] = epsilon
                tPr_t[idx, 1, :] = 1 - epsilon

        # --- Filter update ---
        jPr_t = tPr_t * mPr_filtered_t.unsqueeze(-2)
        mLL_t = torch.full((N,), float("nan"), device=device, dtype=dtype)

        if len(no_missing_indices_now) > 0:
            idx_all = torch.tensor(no_missing_indices_now, device=device, dtype=torch.long)
            log_jPr2_all = torch.log(torch.clamp(jPr_t[idx_all], min=epsilon))
            log_joint_lik2 = jLL_t[idx_all] + log_jPr2_all
            mLL_t[idx_all] = torch.logsumexp(log_joint_lik2.reshape(len(idx_all), -1), dim=1)

        jPr2_t = jPr_t.clone()
        valid_mask = np.intersect1d(no_missing_indices_now, no_dropout_indices_now)
        if len(valid_mask) > 0:
            idx = torch.tensor(valid_mask, device=device, dtype=torch.long)
            log_jPr1 = torch.log(torch.clamp(jPr_t[idx], min=epsilon))
            log_joint_lik1 = jLL_t[idx] + log_jPr1
            log_mLL1 = torch.logsumexp(log_joint_lik1.reshape(len(idx), -1), dim=1)
            jPr2_t[idx] = torch.exp(log_joint_lik1 - log_mLL1.view(-1, 1, 1))

        isect_miss = np.intersect1d(missing_indices_now, no_dropout_indices_now)
        if len(isect_miss) > 0:
            idx = torch.tensor(isect_miss, device=device, dtype=torch.long)
            jPr2_t[idx] = jPr_t[idx]

        mPr_pred_next = torch.clamp(jPr_t.sum(dim=2), min=epsilon, max=1 - epsilon)
        mPr_filtered_next = mPr_pred_next.clone()

        if len(no_dropout_indices_now) > 0:
            idx = torch.tensor(no_dropout_indices_now, device=device, dtype=torch.long)
            mPr_filtered_next[idx] = torch.clamp(jPr2_t[idx].sum(dim=2), min=epsilon, max=1 - epsilon)

        if len(dropout_indices_now) > 0:
            idx = torch.tensor(dropout_indices_now, device=device, dtype=torch.long)
            mPr_filtered_next[idx, 0] = epsilon
            mPr_filtered_next[idx, 1] = 1 - epsilon
            jPr2_t[idx, 0, :] = epsilon
            jPr2_t[idx, 1, 0] = mPr_filtered_t[idx, 0]
            jPr2_t[idx, 1, 1] = mPr_filtered_t[idx, 1]

        # --- Collapsing ---
        W_t = torch.clamp(
            jPr2_t / (mPr_filtered_next.unsqueeze(-1) + epsilon),
            min=epsilon,
            max=1 - epsilon,
        )
        mEta_next = (W_t.unsqueeze(-1) * jEta2_t).sum(dim=2)
        subEta_t = mEta_next.unsqueeze(2) - jEta2_t
        mP_next = (
            W_t.unsqueeze(-1).unsqueeze(-1)
            * (
                jP2_t
                + torch.matmul(subEta_t.unsqueeze(-1), subEta_t.unsqueeze(-2))
            )
        ).sum(dim=2)
        mP_next = mP_next + 1e-6 * safe_eye(U1, device=device, dtype=dtype)

        mEta_list.append(mEta_next)
        mP_list.append(mP_next)
        mPr_filtered_list.append(mPr_filtered_next)
        mPr_pred_list.append(mPr_pred_next)
        mLL_list.append(mLL_t)
        tPr_list.append(tPr_t.detach().clone())

    return {
        "mEta": torch.stack(mEta_list, dim=1),
        "mP": torch.stack(mP_list, dim=1),
        "mPr_filtered": torch.stack(mPr_filtered_list, dim=1),
        "mPr_pred": torch.stack(mPr_pred_list, dim=1),
        "mLL": torch.stack(mLL_list, dim=1),
        "eta2": eta2,
        "tPr": torch.stack(tPr_list, dim=1),   # (N, Nt, S, S)
    }


# ==============================================================================
# Between-level offset update (Muthen 1994 / DSEM latent centering analog)
# Estimates person-level random intercept residuals zeta_{2i} from observed
# person means without placing them in the Kalman filter state vector,
# avoiding the observability problem identified by the reviewer.
# ==============================================================================

def _update_two_stage_offsets(
    theta_best,
    regime_prob,
    eta2_scores,
    n_train,
    y1_obs,
    O1,
    U1,
    previous_offsets,
    previous_q2_diag,
    damping,
):
    """
    Estimate person-level random intercept residuals (zeta_{2i}) from
    observed person means -- Muthen (1994) between-within decomposition /
    DSEM latent centering analog.

    Algorithm (between-level stage):
      1. Compute observed person means from the raw y1 array, preserving missingness.
      2. Build the model-implied observed mean m_i(theta, 0) and Jacobian
         D_i = d m_i / d zeta_{2i}' using regime-weighted stationary means.
      3. Use the empirical-Bayes update
           zeta_hat_i = Q2 D_i' (D_i Q2 D_i' + Vbar_i)^{-1} (ybar_i - m_i(theta, 0))
         with posterior covariance
           C_i = Q2 - Q2 D_i' (D_i Q2 D_i' + Vbar_i)^{-1} D_i Q2.
      4. Update Q2 by the moment estimator
           Q2 = N^{-1} sum_i (zeta_hat_i zeta_hat_i' + C_i).
      5. Apply damping for outer-loop stability.
    """
    epsilon = 1e-6

    def to_np(name):
        return theta_best[name].detach().cpu().numpy().reshape(-1)

    B11 = to_np("B11")   # (U1,)
    B12 = B11 + np.exp(to_np("log_B12_delta"))
    B21 = to_np("B21")
    B22 = to_np("B22")
    B31d = to_np("B31d")
    B32d = to_np("B32d")
    B41d = to_np("B41d")
    B42d = to_np("B42d")
    q1_diag = np.exp(np.clip(to_np("log_Q1d"), np.log(1e-6), np.log(1e1)))
    r1_diag = np.exp(np.clip(to_np("log_R1d"), np.log(1e-6), np.log(1e1)))

    eta2_scores = np.asarray(eta2_scores, dtype=np.float32).reshape(-1)  # (N,)
    previous_q2_diag = np.asarray(previous_q2_diag, dtype=np.float32).reshape(-1)
    q2_prior_diag = np.clip(previous_q2_diag, epsilon, None)

    # Within-level measurement model used in the reduced simulation design.
    Lmd1 = np.zeros((O1, U1), dtype=np.float32)
    Lmd1[0, 0] = 1.0
    Lmd1[1, 0] = 1.0
    Lmd1[2, 1] = 1.0
    Lmd1[3, 1] = 1.0
    R1 = np.diag(r1_diag.astype(np.float32))

    def stable_state_matrices(ar_diag, q1_diag_local):
        ar_diag = np.clip(np.asarray(ar_diag, dtype=np.float32), -0.99, 0.99)
        inv_im_b = np.diag(1.0 / np.clip(1.0 - ar_diag, epsilon, None))
        stationary_var = q1_diag_local / np.clip(1.0 - ar_diag**2, epsilon, None)
        p_lat = np.diag(stationary_var.astype(np.float32))
        return inv_im_b.astype(np.float32), p_lat

    N = y1_obs.shape[0]
    zeta_hat_all = np.zeros((N, U1), dtype=np.float32)
    c_diag_all = np.zeros((N, U1), dtype=np.float32)

    y1_train = np.asarray(y1_obs[:, :n_train, :], dtype=np.float32)
    regime_prob = np.asarray(regime_prob[:, :n_train], dtype=np.float32)

    for i in range(N):
        y_i = y1_train[i]
        obs_mask = ~np.any(np.isnan(y_i), axis=1)
        if not np.any(obs_mask):
            obs_mask = np.ones(n_train, dtype=bool)

        y_obs = y_i[obs_mask]
        y_bar_i = np.nanmean(y_obs, axis=0)
        t_obs_i = max(int(obs_mask.sum()), 1)

        p2_i = regime_prob[i, obs_mask]
        p1_i = 1.0 - p2_i
        w1_i = float(np.nanmean(p1_i))
        w2_i = float(np.nanmean(p2_i))

        b1_s1_i = B11 + eta2_scores[i] * B21
        b1_s2_i = B12 + eta2_scores[i] * B22

        ar_s1_i = B31d + eta2_scores[i] * B41d
        ar_s2_i = B32d + eta2_scores[i] * B42d
        inv_s1_i, p_lat_s1_i = stable_state_matrices(ar_s1_i, q1_diag)
        inv_s2_i, p_lat_s2_i = stable_state_matrices(ar_s2_i, q1_diag)

        D1_i = Lmd1 @ inv_s1_i
        D2_i = Lmd1 @ inv_s2_i
        D_i = w1_i * D1_i + w2_i * D2_i

        m0_i = (w1_i * (D1_i @ b1_s1_i)) + (w2_i * (D2_i @ b1_s2_i))

        V1_i = (D1_i @ p_lat_s1_i @ D1_i.T) + R1
        V2_i = (D2_i @ p_lat_s2_i @ D2_i.T) + R1
        V_bar_i = ((w1_i * V1_i) + (w2_i * V2_i)) / float(t_obs_i)
        V_bar_i = V_bar_i + epsilon * np.eye(O1, dtype=np.float32)

        Q2_i = np.diag(q2_prior_diag.astype(np.float32))
        S_i = D_i @ Q2_i @ D_i.T + V_bar_i
        K_i = Q2_i @ D_i.T @ np.linalg.inv(S_i)

        resid_i = y_bar_i - m0_i
        zeta_hat_i = K_i @ resid_i
        C_i = Q2_i - K_i @ D_i @ Q2_i

        zeta_hat_all[i] = zeta_hat_i.astype(np.float32)
        c_diag_all[i] = np.clip(np.diag(C_i).astype(np.float32), epsilon, None)

    q2_diag_moment = np.nanmean((zeta_hat_all ** 2) + c_diag_all, axis=0)
    q2_diag = (damping * q2_prior_diag) + ((1.0 - damping) * q2_diag_moment)
    q2_diag = np.clip(q2_diag, epsilon, None)

    updated_offsets = (damping * previous_offsets) + ((1.0 - damping) * zeta_hat_all)

    return updated_offsets.astype(np.float32), q2_diag.astype(np.float32)


# ==============================================================================
# Two-stage filtering (profile-likelihood outer loop)
# ==============================================================================

def _filtering_two_stage(
    seed,
    N,
    Nt,
    O1,
    O2,
    U1,
    y1,
    y2,
    DO,
    init,
    maxIter,
    n_train,
    patience,
    min_delta=1e-4,
    compute_se=False,
    se_sample_size=None,
    verbose=False,
    show_progress=False,
    device="cpu",
    two_stage_outer_loops=3,
    two_stage_damping=0.5,
):
    torch.manual_seed(seed + init)
    np.random.seed(seed + init)

    dtype = torch.float32
    epsilon = 1e-12

    if compute_se and verbose:
        print("SE computation is not yet implemented for two_stage; returning SE=NaN.")

    y1 = np.array(y1, dtype=np.float32, copy=True)
    y1_raw = y1.copy()
    y2 = np.array(y2, dtype=np.float32, copy=True)
    DO = np.array(DO)

    y1_na = np.isnan(y1)
    y1[y1_na] = 0.0

    missing = np.max(y1_na, axis=2).astype(int)
    non_missing = np.min(1 - y1_na.astype(int), axis=2).astype(int)

    y1_tensor = torch.tensor(y1, dtype=dtype, device=device)
    y2_tensor = torch.tensor(y2, dtype=dtype, device=device)

    def _make_theta(theta_source=None):
        """Create or clone the parameter dict (no log_Q2d: Q2 is estimated externally)."""
        if theta_source is not None:
            return {
                k: v.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
                for k, v in theta_source.items()
            }

        B11_0 = np.random.normal(0, 0.5, U1)
        Delta_B12_0 = rtruncnorm_positive(U1, mean=0, sd=0.5)
        B21_0 = np.random.normal(0, 0.3, U1)
        Delta_B22_0 = np.random.normal(0, 0.3, U1)
        B22_0 = B21_0 + Delta_B22_0
        B31d_0 = np.random.uniform(0.1, 0.8, U1)
        B32d_0 = np.random.uniform(0.1, 0.8, U1)
        B41d_0 = np.random.normal(0, 0.1, U1)
        Delta_B42_0 = np.random.normal(0, 0.1, U1)
        B42d_0 = B41d_0 + Delta_B42_0

        init_q1d_precision = np.random.gamma(shape=9, scale=1 / 4, size=U1)
        init_q1d = 1 / (init_q1d_precision + epsilon)
        init_r1d_precision = np.random.gamma(shape=9, scale=1 / 4, size=O1)
        init_r1d = 1 / (init_r1d_precision + epsilon)
        R2d_0 = 1 / (np.random.gamma(shape=9, scale=1 / 4, size=O2) + epsilon)
        R2d_0[R2d_0 < epsilon] = epsilon

        gamma1_0 = float(truncnorm.rvs(0, np.inf, loc=0, scale=1))  # TruncN+(0,1)
        gamma2_0 = float(np.random.normal(0, 0.3))
        gamma3_0 = np.random.normal(0, 0.3, U1)
        gamma4_0 = np.random.normal(0, 0.1, U1)
        # P12 bounded in (0, 0.1): sample from Unif(0.01, 0.09), back-transform
        p12_init = float(np.random.uniform(0.01, 0.09))
        logit_p12_b_0 = float(np.log(p12_init / (0.1 - p12_init)))

        return {
            "B11":           torch.tensor(B11_0,                    dtype=dtype, device=device, requires_grad=True),
            "log_B12_delta": torch.tensor(np.log(Delta_B12_0 + epsilon), dtype=dtype, device=device, requires_grad=True),
            "B21":           torch.tensor(B21_0,                    dtype=dtype, device=device, requires_grad=True),
            "B22":           torch.tensor(B22_0,                    dtype=dtype, device=device, requires_grad=True),
            "B31d":          torch.tensor(B31d_0,                   dtype=dtype, device=device, requires_grad=True),
            "B32d":          torch.tensor(B32d_0,                   dtype=dtype, device=device, requires_grad=True),
            "B41d":          torch.tensor(B41d_0,                   dtype=dtype, device=device, requires_grad=True),
            "B42d":          torch.tensor(B42d_0,                   dtype=dtype, device=device, requires_grad=True),
            "log_Q1d":       torch.tensor(np.log(init_q1d),         dtype=dtype, device=device, requires_grad=True),
            "log_R1d":       torch.tensor(np.log(init_r1d),         dtype=dtype, device=device, requires_grad=True),
            "log_R2d":       torch.tensor(np.log(R2d_0),            dtype=dtype, device=device, requires_grad=True),
            "gamma1":        torch.tensor([gamma1_0],               dtype=dtype, device=device, requires_grad=True),
            "gamma2":        torch.tensor(gamma2_0,                 dtype=dtype, device=device, requires_grad=True),
            "gamma3":        torch.tensor(gamma3_0,                 dtype=dtype, device=device, requires_grad=True),
            "gamma4":        torch.tensor(gamma4_0,                 dtype=dtype, device=device, requires_grad=True),
            "logit_P12_b":   torch.tensor([logit_p12_b_0],         dtype=dtype, device=device, requires_grad=True),
        }

    zeta_offsets_np = np.zeros((N, U1), dtype=np.float32)
    q2_diag_np = np.ones(U1, dtype=np.float32)
    q2_diag_best = np.full(U1, np.nan, dtype=np.float32)
    theta_start = None
    overall_best = None
    overall_best_ll = -np.inf
    overall_best_traj = None

    for outer_iter in range(two_stage_outer_loops):
        theta = _make_theta(theta_start)
        optimizer = Rprop(list(theta.values()), lr=0.1, etas=(0.5, 1.2))

        sumLL_best = -np.inf
        patience_counter = 0
        theta_best = None
        zeta_offsets_tensor = torch.tensor(zeta_offsets_np, dtype=dtype, device=device)
        iteration = 1

        while iteration <= maxIter:
            if iteration <= 30:
                optimizer.param_groups[0]["lr"] = 0.005
            elif iteration == 31:
                optimizer.param_groups[0]["lr"] = 0.01

            out = _forward_filter_pass_two_stage(
                theta=theta,
                y1_tensor=y1_tensor,
                y2_tensor=y2_tensor,
                DO_matrix=DO,
                N=N,
                Nt=Nt,
                O1=O1,
                O2=O2,
                U1=U1,
                missing=missing,
                non_missing=non_missing,
                zeta_offsets=zeta_offsets_tensor,
                device=device,
                dtype=dtype,
            )

            loss = -torch.nanmean(out["mLL"][:, :n_train])
            sumLL = -loss.detach().cpu().item()

            if not np.isfinite(sumLL):
                if verbose:
                    print("Warning: infinite/NaN loss in two_stage filtering")
                return None

            if sumLL > sumLL_best + min_delta:
                sumLL_best = sumLL
                theta_best = {k: v.detach().clone() for k, v in theta.items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(theta.values()), max_norm=5.0)
            optimizer.step()

            iteration += 1
            gc.collect()

        if theta_best is None:
            return None

        # Evaluate best theta with current offsets
        with torch.no_grad():
            out_best = _forward_filter_pass_two_stage(
                theta=theta_best,
                y1_tensor=y1_tensor,
                y2_tensor=y2_tensor,
                DO_matrix=DO,
                N=N,
                Nt=Nt,
                O1=O1,
                O2=O2,
                U1=U1,
                missing=missing,
                non_missing=non_missing,
                zeta_offsets=torch.tensor(zeta_offsets_np, dtype=dtype, device=device),
                device=device,
                dtype=dtype,
            )

        mEta_best = out_best["mEta"]
        mPr_filtered_best = out_best["mPr_filtered"]

        prob_regime1 = mPr_filtered_best[:, 1:(Nt + 1), 0].clone()
        prob_regime2 = mPr_filtered_best[:, 1:(Nt + 1), 1].clone()
        eta_regime1 = mEta_best[:, 1:(Nt + 1), 0, :].clone()
        eta_regime2 = mEta_best[:, 1:(Nt + 1), 1, :].clone()

        estimated_eta = prob_regime1.unsqueeze(-1) * eta_regime1 + prob_regime2.unsqueeze(-1) * eta_regime2
        estimated_regime_prob = prob_regime2

        traj_results = {
            "eta_est": estimated_eta.cpu().numpy(),
            "regime_prob": estimated_regime_prob.cpu().numpy(),
        }

        eta2_scores = out_best["eta2"].cpu().numpy()
        previous_offsets = zeta_offsets_np.copy()
        zeta_offsets_np, q2_diag_now = _update_two_stage_offsets(
            theta_best=theta_best,
            regime_prob=traj_results["regime_prob"],
            eta2_scores=eta2_scores,
            n_train=n_train,
            y1_obs=y1_raw,
            O1=O1,
            U1=U1,
            previous_offsets=previous_offsets,
            previous_q2_diag=q2_diag_np,
            damping=two_stage_damping,
        )
        q2_diag_np = q2_diag_now.copy()
        offset_shift = np.max(np.abs(zeta_offsets_np - previous_offsets))

        theta_start = theta_best
        if sumLL_best > overall_best_ll:
            overall_best_ll = sumLL_best
            overall_best = theta_best
            overall_best_traj = traj_results
            q2_diag_best = q2_diag_now.copy()

        if offset_shift < 1e-4 and outer_iter > 0:
            break

    if overall_best is None:
        return None

    # ------------------------------------------------------------------
    # P2 estimate (empirical variance of Bartlett factor scores)
    # ------------------------------------------------------------------
    R2_best = torch.diag(torch.exp(torch.clamp(overall_best["log_R2d"], min=math.log(1e-6), max=math.log(1e1))))
    Lmd2_fixed = torch.ones((O2, 1), device=device, dtype=dtype)
    term1_best = Lmd2_fixed.T @ torch.linalg.inv(R2_best) @ Lmd2_fixed
    W_bartlett_best = (
        torch.linalg.inv(term1_best + 1e-6 * torch.eye(1, device=device, dtype=dtype))
        @ Lmd2_fixed.T
        @ torch.linalg.inv(R2_best)
    )
    P2_estimated_val = torch.var((y2_tensor @ W_bartlett_best.T).squeeze(-1)).detach().cpu().item()

    # ------------------------------------------------------------------
    # Build final estimates dataframe
    # ------------------------------------------------------------------
    param_names = list(overall_best.keys())
    est_vec = []
    grad_param_names = []
    for nm in param_names:
        vals = overall_best[nm].detach().cpu().numpy().reshape(-1)
        est_vec.extend(vals.tolist())
        if len(vals) > 1:
            grad_param_names.extend([f"{nm}_{i+1}" for i in range(len(vals))])
        else:
            grad_param_names.append(nm)

    final_estimates_df = pd.DataFrame({
        "Parameter": grad_param_names,
        "Estimate": est_vec,
        "SE": np.nan,
    })

    # Append derived B12
    B11_val = overall_best["B11"].detach().cpu().numpy().reshape(-1)
    log_delta_val = overall_best["log_B12_delta"].detach().cpu().numpy().reshape(-1)
    B12_est = B11_val + np.exp(log_delta_val)
    B12_names = [f"B12_{i+1}" for i in range(len(B11_val))] if len(B11_val) > 1 else ["B12"]
    B12_df = pd.DataFrame({"Parameter": B12_names, "Estimate": B12_est, "SE": np.nan})
    final_estimates_df = pd.concat([final_estimates_df, B12_df], ignore_index=True)

    # Back-transform log-scale variance parameters
    rows_to_transform = final_estimates_df["Parameter"].str.contains(r"^log_[QR]", regex=True)
    for i in final_estimates_df.index[rows_to_transform]:
        est_log = final_estimates_df.at[i, "Estimate"]
        final_estimates_df.at[i, "Estimate"] = np.exp(est_log)
        final_estimates_df.at[i, "Parameter"] = final_estimates_df.at[i, "Parameter"].replace("log_", "", 1)

    final_estimates_df = final_estimates_df[
        ~final_estimates_df["Parameter"].str.contains(r"^log_B12_delta", regex=True)
    ].reset_index(drop=True)

    # Back-transform logit_P12_b → P12 = 0.1 * sigmoid(logit_P12_b)
    mask_p12 = final_estimates_df["Parameter"] == "logit_P12_b"
    if mask_p12.any():
        logit_val = final_estimates_df.loc[mask_p12, "Estimate"].values[0]
        final_estimates_df.loc[mask_p12, "Estimate"] = 0.1 * float(torch.sigmoid(torch.tensor(logit_val)).item())
        final_estimates_df.loc[mask_p12, "Parameter"] = "P12"

    # Insert Q2 diagonal from the between-level moment estimator
    for dim_idx, q2_est in enumerate(q2_diag_best, start=1):
        new_row = pd.DataFrame({
            "Parameter": [f"Q2d_{dim_idx}"],
            "Estimate": [float(q2_est)],
            "SE": [np.nan],
        })
        final_estimates_df = pd.concat([final_estimates_df, new_row], ignore_index=True)

    return {
        "sumLL_best": overall_best_ll,
        "theta_best": overall_best,
        "P2_estimated": P2_estimated_val,
        "final_estimates": final_estimates_df,
        "trajectories": overall_best_traj,
        "Q2_estimated_diag": q2_diag_best,
        "zeta2_estimated": zeta_offsets_np,
        "method": "two_stage",
    }


# ==============================================================================
# Public entry point
# ==============================================================================

def filtering(
    seed,
    N,
    Nt,
    O1,
    O2,
    U1,
    y1,
    y2,
    DO,
    init,
    maxIter,
    n_train,
    patience,
    min_delta=1e-4,
    compute_se=False,
    se_sample_size=None,
    verbose=False,
    show_progress=False,
    device="cpu",
    method="two_stage",
    two_stage_outer_loops=3,
    two_stage_damping=0.5,
):
    if method != "two_stage":
        raise ValueError(
            f"Unknown filtering method: '{method}'. Only 'two_stage' is supported."
        )
    return _filtering_two_stage(
        seed=seed,
        N=N,
        Nt=Nt,
        O1=O1,
        O2=O2,
        U1=U1,
        y1=y1,
        y2=y2,
        DO=DO,
        init=init,
        maxIter=maxIter,
        n_train=n_train,
        patience=patience,
        min_delta=min_delta,
        compute_se=compute_se,
        se_sample_size=se_sample_size,
        verbose=verbose,
        show_progress=show_progress,
        device=device,
        two_stage_outer_loops=two_stage_outer_loops,
        two_stage_damping=two_stage_damping,
    )
