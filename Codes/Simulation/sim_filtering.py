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

GAMMA1_FIXED = 4.5951      # logit(0.99) = log(99); fixed for identification
                            # Enforces P(regime 1 at t=0) ≈ 0.99 (Rubicon prior)
B3_CLIP      = 0.95        # stationarity clip for the AR diagonal in the model-implied
                            # initial mean/covariance (Assumption 4: |(B3is)_jj| < 1); keeps
                            # (I-B3)^{-1} finite near the unit-root boundary during optimisation.


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
    fix_gamma3=False,
    fix_gamma4=False,
    fix_gamma1=False,
    fix_p12=False,
    p12_fixed_value=1e-12,
    gamma3_fixed_value=None,
    gamma4_fixed_value=None,
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

    gamma1 = (torch.tensor([GAMMA1_FIXED], dtype=dtype, device=device)
              if fix_gamma1
              else theta["gamma1"])
    gamma2 = theta["gamma2"]
    if fix_gamma3:
        _g3 = gamma3_fixed_value if gamma3_fixed_value is not None else np.zeros(U1)
        gamma3 = torch.tensor(_g3, dtype=dtype, device=device)
    else:
        gamma3 = theta.get("gamma3", torch.zeros(U1, dtype=dtype, device=device))
    if fix_gamma4:
        _g4 = gamma4_fixed_value if gamma4_fixed_value is not None else np.zeros(U1)
        gamma4 = torch.tensor(_g4, dtype=dtype, device=device)
    else:
        gamma4 = theta.get("gamma4", torch.zeros(U1, dtype=dtype, device=device))

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
    # mEta_0 (stationary initial mean) and mP_0 (stationary initial covariance) are the
    # model-implied initialisation; both are computed below, after B1_is, Q1 and B3_is
    # are defined (the AR diagonal is clipped to the stationary region there).

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

    # Model-implied stationary initialisation (diagonal B3, Q1; Harvey 1990; Du Toit & Browne 2007).
    # The AR diagonal is clipped to the stationary region |b_jj| <= B3_CLIP (Assumption 4) so that
    # both the implied mean and covariance stay finite near the unit-root boundary; clipping affects
    # only the initial-condition prior, not the AR dynamics.
    _b0      = torch.diagonal(B3_is, dim1=-2, dim2=-1)          # (N, 2, U1)  AR diagonal
    _b0_clip = torch.clamp(_b0, min=-B3_CLIP, max=B3_CLIP)
    _q0      = torch.diagonal(Q1).unsqueeze(0).unsqueeze(0)     # (1, 1, U1)
    # stationary mean:  mu_jj = b_{1,jj} / (1 - b_jj)
    mEta_0 = B1_is / (1.0 - _b0_clip)                           # (N, 2, U1)
    # stationary covariance:  Sigma_jj = Q1_jj / (1 - b_jj^2)
    _var0 = _q0 / (1.0 - _b0_clip**2)                           # bounded since |b_clip| <= B3_CLIP < 1
    mP_0 = torch.diag_embed(_var0).clone()

    P12 = (torch.tensor(p12_fixed_value, dtype=dtype, device=device)
          if fix_p12
          else torch.sigmoid(theta["logit_P12_b"]))

    # ------ regime-1 stationary mean for centering --------------------------
    # mu1i1 = (I - B3_i1)^{-1} B1_i1  (N, U1)
    # Since the transition P(S_t=1|S_{t-1}=1) conditions on regime 1, we use
    # the regime-1 conditional filtered estimate centered at its stationary mean.
    # regime-1 stationary mean: mu1i1_j = b_{1,0,j} / (1 - b_{0,jj}) (B3 diagonal), with the AR
    # diagonal clipped to the stationary region (Assumption 4) so it stays finite near unit root.
    mu1i1 = B1_is[:, 0, :] / (1.0 - _b0_clip[:, 0, :])   # (N, U1)

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
        # Use regime-1 conditional estimate centered at regime-1 stationary mean.
        eta1_pred_t = mEta_t[:, 0, :] - mu1i1   # (N, U1)

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
# OPG standard errors for two-stage estimator
# ==============================================================================

def _compute_opg_se_two_stage(
    theta_best, y1_tensor, y2_tensor, DO_matrix,
    N, Nt, O1, O2, U1, missing, non_missing,
    zeta_offsets_tensor, n_train,
    se_sample_size, device, dtype,
    fix_gamma3=False, fix_gamma4=False,
    fix_gamma1=False, fix_p12=False,
    p12_fixed_value=1e-12,
    gamma3_fixed_value=None, gamma4_fixed_value=None,
):
    """
    OPG approximation of SE for the two-stage estimator.
    Only free parameters (requires_grad=True in theta_best) are included.
    g_i = d/dtheta sum_t log f(y_{1it} | D_{1:t-1})
    OPG = sum_i g_i g_i'
    SE  = sqrt(diag(OPG^{-1}))
    """
    # Rebuild free-parameter dict from theta_best.
    # theta_best values are detached (requires_grad=False), so we cannot rely on
    # v.requires_grad to identify free params.  Instead, exclude keys that are
    # structurally fixed: 'gamma1' when fix_gamma1=True.
    # gamma3/gamma4/logit_P12_b are absent from theta_best when fixed, so no
    # special handling needed for those.
    fixed_keys = set()
    if fix_gamma1:
        fixed_keys.add("gamma1")
    theta_g = {k: v.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
               for k, v in theta_best.items()
               if k not in fixed_keys}

    param_names_flat = []
    for k, v in theta_g.items():
        n = v.numel()
        if n > 1:
            param_names_flat.extend([f"{k}_{j+1}" for j in range(n)])
        else:
            param_names_flat.append(k)
    n_params = len(param_names_flat)

    if se_sample_size is not None and se_sample_size < N:
        idx_sample = np.random.choice(N, size=se_sample_size, replace=False)
        scale = N / se_sample_size
    else:
        idx_sample = np.arange(N)
        scale = 1.0

    OPG = torch.zeros((n_params, n_params), device=device, dtype=dtype)
    n_skipped = 0

    for i in idx_sample:
        out_i = _forward_filter_pass_two_stage(
            theta=theta_g,
            y1_tensor=y1_tensor[i:i+1], y2_tensor=y2_tensor[i:i+1],
            DO_matrix=DO_matrix[i:i+1],
            N=1, Nt=Nt, O1=O1, O2=O2, U1=U1,
            missing=missing[i:i+1], non_missing=non_missing[i:i+1],
            zeta_offsets=zeta_offsets_tensor[i:i+1],
            device=device, dtype=dtype,
            fix_gamma3=fix_gamma3, fix_gamma4=fix_gamma4,
            fix_gamma1=fix_gamma1, fix_p12=fix_p12,
            p12_fixed_value=p12_fixed_value,
            gamma3_fixed_value=gamma3_fixed_value,
            gamma4_fixed_value=gamma4_fixed_value,
        )
        ll_row = out_i["mLL"][0, :n_train]
        valid_mask = torch.isfinite(ll_row)
        if not valid_mask.any():
            n_skipped += 1
            continue
        ll_i = ll_row[valid_mask].sum()
        grads = torch.autograd.grad(ll_i, list(theta_g.values()),
                                    retain_graph=False, allow_unused=True)
        g_flat = torch.cat([
            (g.reshape(-1) if g is not None else torch.zeros_like(v).reshape(-1))
            for g, v in zip(grads, theta_g.values())
        ])
        if not torch.isfinite(g_flat).all():
            n_skipped += 1
            continue
        OPG += torch.outer(g_flat, g_flat)
        theta_g = {k: v.detach().clone().requires_grad_(True) for k, v in theta_g.items()}

    if n_skipped > 0:
        print(f"[OPG] Skipped {n_skipped}/{len(idx_sample)} individuals with non-finite gradients")

    OPG *= scale

    # Diagonal scaling + ridge regularization
    diag_J = torch.diag(OPG)
    eps_diag = 1e-12
    scale_factors = torch.where(
        diag_J > eps_diag,
        1.0 / torch.sqrt(diag_J.clamp(min=eps_diag)),
        torch.zeros_like(diag_J),
    )
    S = torch.diag(scale_factors)
    OPG_scaled = S @ OPG @ S + torch.eye(n_params, device=device, dtype=dtype) * 1e-6

    try:
        cov_scaled = torch.linalg.inv(OPG_scaled)
    except Exception:
        print("[OPG] inv() failed on scaled matrix, using pinv()")
        cov_scaled = torch.linalg.pinv(OPG_scaled, rcond=1e-6)

    cov_final = S @ cov_scaled @ S
    diag_cov = torch.diag(cov_final)
    if not torch.isfinite(diag_cov).all():
        print("[OPG] Non-finite diagonal in final covariance, SE set to NaN")
        se_vec = torch.full((n_params,), float("nan"), device=device, dtype=dtype)
    else:
        se_vec = torch.sqrt(torch.clamp(diag_cov, min=0.0))

    return se_vec.detach().cpu().numpy(), param_names_flat


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
    fix_gamma3=False,
    fix_gamma4=False,
    fix_gamma1=False,
    fix_p12=False,
    p12_fixed_value=1e-12,
    gamma3_fixed_value=None,
    gamma4_fixed_value=None,
    sim_prior=None,
):
    torch.manual_seed(seed + init)
    np.random.seed(seed + init)

    dtype = torch.float32
    epsilon = 1e-12

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

        if sim_prior is not None:
            # Kelava warm start: sample from Normal(mean, sd) using prior CSV
            def _sp(key):  # returns (mean_arr, sd_arr)
                return sim_prior[key]['mean'], sim_prior[key]['sd']

            m, s = _sp('gamma1')
            gamma1_0 = float(truncnorm.rvs(0, np.inf, loc=m, scale=s))
            m, s = _sp('gamma2')
            gamma2_0 = float(np.random.normal(m, s))
            gamma3_0 = np.random.normal(*_sp('gamma3'))
            gamma4_0 = np.random.normal(*_sp('gamma4'))
            B11_0 = np.random.normal(*_sp('B11'))
            Delta_B12_raw = np.random.normal(*_sp('log_B12_delta'))
            B21_0 = np.random.normal(*_sp('B21'))
            B22_0 = np.random.normal(*_sp('B22'))
            B31d_0 = np.random.normal(*_sp('B31d'))
            B32d_0 = np.random.normal(*_sp('B32d'))
            B41d_0 = np.random.normal(*_sp('B41d'))
            B42d_0 = np.random.normal(*_sp('B42d'))
            log_Q1d_0 = np.random.normal(*_sp('log_Q1d'))
            log_R1d_0 = np.random.normal(*_sp('log_R1d'))
            log_R2d_0 = np.random.normal(*_sp('log_R2d'))
            # P12 free: sample from Unif(0.01, 0.10), back-transform (unchanged)
            p12_init = float(np.random.uniform(0.01, 0.10))
            logit_p12_b_0 = float(np.log(p12_init / (1.0 - p12_init)))
        else:
            B11_0 = np.random.normal(0, 0.5, U1)
            Delta_B12_raw = np.log(rtruncnorm_positive(U1, mean=0, sd=0.5) + epsilon)
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
            log_Q1d_0 = np.log(init_q1d)
            init_r1d_precision = np.random.gamma(shape=9, scale=1 / 4, size=O1)
            init_r1d = 1 / (init_r1d_precision + epsilon)
            log_R1d_0 = np.log(init_r1d)
            R2d_0 = 1 / (np.random.gamma(shape=9, scale=1 / 4, size=O2) + epsilon)
            R2d_0[R2d_0 < epsilon] = epsilon
            log_R2d_0 = np.log(R2d_0)
            gamma1_0 = float(truncnorm.rvs(0, np.inf, loc=0, scale=1))  # TruncN+(0,1)
            gamma2_0 = float(np.random.normal(0, 0.3))
            gamma3_0 = np.random.normal(0, 0.3, U1)
            gamma4_0 = np.random.normal(0, 0.1, U1)
            # P12 free in (0, 1): sample from Unif(0.01, 0.10), back-transform
            p12_init = float(np.random.uniform(0.01, 0.10))
            logit_p12_b_0 = float(np.log(p12_init / (1.0 - p12_init)))

        d = {
            "gamma1":        torch.tensor([GAMMA1_FIXED if fix_gamma1 else gamma1_0], dtype=dtype, device=device, requires_grad=(not fix_gamma1)),
            "B11":           torch.tensor(B11_0,             dtype=dtype, device=device, requires_grad=True),
            "log_B12_delta": torch.tensor(Delta_B12_raw,     dtype=dtype, device=device, requires_grad=True),
            "B21":           torch.tensor(B21_0,             dtype=dtype, device=device, requires_grad=True),
            "B22":           torch.tensor(B22_0,             dtype=dtype, device=device, requires_grad=True),
            "B31d":          torch.tensor(B31d_0,            dtype=dtype, device=device, requires_grad=True),
            "B32d":          torch.tensor(B32d_0,            dtype=dtype, device=device, requires_grad=True),
            "B41d":          torch.tensor(B41d_0,            dtype=dtype, device=device, requires_grad=True),
            "B42d":          torch.tensor(B42d_0,            dtype=dtype, device=device, requires_grad=True),
            "log_Q1d":       torch.tensor(log_Q1d_0,         dtype=dtype, device=device, requires_grad=True),
            "log_R1d":       torch.tensor(log_R1d_0,         dtype=dtype, device=device, requires_grad=True),
            "log_R2d":       torch.tensor(log_R2d_0,         dtype=dtype, device=device, requires_grad=True),
            "gamma2":        torch.tensor(gamma2_0,          dtype=dtype, device=device, requires_grad=True),
        }
        if not fix_gamma3:
            d["gamma3"] = torch.tensor(gamma3_0, dtype=dtype, device=device, requires_grad=True)
        if not fix_gamma4:
            d["gamma4"] = torch.tensor(gamma4_0, dtype=dtype, device=device, requires_grad=True)
        if not fix_p12:
            d["logit_P12_b"] = torch.tensor([logit_p12_b_0], dtype=dtype, device=device, requires_grad=True)
        return d

    zeta_offsets_np = np.zeros((N, U1), dtype=np.float32)
    q2_diag_np = np.ones(U1, dtype=np.float32)
    q2_diag_best = np.full(U1, np.nan, dtype=np.float32)
    theta_start = None
    overall_best = None
    overall_best_ll  = -np.inf
    overall_best_traj = None
    no_improve_count = 0       # consecutive outer loops without improvement

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
                fix_gamma3=fix_gamma3,
                fix_gamma4=fix_gamma4,
                fix_gamma1=fix_gamma1,
                fix_p12=fix_p12,
                p12_fixed_value=p12_fixed_value,
                gamma3_fixed_value=gamma3_fixed_value,
                gamma4_fixed_value=gamma4_fixed_value,
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
                fix_gamma3=fix_gamma3,
                fix_gamma4=fix_gamma4,
                fix_gamma1=fix_gamma1,
                fix_p12=fix_p12,
                p12_fixed_value=p12_fixed_value,
                gamma3_fixed_value=gamma3_fixed_value,
                gamma4_fixed_value=gamma4_fixed_value,
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
        if sumLL_best > overall_best_ll + 1e-6:
            overall_best_ll = sumLL_best
            overall_best = theta_best
            overall_best_traj = traj_results
            q2_diag_best = q2_diag_now.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Convergence: stop after 3 consecutive outer loops without improvement
        if no_improve_count >= 3:
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

    # Back-transform logit_P12_b → P12 = sigmoid(logit_P12_b)
    mask_p12 = final_estimates_df["Parameter"] == "logit_P12_b"
    if mask_p12.any():
        logit_val = final_estimates_df.loc[mask_p12, "Estimate"].values[0]
        final_estimates_df.loc[mask_p12, "Estimate"] = float(torch.sigmoid(torch.tensor(logit_val)).item())
        final_estimates_df.loc[mask_p12, "Parameter"] = "P12"

    # Add fixed parameters as constant rows
    if fix_gamma3:
        _g3_vals = gamma3_fixed_value if gamma3_fixed_value is not None else np.zeros(U1)
        for k in range(1, U1 + 1):
            final_estimates_df = pd.concat([final_estimates_df,
                pd.DataFrame({"Parameter": [f"gamma3_{k}"], "Estimate": [float(_g3_vals[k-1])], "SE": [np.nan]})],
                ignore_index=True)
    if fix_gamma4:
        _g4_vals = gamma4_fixed_value if gamma4_fixed_value is not None else np.zeros(U1)
        for k in range(1, U1 + 1):
            final_estimates_df = pd.concat([final_estimates_df,
                pd.DataFrame({"Parameter": [f"gamma4_{k}"], "Estimate": [float(_g4_vals[k-1])], "SE": [np.nan]})],
                ignore_index=True)
    if fix_p12:
        final_estimates_df = pd.concat([final_estimates_df,
            pd.DataFrame({"Parameter": ["P12"], "Estimate": [p12_fixed_value], "SE": [np.nan]})],
            ignore_index=True)
    # Insert Q2 diagonal from the between-level moment estimator
    for dim_idx, q2_est in enumerate(q2_diag_best, start=1):
        new_row = pd.DataFrame({
            "Parameter": [f"Q2d_{dim_idx}"],
            "Estimate": [float(q2_est)],
            "SE": [np.nan],
        })
        final_estimates_df = pd.concat([final_estimates_df, new_row], ignore_index=True)

    # ------------------------------------------------------------------
    # OPG SE (free parameters only; log-scale params mapped to output names)
    # ------------------------------------------------------------------
    if compute_se and overall_best is not None:
        zeta_t = torch.tensor(zeta_offsets_np, dtype=dtype, device=device)
        se_np, se_names = _compute_opg_se_two_stage(
            theta_best=overall_best,
            y1_tensor=y1_tensor, y2_tensor=y2_tensor, DO_matrix=DO,
            N=N, Nt=Nt, O1=O1, O2=O2, U1=U1,
            missing=missing, non_missing=non_missing,
            zeta_offsets_tensor=zeta_t,
            n_train=n_train, se_sample_size=se_sample_size,
            device=device, dtype=dtype,
            fix_gamma3=fix_gamma3, fix_gamma4=fix_gamma4,
            fix_gamma1=fix_gamma1, fix_p12=fix_p12,
            p12_fixed_value=p12_fixed_value,
            gamma3_fixed_value=gamma3_fixed_value,
            gamma4_fixed_value=gamma4_fixed_value,
        )
        se_map = dict(zip(se_names, se_np))
        for param_raw, se_val in se_map.items():
            out_name = param_raw
            for pfx, sfx in [("log_Q1d", "Q1d"), ("log_R1d", "R1d"), ("log_R2d", "R2d")]:
                if out_name.startswith(pfx):
                    out_name = out_name.replace(pfx, sfx, 1)
            if out_name.startswith("log_B12_delta"):
                continue  # B12 is derived; delta method needed, skip for now
            mask = final_estimates_df["Parameter"] == out_name
            if mask.any():
                final_estimates_df.loc[mask, "SE"] = float(se_val)

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
    fix_gamma3=False,
    fix_gamma4=False,
    fix_gamma1=False,
    fix_p12=False,
    p12_fixed_value=1e-12,
    gamma3_fixed_value=None,
    gamma4_fixed_value=None,
    sim_prior=None,
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
        fix_gamma1=fix_gamma1,
        fix_gamma3=fix_gamma3,
        fix_gamma4=fix_gamma4,
        fix_p12=fix_p12,
        p12_fixed_value=p12_fixed_value,
        gamma3_fixed_value=gamma3_fixed_value,
        gamma4_fixed_value=gamma4_fixed_value,
        sim_prior=sim_prior,
    )
