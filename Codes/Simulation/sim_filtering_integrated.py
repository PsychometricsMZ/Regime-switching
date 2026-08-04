"""
Integrated (marginalized) estimator: zeta_{2i} carried in the state vector
with its proper N(0, Q2) prior as initial condition, Q2 estimated by ML.

Motivation
----------
The two-stage estimator (sim_filtering.py) estimates zeta_{2i} and Q2 outside
the Kim filter from finite-T person means. Person means are not orthogonal to
a persistent within-person process (contamination scales with 1/(1 - B3)), so
the between-person level competes with autoregressive persistence sequentially
rather than jointly, producing the persistence/heterogeneity confound seen in
the simulation study (B3 biased up, B2/Q2 collapsed).

Here the augmented state is x = [eta1; zeta2] with transition
    x_t = [B1_is; 0] + [[B3_is, I], [0, I]] x_{t-1} + [zeta1; 0],
measurement y = [Lmd1, 0] x + eps, and the zeta2 block initialized at its
proper prior N(0, Q2) with zero process noise. The Kalman filter then
marginalizes zeta_{2i} exactly (conditional on the regime path; the Kim
collapsing approximation still applies), so the likelihood weighs the
autoregressive and between-person explanations of stable individual levels
jointly. Q2 enters the likelihood through the initial covariance and is
estimated by ML together with all other parameters (autodiff handles the
gradient through the filter).

The initial state distribution is the joint stationary distribution of
(eta1, zeta2) under regime 1:
    E[x_0]   = [A1 b_i1; 0],                    A_s = (I - B3_is)^{-1}
    Var[x_0] = [[S_s + A_s Q2 A_s, A_s Q2],
                [Q2 A_s,           Q2     ]],   S_s = Q1 / (1 - b^2)
(all blocks diagonal under Assumption 4). The regime-1 stationary mean used
for centering (Assumption 6) becomes mu1i1_t = A1 (b_i1 + zetahat_t), where
zetahat_t is the regime-1 conditional filtered mean of the zeta2 block.

Usage: SIM_INTEGRATED=1 in the environment (see sim_main.py); results go to
output_integrated*/ and never touch the two-stage outputs.
"""

import gc
import math
import os
import numpy as np
import pandas as pd
import torch
from torch.optim import Rprop
from scipy.stats import truncnorm

# --- Assumption 6 switch ----------------------------------------------------
# CENTER_MS_PREDICTOR=0 disables the person-mean centering of the transition
# predictor (Assumption 6): the raw regime-1 conditional filtered mean of eta1
# enters the sigmoid, matching the (uncentered) data-generating process in
# sim_data_generation.py and the Kelava (2022) parameterization.
# Default (unset or 1) = centered, identical to all previous runs.
CENTER_MS_PREDICTOR = os.environ.get("CENTER_MS_PREDICTOR", "1") not in ("0", "false", "False")
if not CENTER_MS_PREDICTOR:
    print("[sim_filtering_integrated] CENTER_MS_PREDICTOR=0: "
          "transition predictor NOT centered (Assumption 6 disabled)")

from sim_filtering import (
    GAMMA1_FIXED,
    B3_CLIP,
    torch_diag_from_vector,
    safe_eye,
    rtruncnorm_positive,
    clip_prob,
    which,
)


# ==============================================================================
# Forward filter pass with augmented state [eta1; zeta2]
# ==============================================================================

def _forward_filter_pass_integrated(
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
    D = 2 * U1   # augmented state dimension

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
    log_Q2d = theta["log_Q2d"]

    # ------------------------------------------------------------------
    # CFA for eta2 (Bartlett factor score) -- unchanged
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
    # Regime probabilities at t=0
    # ------------------------------------------------------------------
    mPr_pred_0 = torch.full((N, 2), float("nan"), device=device, dtype=dtype)
    mPr_filtered_0 = torch.full((N, 2), float("nan"), device=device, dtype=dtype)
    mPr_pred_0[:, 0] = 1 - epsilon
    mPr_pred_0[:, 1] = epsilon
    mPr_filtered_0[:, 0] = 1 - epsilon
    mPr_filtered_0[:, 1] = epsilon

    # ------------------------------------------------------------------
    # Measurement and noise matrices (augmented)
    # ------------------------------------------------------------------
    Lmd1 = torch.zeros((O1, U1), device=device, dtype=dtype)
    Lmd1[0, 0] = 1
    Lmd1[1, 0] = 1
    Lmd1[2, 1] = 1
    Lmd1[3, 1] = 1
    Lmd_aug = torch.cat(
        [Lmd1, torch.zeros((O1, U1), device=device, dtype=dtype)], dim=1
    )  # (O1, D)
    Lmd_augT = Lmd_aug.T

    q1_vec = torch.exp(torch.clamp(log_Q1d, min=math.log(1e-6), max=math.log(1e1)))
    Q1 = torch_diag_from_vector(q1_vec)
    R1 = torch_diag_from_vector(
        torch.exp(torch.clamp(log_R1d, min=math.log(1e-6), max=math.log(1e1)))
    )
    q2_vec = torch.exp(torch.clamp(log_Q2d, min=math.log(1e-6), max=math.log(1e1)))

    # Q_aug = blkdiag(Q1, 0): zeta2 is time-invariant (zero process noise)
    Q_aug = torch.zeros((D, D), device=device, dtype=dtype)
    Q_aug[:U1, :U1] = Q1

    # ------------------------------------------------------------------
    # Person- and regime-specific transition structure
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

    # NOTE: no zeta offsets here -- zeta2 lives in the state
    B1_is = B1_s.unsqueeze(0) + B2_s.unsqueeze(0) * eta2_expanded_B1B2      # (N, 2, U1)
    B3_is = B3_s_diag.unsqueeze(0) + B4_s_diag.unsqueeze(0) * eta2_expanded_B3B4  # (N, 2, U1, U1)

    # Augmented transition matrix T_is = [[B3, I], [0, I]]  (N, 2, D, D)
    eyeU = safe_eye(U1, device=device, dtype=dtype)
    T_is = torch.zeros((N, 2, D, D), device=device, dtype=dtype)
    T_is[:, :, :U1, :U1] = B3_is
    T_is[:, :, :U1, U1:] = eyeU
    T_is[:, :, U1:, U1:] = eyeU
    T_is_T = T_is.transpose(-1, -2)

    # Augmented intercept c_is = [B1_is; 0]  (N, 2, D)
    c_is = torch.cat(
        [B1_is, torch.zeros((N, 2, U1), device=device, dtype=dtype)], dim=-1
    )

    # ------------------------------------------------------------------
    # Joint stationary initialization of (eta1, zeta2), regime-conditional
    # ------------------------------------------------------------------
    _b0 = torch.diagonal(B3_is, dim1=-2, dim2=-1)          # (N, 2, U1)
    _b0_clip = torch.clamp(_b0, min=-B3_CLIP, max=B3_CLIP)
    _q0 = torch.diagonal(Q1).unsqueeze(0).unsqueeze(0)     # (1, 1, U1)
    A_s = 1.0 / (1.0 - _b0_clip)                           # (N, 2, U1) diag of (I-B3)^{-1}

    mean_eta0 = B1_is * A_s                                # (N, 2, U1) A_s b_is
    mEta_0 = torch.cat(
        [mean_eta0, torch.zeros((N, 2, U1), device=device, dtype=dtype)], dim=-1
    )                                                      # (N, 2, D)

    q2_b = q2_vec.unsqueeze(0).unsqueeze(0)                # (1, 1, U1)
    var_eta0 = _q0 / (1.0 - _b0_clip ** 2) + (A_s ** 2) * q2_b   # (N, 2, U1)
    cov_ez0 = A_s * q2_b                                   # (N, 2, U1)
    var_z0 = q2_b.expand(N, 2, U1)                         # (N, 2, U1)

    mP_0 = torch.zeros((N, 2, D, D), device=device, dtype=dtype)
    idxU = torch.arange(U1, device=device)
    mP_0[:, :, idxU, idxU] = var_eta0
    mP_0[:, :, U1 + idxU, U1 + idxU] = var_z0
    mP_0[:, :, idxU, U1 + idxU] = cov_ez0
    mP_0[:, :, U1 + idxU, idxU] = cov_ez0
    mP_0 = mP_0 + 1e-6 * safe_eye(D, device=device, dtype=dtype)

    P12 = (torch.tensor(p12_fixed_value, dtype=dtype, device=device)
           if fix_p12
           else torch.sigmoid(theta["logit_P12_b"]))

    # Regime-1 centering base: mu1i1_t = A1 (b_i1 + zetahat_t)
    A1 = A_s[:, 0, :]                                       # (N, U1)
    mu1i1_base = B1_is[:, 0, :] * A1                        # (N, U1)

    # ------------------------------------------------------------------
    # History lists
    # ------------------------------------------------------------------
    mEta_list = [mEta_0]
    mP_list = [mP_0]
    mPr_filtered_list = [mPr_filtered_0]
    mPr_pred_list = [mPr_pred_0]
    mLL_list = []
    tPr_list = []

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

        # --- Prediction step (x_row @ T^T = (T x)^T) ---
        jEta_t = (
            c_is.unsqueeze(2)
            + torch.matmul(
                mEta_t.unsqueeze(1).unsqueeze(-2),
                T_is_T.unsqueeze(2),
            ).squeeze(-2)
        )                                                   # (N, 2, 2, D)

        jP_t = (
            torch.matmul(
                torch.matmul(T_is.unsqueeze(2), mP_t.unsqueeze(1)),
                T_is_T.unsqueeze(2),
            )
            + Q_aug.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
        jP_t = jP_t + 1e-6 * safe_eye(D, device=device, dtype=dtype)

        jEta2_t = jEta_t.clone()
        jP2_t = jP_t.clone()
        jLL_t = torch.zeros((N, 2, 2), device=device, dtype=dtype)

        # --- Update step ---
        if len(no_missing_indices_now) > 0:
            idx = torch.tensor(no_missing_indices_now, device=device, dtype=torch.long)

            jV_nm = (
                y1_tensor[idx, t, :].unsqueeze(1).unsqueeze(1)
                - torch.matmul(jEta_t[idx], Lmd_augT.unsqueeze(0).unsqueeze(0))
            )

            jF_nm = (
                torch.matmul(
                    Lmd_aug.unsqueeze(0).unsqueeze(0),
                    torch.matmul(jP_t[idx], Lmd_augT.unsqueeze(0).unsqueeze(0)),
                )
                + R1.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            jF_nm = jF_nm + 1e-6 * safe_eye(O1, device=device, dtype=dtype)

            C = torch.matmul(jP_t[idx], Lmd_augT.unsqueeze(0).unsqueeze(0))
            XT = torch.linalg.solve(jF_nm.transpose(-1, -2), C.transpose(-1, -2))
            KG_nm = XT.transpose(-1, -2)

            jEta2_nm = jEta_t[idx] + torch.matmul(KG_nm, jV_nm.unsqueeze(-1)).squeeze(-1)

            I_KGLmd_nm = (
                safe_eye(D, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                - torch.matmul(KG_nm, Lmd_aug.unsqueeze(0).unsqueeze(0))
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
        # Centering (Assumption 6, choice (a)): mu1i1_t = A1 (b_i1 + zetahat_t),
        # with zetahat_t the regime-1 conditional filtered mean of the zeta2 block.
        if CENTER_MS_PREDICTOR:
            zeta_hat_t = mEta_t[:, 0, U1:]                  # (N, U1)
            mu1i1_t = mu1i1_base + A1 * zeta_hat_t          # (N, U1)
            eta1_pred_t = mEta_t[:, 0, :U1] - mu1i1_t       # (N, U1)
        else:
            # Assumption 6 disabled: raw regime-1 filtered mean, as in the DGP
            eta1_pred_t = mEta_t[:, 0, :U1]                 # (N, U1)

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

        # --- Filter update (Hamilton) ---
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

        # --- Collapsing (approximation; touches the zeta2 block too) ---
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
        mP_next = mP_next + 1e-6 * safe_eye(D, device=device, dtype=dtype)

        mEta_list.append(mEta_next)
        mP_list.append(mP_next)
        mPr_filtered_list.append(mPr_filtered_next)
        mPr_pred_list.append(mPr_pred_next)
        mLL_list.append(mLL_t)
        tPr_list.append(tPr_t.detach().clone())

    return {
        "mEta": torch.stack(mEta_list, dim=1),      # (N, Nt+1, 2, D)
        "mP": torch.stack(mP_list, dim=1),
        "mPr_filtered": torch.stack(mPr_filtered_list, dim=1),
        "mPr_pred": torch.stack(mPr_pred_list, dim=1),
        "mLL": torch.stack(mLL_list, dim=1),
        "eta2": eta2,
        "tPr": torch.stack(tPr_list, dim=1),
    }


# ==============================================================================
# OPG standard errors (integrated estimator; log_Q2d included automatically)
# ==============================================================================

def _compute_opg_se_integrated(
    theta_best, y1_tensor, y2_tensor, DO_matrix,
    N, Nt, O1, O2, U1, missing, non_missing, n_train,
    se_sample_size, device, dtype,
    fix_gamma3=False, fix_gamma4=False,
    fix_gamma1=False, fix_p12=False,
    p12_fixed_value=1e-12,
    gamma3_fixed_value=None, gamma4_fixed_value=None,
):
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
        out_i = _forward_filter_pass_integrated(
            theta=theta_g,
            y1_tensor=y1_tensor[i:i+1], y2_tensor=y2_tensor[i:i+1],
            DO_matrix=DO_matrix[i:i+1],
            N=1, Nt=Nt, O1=O1, O2=O2, U1=U1,
            missing=missing[i:i+1], non_missing=non_missing[i:i+1],
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
# Integrated filtering (single optimization loop; no offset stage)
# ==============================================================================

def _filtering_integrated(
    seed, N, Nt, O1, O2, U1, y1, y2, DO, init, maxIter, n_train, patience,
    min_delta=1e-4, compute_se=False, se_sample_size=None,
    verbose=False, show_progress=False, device="cpu",
    fix_gamma3=False, fix_gamma4=False, fix_gamma1=False, fix_p12=False,
    p12_fixed_value=1e-12, gamma3_fixed_value=None, gamma4_fixed_value=None,
    sim_prior=None,
):
    torch.manual_seed(seed + init)
    np.random.seed(seed + init)

    dtype = torch.float32
    epsilon = 1e-12

    y1 = np.array(y1, dtype=np.float32, copy=True)
    y2 = np.array(y2, dtype=np.float32, copy=True)
    DO = np.array(DO)

    y1_na = np.isnan(y1)
    y1[y1_na] = 0.0

    missing = np.max(y1_na, axis=2).astype(int)
    non_missing = np.min(1 - y1_na.astype(int), axis=2).astype(int)

    y1_tensor = torch.tensor(y1, dtype=dtype, device=device)
    y2_tensor = torch.tensor(y2, dtype=dtype, device=device)

    def _make_theta():
        if sim_prior is not None:
            def _sp(key):
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
            p12_init = float(np.random.uniform(0.01, 0.10))
            logit_p12_b_0 = float(np.log(p12_init / (1.0 - p12_init)))
        else:
            B11_0 = np.random.normal(0, 0.5, U1)
            Delta_B12_raw = np.log(rtruncnorm_positive(U1, mean=0, sd=0.5) + epsilon)
            B21_0 = np.random.normal(0, 0.3, U1)
            B22_0 = B21_0 + np.random.normal(0, 0.3, U1)
            B31d_0 = np.random.uniform(0.1, 0.8, U1)
            B32d_0 = np.random.uniform(0.1, 0.8, U1)
            B41d_0 = np.random.normal(0, 0.1, U1)
            B42d_0 = B41d_0 + np.random.normal(0, 0.1, U1)
            log_Q1d_0 = np.log(1 / (np.random.gamma(shape=9, scale=1 / 4, size=U1) + epsilon))
            log_R1d_0 = np.log(1 / (np.random.gamma(shape=9, scale=1 / 4, size=O1) + epsilon))
            R2d_0 = 1 / (np.random.gamma(shape=9, scale=1 / 4, size=O2) + epsilon)
            R2d_0[R2d_0 < epsilon] = epsilon
            log_R2d_0 = np.log(R2d_0)
            gamma1_0 = float(truncnorm.rvs(0, np.inf, loc=0, scale=1))
            gamma2_0 = float(np.random.normal(0, 0.3))
            gamma3_0 = np.random.normal(0, 0.3, U1)
            gamma4_0 = np.random.normal(0, 0.1, U1)
            p12_init = float(np.random.uniform(0.01, 0.10))
            logit_p12_b_0 = float(np.log(p12_init / (1.0 - p12_init)))

        # Q2 start: moderate between-person variance (prior CSV has no Q2 entry)
        log_Q2d_0 = np.log(np.random.uniform(0.05, 0.30, U1))

        d = {
            "gamma1":        torch.tensor([GAMMA1_FIXED if fix_gamma1 else gamma1_0], dtype=dtype, device=device, requires_grad=(not fix_gamma1)),
            "B11":           torch.tensor(B11_0,         dtype=dtype, device=device, requires_grad=True),
            "log_B12_delta": torch.tensor(Delta_B12_raw, dtype=dtype, device=device, requires_grad=True),
            "B21":           torch.tensor(B21_0,         dtype=dtype, device=device, requires_grad=True),
            "B22":           torch.tensor(B22_0,         dtype=dtype, device=device, requires_grad=True),
            "B31d":          torch.tensor(B31d_0,        dtype=dtype, device=device, requires_grad=True),
            "B32d":          torch.tensor(B32d_0,        dtype=dtype, device=device, requires_grad=True),
            "B41d":          torch.tensor(B41d_0,        dtype=dtype, device=device, requires_grad=True),
            "B42d":          torch.tensor(B42d_0,        dtype=dtype, device=device, requires_grad=True),
            "log_Q1d":       torch.tensor(log_Q1d_0,     dtype=dtype, device=device, requires_grad=True),
            "log_R1d":       torch.tensor(log_R1d_0,     dtype=dtype, device=device, requires_grad=True),
            "log_R2d":       torch.tensor(log_R2d_0,     dtype=dtype, device=device, requires_grad=True),
            "log_Q2d":       torch.tensor(log_Q2d_0,     dtype=dtype, device=device, requires_grad=True),
            "gamma2":        torch.tensor(gamma2_0,      dtype=dtype, device=device, requires_grad=True),
        }
        if not fix_gamma3:
            d["gamma3"] = torch.tensor(gamma3_0, dtype=dtype, device=device, requires_grad=True)
        if not fix_gamma4:
            d["gamma4"] = torch.tensor(gamma4_0, dtype=dtype, device=device, requires_grad=True)
        if not fix_p12:
            d["logit_P12_b"] = torch.tensor([logit_p12_b_0], dtype=dtype, device=device, requires_grad=True)
        return d

    theta = _make_theta()
    optimizer = Rprop(list(theta.values()), lr=0.1, etas=(0.5, 1.2))

    sumLL_best = -np.inf
    patience_counter = 0
    theta_best = None
    iteration = 1

    def _run_pass(th):
        return _forward_filter_pass_integrated(
            theta=th, y1_tensor=y1_tensor, y2_tensor=y2_tensor, DO_matrix=DO,
            N=N, Nt=Nt, O1=O1, O2=O2, U1=U1,
            missing=missing, non_missing=non_missing,
            device=device, dtype=dtype,
            fix_gamma3=fix_gamma3, fix_gamma4=fix_gamma4,
            fix_gamma1=fix_gamma1, fix_p12=fix_p12,
            p12_fixed_value=p12_fixed_value,
            gamma3_fixed_value=gamma3_fixed_value,
            gamma4_fixed_value=gamma4_fixed_value,
        )

    while iteration <= maxIter:
        if iteration <= 30:
            optimizer.param_groups[0]["lr"] = 0.005
        elif iteration == 31:
            optimizer.param_groups[0]["lr"] = 0.01

        out = _run_pass(theta)
        loss = -torch.nanmean(out["mLL"][:, :n_train])
        sumLL = -loss.detach().cpu().item()

        if not np.isfinite(sumLL):
            if verbose:
                print("Warning: infinite/NaN loss in integrated filtering")
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

    # Final trajectories at the best parameters
    with torch.no_grad():
        out_best = _run_pass(theta_best)

    mEta_best = out_best["mEta"]                     # (N, Nt+1, 2, D)
    mPr_filtered_best = out_best["mPr_filtered"]

    prob_regime1 = mPr_filtered_best[:, 1:(Nt + 1), 0].clone()
    prob_regime2 = mPr_filtered_best[:, 1:(Nt + 1), 1].clone()
    eta_regime1 = mEta_best[:, 1:(Nt + 1), 0, :U1].clone()
    eta_regime2 = mEta_best[:, 1:(Nt + 1), 1, :U1].clone()

    estimated_eta = prob_regime1.unsqueeze(-1) * eta_regime1 + prob_regime2.unsqueeze(-1) * eta_regime2
    estimated_regime_prob = prob_regime2

    traj_results = {
        "eta_est": estimated_eta.cpu().numpy(),
        "regime_prob": estimated_regime_prob.cpu().numpy(),
    }

    # zeta2 point estimates: regime-weighted filtered mean of the zeta block at n_train
    z1 = mEta_best[:, n_train, 0, U1:]
    z2 = mEta_best[:, n_train, 1, U1:]
    p1_T = mPr_filtered_best[:, n_train, 0].unsqueeze(-1)
    p2_T = mPr_filtered_best[:, n_train, 1].unsqueeze(-1)
    zeta2_estimated = (p1_T * z1 + p2_T * z2).cpu().numpy()

    q2_diag_est = torch.exp(
        torch.clamp(theta_best["log_Q2d"], min=math.log(1e-6), max=math.log(1e1))
    ).detach().cpu().numpy()

    # P2 estimate (empirical variance of Bartlett factor scores) -- unchanged
    R2_best = torch.diag(torch.exp(torch.clamp(theta_best["log_R2d"], min=math.log(1e-6), max=math.log(1e1))))
    Lmd2_fixed = torch.ones((O2, 1), device=device, dtype=dtype)
    term1_best = Lmd2_fixed.T @ torch.linalg.inv(R2_best) @ Lmd2_fixed
    W_bartlett_best = (
        torch.linalg.inv(term1_best + 1e-6 * torch.eye(1, device=device, dtype=dtype))
        @ Lmd2_fixed.T
        @ torch.linalg.inv(R2_best)
    )
    P2_estimated_val = torch.var((y2_tensor @ W_bartlett_best.T).squeeze(-1)).detach().cpu().item()

    # ------------------------------------------------------------------
    # Final estimates dataframe (log_Q2d handled by the ^log_[QR] transform)
    # ------------------------------------------------------------------
    param_names = list(theta_best.keys())
    est_vec = []
    grad_param_names = []
    for nm in param_names:
        vals = theta_best[nm].detach().cpu().numpy().reshape(-1)
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

    B11_val = theta_best["B11"].detach().cpu().numpy().reshape(-1)
    log_delta_val = theta_best["log_B12_delta"].detach().cpu().numpy().reshape(-1)
    B12_est = B11_val + np.exp(log_delta_val)
    B12_names = [f"B12_{i+1}" for i in range(len(B11_val))] if len(B11_val) > 1 else ["B12"]
    B12_df = pd.DataFrame({"Parameter": B12_names, "Estimate": B12_est, "SE": np.nan})
    final_estimates_df = pd.concat([final_estimates_df, B12_df], ignore_index=True)

    rows_to_transform = final_estimates_df["Parameter"].str.contains(r"^log_[QR]", regex=True)
    for i2 in final_estimates_df.index[rows_to_transform]:
        est_log = final_estimates_df.at[i2, "Estimate"]
        final_estimates_df.at[i2, "Estimate"] = np.exp(est_log)
        final_estimates_df.at[i2, "Parameter"] = final_estimates_df.at[i2, "Parameter"].replace("log_", "", 1)

    final_estimates_df = final_estimates_df[
        ~final_estimates_df["Parameter"].str.contains(r"^log_B12_delta", regex=True)
    ].reset_index(drop=True)

    mask_p12 = final_estimates_df["Parameter"] == "logit_P12_b"
    if mask_p12.any():
        logit_val = final_estimates_df.loc[mask_p12, "Estimate"].values[0]
        final_estimates_df.loc[mask_p12, "Estimate"] = float(torch.sigmoid(torch.tensor(logit_val)).item())
        final_estimates_df.loc[mask_p12, "Parameter"] = "P12"

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

    # ------------------------------------------------------------------
    # OPG SE (log_Q2d included; mapped to Q2d rows)
    # ------------------------------------------------------------------
    if compute_se and theta_best is not None:
        se_np, se_names = _compute_opg_se_integrated(
            theta_best=theta_best,
            y1_tensor=y1_tensor, y2_tensor=y2_tensor, DO_matrix=DO,
            N=N, Nt=Nt, O1=O1, O2=O2, U1=U1,
            missing=missing, non_missing=non_missing,
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
            for pfx, sfx in [("log_Q1d", "Q1d"), ("log_Q2d", "Q2d"),
                             ("log_R1d", "R1d"), ("log_R2d", "R2d")]:
                if out_name.startswith(pfx):
                    out_name = out_name.replace(pfx, sfx, 1)
            if out_name.startswith("log_B12_delta"):
                continue
            mask = final_estimates_df["Parameter"] == out_name
            if mask.any():
                final_estimates_df.loc[mask, "SE"] = float(se_val)

    return {
        "sumLL_best": sumLL_best,
        "theta_best": theta_best,
        "P2_estimated": P2_estimated_val,
        "final_estimates": final_estimates_df,
        "trajectories": traj_results,
        "Q2_estimated_diag": q2_diag_est,
        "zeta2_estimated": zeta2_estimated,
        "method": "integrated",
    }


# ==============================================================================
# Public entry point -- signature-compatible with sim_filtering.filtering()
# (two_stage_* arguments are accepted and ignored)
# ==============================================================================

def filtering_integrated(
    seed, N, Nt, O1, O2, U1, y1, y2, DO, init, maxIter, n_train, patience,
    min_delta=1e-4, compute_se=False, se_sample_size=None,
    verbose=False, show_progress=False, device="cpu",
    method="two_stage",
    two_stage_outer_loops=None, two_stage_damping=None,
    fix_gamma3=False, fix_gamma4=False, fix_gamma1=False, fix_p12=False,
    p12_fixed_value=1e-12, gamma3_fixed_value=None, gamma4_fixed_value=None,
    sim_prior=None,
):
    return _filtering_integrated(
        seed=seed, N=N, Nt=Nt, O1=O1, O2=O2, U1=U1,
        y1=y1, y2=y2, DO=DO, init=init, maxIter=maxIter,
        n_train=n_train, patience=patience, min_delta=min_delta,
        compute_se=compute_se, se_sample_size=se_sample_size,
        verbose=verbose, show_progress=show_progress, device=device,
        fix_gamma3=fix_gamma3, fix_gamma4=fix_gamma4,
        fix_gamma1=fix_gamma1, fix_p12=fix_p12,
        p12_fixed_value=p12_fixed_value,
        gamma3_fixed_value=gamma3_fixed_value,
        gamma4_fixed_value=gamma4_fixed_value,
        sim_prior=sim_prior,
    )
