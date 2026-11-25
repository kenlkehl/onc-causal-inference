"""High-dimensional Sparse Tabular CausalPFN (``SparseTabularCausalPFN``).

Scales cleanly to P = 2,000+ candidate features and N = 10,000+ patients via:
  1. Strictly context-only standardization & O(N * P) observational per-column
     sufficient statistics (separating confounders, prognostics, modifiers,
     pure instruments, and collinear duplicates).
  2. O(P) DeepSets column role encoder & screener (with auxiliary 6-class role
     supervision during synthetic pretraining and interpretable role posteriors
     at inference time).
  3. Round-1 top-K_nuis correlation-biased column self-attention -> soft ARD
     spike-and-slab gates -> closed-form differentiable LOO-CV Ridge nuisances
     (e_loo, m_loo) with instrument suppression.
  4. Round-2 ALL-P adjusted Robinson interaction scan -> promotes subtle effect
     modifiers masked by strong confounders -> top-K_cate correlation-biased
     attention -> closed-form differentiable ARD LOO-CV Robinson R-learner.
  5. Residual two-way TabPFN study layers over top gated slots + 129-bin sinh
     posterior uncertainty head.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from oci.longitudinal.model import ArmScale, RidgeFit, _predict_ridge, _sinh_grid

from .prior import ROLE_NAMES

CHECKPOINT_FORMAT_VERSION = 1
COL_STAT_DIM = 24
COL_STAT_DIM_V3 = 32
ROB_STAT_DIM = 6
ROB_STAT_DIM_V3 = 12
ROB_STAT_DIM_V4 = 16
NUIS_PAIR_STAT_DIM = 6
CATE_PAIR_STAT_DIM = 4
NUM_ROLES = len(ROLE_NAMES)


@dataclass(frozen=True)
class TabularModelConfig:
    col_dim: int = 128
    study_width: int = 128
    study_layers: int = 2
    col_attn_layers: int = 2
    heads: int = 4
    bins: int = 129
    top_k_nuis: int = 64
    top_k_cate: int = 48
    attn_slots: int = 16
    ridge_grid_size: int = 7
    cate_loss_weight: float = 2.0
    propensity_loss_weight: float = 0.5
    role_loss_weight: float = 0.3
    decouple_rlearner_propensity: bool = False
    use_v3_solver_basis: bool = False
    use_v4_interactions: bool = False
    pair_top_k: int = 16

    def __post_init__(self):
        if (
            min(
                self.col_dim,
                self.study_width,
                self.heads,
                self.study_layers,
                self.col_attn_layers,
                self.top_k_nuis,
                self.top_k_cate,
                self.attn_slots,
            )
            < 1
            or self.pair_top_k < 2
            or self.ridge_grid_size < 2
            or self.bins < 3
            or self.cate_loss_weight < 0
            or self.propensity_loss_weight < 0
            or self.role_loss_weight < 0
        ):
            raise ValueError("Invalid tabular PFN model configuration")
        if self.col_dim % self.heads or self.study_width % self.heads:
            raise ValueError("col_dim and study_width must be divisible by heads")
        if self.use_v4_interactions and not self.use_v3_solver_basis:
            raise ValueError("use_v4_interactions requires use_v3_solver_basis=True")


@dataclass
class TabularMemory:
    """Context-conditioned memory for zero-shot query scoring & role inspection."""

    col_mean: torch.Tensor  # (1, P)
    col_sd: torch.Tensor  # (1, P)
    x2_mean: torch.Tensor  # (1, P)
    x2_sd: torch.Tensor  # (1, P)
    valid_col: torch.Tensor  # (P,) bool
    col_embs: torch.Tensor  # (P, col_dim)
    role_logits: torch.Tensor  # (P, 6)
    nuis_screen_logits: torch.Tensor  # (P,)
    mod_screen_logits: torch.Tensor  # (P,)
    nuis_idx: torch.Tensor  # (K1,)
    cate_idx: torch.Tensor  # (K2,)
    attn_idx: torch.Tensor  # (K_attn,)
    g_e_lin: torch.Tensor  # (K1,)
    g_e_nl: torch.Tensor  # (K1,)
    g_er_lin: torch.Tensor  # (K1,) outcome-adaptive R-learner propensity gate
    g_er_nl: torch.Tensor  # (K1,)
    g_m_lin: torch.Tensor  # (K1,)
    g_m_nl: torch.Tensor  # (K1,)
    g_tau_lin: torch.Tensor  # (K2,)
    g_tau_nl: torch.Tensor  # (K2,)
    attn_gate_feats: torch.Tensor  # (K_attn, 3)
    propensity_fit: RidgeFit
    rlearner_prop_fit: RidgeFit
    outcome_fit: RidgeFit
    cate_fit: RidgeFit
    attn_memories: list
    n_ctx: int
    n_features: int
    # Optional v3 basis stats & gates (None for v1/v2 checkpoints)
    xsat_mean: torch.Tensor | None = None
    xsat_sd: torch.Tensor | None = None
    miss_mean: torch.Tensor | None = None
    miss_sd: torch.Tensor | None = None
    miss_weight: torch.Tensor | None = None
    g_e_sat: torch.Tensor | None = None
    g_e_miss: torch.Tensor | None = None
    g_er_sat: torch.Tensor | None = None
    g_er_miss: torch.Tensor | None = None
    g_m_sat: torch.Tensor | None = None
    g_m_miss: torch.Tensor | None = None
    g_tau_sat: torch.Tensor | None = None
    g_tau_miss: torch.Tensor | None = None
    # Optional v4 pairwise interaction basis stats & gates (None for v1/v2/v3 checkpoints)
    nuis_pair_a: torch.Tensor | None = None
    nuis_pair_b: torch.Tensor | None = None
    nuis_pair_mean: torch.Tensor | None = None
    nuis_pair_sd: torch.Tensor | None = None
    nuis_pair_weight: torch.Tensor | None = None
    nuis_pair_valid: torch.Tensor | None = None
    g_pe: torch.Tensor | None = None
    g_per: torch.Tensor | None = None
    g_pm: torch.Tensor | None = None
    cate_pair_a: torch.Tensor | None = None
    cate_pair_b: torch.Tensor | None = None
    cate_pair_mean: torch.Tensor | None = None
    cate_pair_sd: torch.Tensor | None = None
    cate_pair_weight: torch.Tensor | None = None
    cate_pair_valid: torch.Tensor | None = None
    g_ptau: torch.Tensor | None = None

    def max_gate(self, kind: str) -> torch.Tensor:
        if kind == "er":
            parts = [self.g_er_lin, self.g_er_nl, self.g_er_sat, self.g_er_miss]
        elif kind == "m":
            parts = [self.g_m_lin, self.g_m_nl, self.g_m_sat, self.g_m_miss]
        elif kind == "tau":
            parts = [self.g_tau_lin, self.g_tau_nl, self.g_tau_sat, self.g_tau_miss]
        elif kind == "e":
            parts = [self.g_e_lin, self.g_e_nl, self.g_e_sat, self.g_e_miss]
        else:
            raise ValueError(f"Unknown gate kind {kind!r}")
        valid = [t for t in parts if t is not None]
        return torch.stack(valid, dim=-1).max(dim=-1).values


def _fit_ard_loo_cv_ridge(A_ctx, y_ctx, lambdas, cv_temp, *, intercept_reg=1e-4):
    """Differentiable closed-form batched LOO-CV Ridge via Cholesky (exact, strictly PD, fp32)."""
    d = A_ctx.shape[1]
    k = len(lambdas)
    G = A_ctx.T @ A_ctx  # (D, D)
    rhs = (A_ctx.T @ y_ctx).unsqueeze(0).expand(k, -1).unsqueeze(-1)  # (K, D, 1)

    eye = torch.eye(d, device=A_ctx.device, dtype=A_ctx.dtype)
    pen = lambdas[:, None, None] * eye[None]
    pen[:, 0, 0] = intercept_reg
    chol = torch.linalg.cholesky(G.unsqueeze(0) + pen)  # (K, D, D)
    w = torch.cholesky_solve(rhs, chol).squeeze(-1)  # (K, D)

    V = torch.linalg.solve_triangular(
        chol, A_ctx.T.unsqueeze(0).expand(k, -1, -1), upper=False
    )  # (K, D, N)
    h = (V * V).sum(dim=1)  # (K, N)
    y_hat = (A_ctx @ w.T).T  # (K, N)
    denom = (1.0 - h).clamp_min(1e-2)
    loo_resid = (y_ctx.unsqueeze(0) - y_hat) / denom
    loo_mse = loo_resid.square().mean(dim=1)  # (K,)
    y_loo_k = y_ctx.unsqueeze(0) - loo_resid

    scale = loo_mse.detach().std(unbiased=False).clamp_min(1e-5)
    alpha = F.softmax(-cv_temp * (loo_mse - loo_mse.min()) / scale, dim=0)
    w_blend = (alpha[:, None] * w).sum(dim=0)
    y_loo = (alpha[:, None] * y_loo_k).sum(dim=0)
    return RidgeFit(weights=w_blend, chol=chol, alpha=alpha), y_loo


@torch.no_grad()
def _standardize_context(X_ctx: torch.Tensor, M_ctx: torch.Tensor | None = None):
    """Fit context-only column mean/std and return standardized (X_s, X2_s, M_s, mean, sd, valid)."""
    X_f = X_ctx.float()
    nan_mask = ~torch.isfinite(X_f)
    if M_ctx is None:
        M_s = nan_mask.float()
    else:
        M_s = torch.maximum(M_ctx.float(), nan_mask.float()).clamp(0.0, 1.0)
    obs = (~nan_mask).float()
    obs_cnt = obs.sum(dim=0, keepdim=True)  # (1, P)
    X_zero = torch.where(nan_mask, torch.zeros_like(X_f), X_f)
    col_mean = X_zero.sum(dim=0, keepdim=True) / obs_cnt.clamp_min(1.0)
    diff = torch.where(nan_mask, torch.zeros_like(X_f), X_f - col_mean)
    var = diff.square().sum(dim=0, keepdim=True) / obs_cnt.clamp_min(1.0)
    valid_col = (obs_cnt.squeeze(0) >= 2.0) & (var.squeeze(0) > 1e-10)
    col_sd = var.sqrt().clamp_min(1e-4)

    X_s = ((X_zero - col_mean) / col_sd) * obs
    X_s = X_s.clamp(-6.0, 6.0)
    X_s = X_s * valid_col.float().unsqueeze(0)

    # Centered nonlinear quadratic feature (zeroed on binary/constant columns where X^2 is collinear)
    X2_raw = (X_s * X_s - 1.0) * obs
    X2_mu = X2_raw.mean(dim=0, keepdim=True)
    X2_c = X2_raw - X2_mu
    X2_sd = X2_c.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    X2_s = (X2_c / X2_sd).clamp(-6.0, 6.0) * valid_col.float().unsqueeze(0)
    return X_s, X2_s, M_s, col_mean, col_sd, X2_mu, X2_sd, valid_col


@torch.no_grad()
def _apply_standardize(
    X: torch.Tensor,
    M: torch.Tensor | None,
    col_mean: torch.Tensor,
    col_sd: torch.Tensor,
    x2_mean: torch.Tensor,
    x2_sd: torch.Tensor,
    valid_col: torch.Tensor,
):
    X_f = X.float()
    nan_mask = ~torch.isfinite(X_f)
    if M is None:
        M_s = nan_mask.float()
    else:
        M_s = torch.maximum(M.float(), nan_mask.float()).clamp(0.0, 1.0)
    obs = (~nan_mask).float()
    X_zero = torch.where(nan_mask, col_mean.expand_as(X_f), X_f)
    X_s = (((X_zero - col_mean) / col_sd) * obs).clamp(-6.0, 6.0) * valid_col.float().unsqueeze(0)
    X2_raw = (X_s * X_s - 1.0) * obs
    X2_s = ((X2_raw - x2_mean) / x2_sd).clamp(-6.0, 6.0) * valid_col.float().unsqueeze(0)
    return X_s, X2_s, M_s


@torch.no_grad()
def _compute_v3_basis_context(
    X_s: torch.Tensor,
    M_s: torch.Tensor,
    obs: torch.Tensor,
    valid_col: torch.Tensor,
):
    """Smooth C^inf saturating threshold/outlier basis (X_sat) + variance-gated missingness (M_c)."""
    xsat_raw = torch.tanh(1.5 * X_s) * obs
    xsat_mean = xsat_raw.mean(dim=0, keepdim=True)
    xsat_c = xsat_raw - xsat_mean
    xsat_sd = xsat_c.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    X_sat = (xsat_c / xsat_sd).clamp(-6.0, 6.0) * valid_col.float().unsqueeze(0)

    miss_mean = M_s.mean(dim=0, keepdim=True)
    miss_c = M_s - miss_mean
    miss_var = miss_c.square().mean(dim=0, keepdim=True)
    miss_sd = miss_var.sqrt().clamp_min(1e-3)
    miss_weight = (miss_var / (miss_var + 2e-3)).sqrt()
    M_c = ((miss_c / miss_sd) * miss_weight).clamp(-6.0, 6.0) * valid_col.float().unsqueeze(0)
    return X_sat, M_c, xsat_mean, xsat_sd, miss_mean, miss_sd, miss_weight


@torch.no_grad()
def _apply_v3_basis(
    X_s: torch.Tensor,
    M_s: torch.Tensor,
    obs: torch.Tensor,
    valid_col: torch.Tensor,
    xsat_mean: torch.Tensor,
    xsat_sd: torch.Tensor,
    miss_mean: torch.Tensor,
    miss_sd: torch.Tensor,
    miss_weight: torch.Tensor,
):
    xsat_raw = torch.tanh(1.5 * X_s) * obs
    X_sat = ((xsat_raw - xsat_mean) / xsat_sd).clamp(-6.0, 6.0) * valid_col.float().unsqueeze(0)
    M_c = (((M_s - miss_mean) / miss_sd) * miss_weight).clamp(-6.0, 6.0) * valid_col.float().unsqueeze(0)
    return X_sat, M_c


@torch.no_grad()
def _build_pairs_context(
    X_s: torch.Tensor,
    valid_col: torch.Tensor,
    col_a: torch.Tensor,
    col_b: torch.Tensor,
):
    """Build smooth C^inf variance-weighted standardized pairwise products on context."""
    raw = X_s[:, col_a] * X_s[:, col_b]  # (N, M)
    p_mean = raw.mean(dim=0, keepdim=True)
    p_c = raw - p_mean
    p_var = p_c.square().mean(dim=0, keepdim=True)
    p_sd = p_var.sqrt().clamp_min(1e-3)
    # Smoothly zero out degenerate / mutually exclusive one-hot dummy pairs
    p_weight = (p_var / (p_var + 2e-3)).sqrt()
    p_valid = (valid_col[col_a] & valid_col[col_b]).float()
    P_s = ((p_c / p_sd) * p_weight).clamp(-6.0, 6.0) * p_valid.unsqueeze(0)
    return P_s, p_mean, p_sd, p_weight, p_valid


@torch.no_grad()
def _apply_pairs(
    X_s: torch.Tensor,
    col_a: torch.Tensor,
    col_b: torch.Tensor,
    p_mean: torch.Tensor,
    p_sd: torch.Tensor,
    p_weight: torch.Tensor,
    p_valid: torch.Tensor,
) -> torch.Tensor:
    raw = X_s[:, col_a] * X_s[:, col_b]
    return (((raw - p_mean) / p_sd) * p_weight).clamp(-6.0, 6.0) * p_valid.unsqueeze(0)


@torch.no_grad()
def _corr_and_tstat(X: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None):
    """Vectorized column-wise Pearson r and t-statistic against a 1D target."""
    if mask is not None:
        w = mask.float()
        n = w.sum().clamp_min(3.0)
        x_mu = (X * w[:, None]).sum(dim=0, keepdim=True) / n
        y_mu = (target * w).sum() / n
        xc = (X - x_mu) * w[:, None]
        yc = (target - y_mu) * w
    else:
        n = X.new_tensor(float(max(len(target), 3)))
        xc = X - X.mean(dim=0, keepdim=True)
        yc = target - target.mean()
    cov = (xc * yc[:, None]).sum(dim=0)
    denom = (xc.square().sum(dim=0).clamp_min(1e-8) * yc.square().sum().clamp_min(1e-8)).sqrt()
    r = (cov / denom).clamp(-0.995, 0.995)
    t_stat = r * ((n - 2.0) / (1.0 - r * r).clamp_min(1e-4)).sqrt()
    return r, t_stat.clamp(-25.0, 25.0)


@torch.no_grad()
def _column_sufficient_stats(
    X_s: torch.Tensor,
    X2_s: torch.Tensor,
    M_s: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    valid_col: torch.Tensor,
    *,
    X_sat: torch.Tensor | None = None,
    M_c: torch.Tensor | None = None,
):
    """Compute 24-d (v1/v2) or 32-d (v3) strictly observational per-column descriptors in O(N * P)."""
    n, p = X_s.shape
    t_f = t.float()
    y_f = y.float()
    m0 = t_f < 0.5
    m1 = ~m0
    n0 = m0.float().sum().clamp_min(2.0)
    n1 = m1.float().sum().clamp_min(2.0)

    r_t, t_t = _corr_and_tstat(X_s, t_f)
    r_t2, _ = _corr_and_tstat(X2_s, t_f)
    r_y, t_y = _corr_and_tstat(X_s, y_f)
    r_y0, t_y0 = _corr_and_tstat(X_s, y_f, mask=m0)
    r_y1, t_y1 = _corr_and_tstat(X_s, y_f, mask=m1)
    r_y0_2, _ = _corr_and_tstat(X2_s, y_f, mask=m0)
    r_y1_2, _ = _corr_and_tstat(X2_s, y_f, mask=m1)

    # Zero arm stats if an arm is degenerate
    deg0 = int(m0.sum().item()) < 3
    deg1 = int(m1.sum().item()) < 3
    if deg0:
        r_y0.zero_(); t_y0.zero_(); r_y0_2.zero_()
    if deg1:
        r_y1.zero_(); t_y1.zero_(); r_y1_2.zero_()

    abs_t_t = t_t.abs()
    y_cond_t = torch.maximum(t_y0.abs(), t_y1.abs())
    se_diff = (1.0 / (n0 - 1.0).clamp_min(1.0) + 1.0 / (n1 - 1.0).clamp_min(1.0)).sqrt()
    z_inter = ((r_y1 - r_y0) / se_diff).clamp(-25.0, 25.0)
    z_inter2 = ((r_y1_2 - r_y0_2) / se_diff).clamp(-25.0, 25.0)
    abs_z_inter = z_inter.abs()

    # Distributional descriptors
    miss_rate = M_s.mean(dim=0)
    # Binary/endpoint proxy and heavy-tail weight (smooth wrt X_s so floating-point
    # reduction order under row/column permutations cannot flip discrete thresholds)
    col_min = X_s.min(dim=0).values
    col_max = X_s.max(dim=0).values
    span = (col_max - col_min).clamp_min(1e-4)
    d_end = torch.minimum(X_s - col_min, col_max - X_s) / span
    near_end = torch.exp(-((d_end / 0.05).square())).mean(dim=0)
    tail_frac = torch.sigmoid(4.0 * (X_s.abs() - 2.0)).mean(dim=0)
    log_n = X_s.new_full((p,), math.log1p(n))
    log_p = X_s.new_full((p,), math.log1p(p))

    # Fast O(32 * N * P) redundancy proxy against top-32 signal-bearing columns
    raw_sig = (abs_t_t + y_cond_t + abs_z_inter) * valid_col.float()
    k_ref = min(32, p)
    ref_idx = torch.topk(raw_sig, k_ref, largest=True).indices
    X_norm = F.normalize(X_s - X_s.mean(dim=0, keepdim=True), p=2, dim=0)
    corr_ref = (X_norm.T @ X_norm[:, ref_idx]).abs()  # (P, k_ref)
    # Mask self-correlation cleanly
    self_mask = torch.arange(p, device=X_s.device)[:, None] == ref_idx[None, :]
    corr_ref = corr_ref.masked_fill(self_mask, 0.0)
    max_corr = corr_ref.max(dim=1).values
    k3 = min(3, k_ref)
    mean_top3_corr = torch.topk(corr_ref, k3, dim=1).values.mean(dim=1)

    stat_list = [
        r_t,
        t_t / 5.0,
        abs_t_t / 5.0,
        torch.log1p(abs_t_t),
        r_t2,
        r_y,
        t_y / 5.0,
        r_y0,
        t_y0 / 5.0,
        r_y1,
        t_y1 / 5.0,
        y_cond_t / 5.0,
        torch.log1p(y_cond_t),
        z_inter / 5.0,
        abs_z_inter / 5.0,
        torch.log1p(abs_z_inter),
        z_inter2 / 5.0,
        miss_rate,
        near_end,
        tail_frac,
        log_n / 8.0,
        log_p / 8.0,
        max_corr,
        mean_top3_corr,
    ]
    raw_nuis_priority = (
        torch.log1p(y_cond_t) + 0.35 * torch.log1p(abs_t_t) + 0.5 * torch.log1p(abs_z_inter)
    )

    if X_sat is not None and M_c is not None:
        r_t_sat, _ = _corr_and_tstat(X_sat, t_f)
        r_y0_sat, t_y0_sat = _corr_and_tstat(X_sat, y_f, mask=m0)
        r_y1_sat, t_y1_sat = _corr_and_tstat(X_sat, y_f, mask=m1)
        r_t_m, _ = _corr_and_tstat(M_c, t_f)
        r_y0_m, t_y0_m = _corr_and_tstat(M_c, y_f, mask=m0)
        r_y1_m, t_y1_m = _corr_and_tstat(M_c, y_f, mask=m1)
        if deg0:
            r_y0_sat.zero_(); t_y0_sat.zero_(); r_y0_m.zero_(); t_y0_m.zero_()
        if deg1:
            r_y1_sat.zero_(); t_y1_sat.zero_(); r_y1_m.zero_(); t_y1_m.zero_()
        z_inter_sat = ((r_y1_sat - r_y0_sat) / se_diff).clamp(-25.0, 25.0)
        z_inter_m = ((r_y1_m - r_y0_m) / se_diff).clamp(-25.0, 25.0)
        stat_list.extend(
            [
                r_t_sat,
                t_y0_sat / 5.0,
                t_y1_sat / 5.0,
                z_inter_sat / 5.0,
                r_t_m,
                t_y0_m / 5.0,
                t_y1_m / 5.0,
                z_inter_m / 5.0,
            ]
        )
        y_cond_sat = torch.maximum(t_y0_sat.abs(), t_y1_sat.abs())
        y_cond_m = torch.maximum(t_y0_m.abs(), t_y1_m.abs())
        raw_nuis_priority = (
            raw_nuis_priority + 0.25 * torch.log1p(y_cond_sat) + 0.25 * torch.log1p(y_cond_m)
        )

    stats = torch.stack(stat_list, dim=-1) * valid_col.float().unsqueeze(-1)
    raw_nuis_priority = torch.where(valid_col, raw_nuis_priority, raw_nuis_priority.new_full((), -1e4))
    return stats, raw_nuis_priority


@torch.no_grad()
def _robinson_all_p_stats(
    X_s: torch.Tensor,
    X2_s: torch.Tensor,
    t_res: torch.Tensor,
    y_res: torch.Tensor,
    valid_col: torch.Tensor,
    *,
    X_sat: torch.Tensor | None = None,
    M_c: torch.Tensor | None = None,
):
    """Scan ALL P columns against adjusted LOO residuals (t - e_loo, y - m_loo) in O(N * P)."""
    y_var = y_res.square().mean().clamp_min(1e-5)

    def _z_score(Z):
        zt = Z * t_res[:, None]
        num = (zt * y_res[:, None]).sum(dim=0)
        denom = (zt.square().sum(dim=0).clamp_min(1e-6) * y_var).sqrt()
        return (num / denom).clamp(-25.0, 25.0)

    z_lin = _z_score(X_s) * valid_col.float()
    z_nl = _z_score(X2_s) * valid_col.float()
    abs_lin = z_lin.abs()
    abs_nl = z_nl.abs()
    rob_list = [
        z_lin / 5.0,
        abs_lin / 5.0,
        torch.log1p(abs_lin),
        z_nl / 5.0,
        abs_nl / 5.0,
        torch.log1p(abs_nl),
    ]
    raw_mod_priority = torch.log1p(abs_lin) + 0.6 * torch.log1p(abs_nl)
    if X_sat is not None and M_c is not None:
        z_sat = _z_score(X_sat) * valid_col.float()
        z_m = _z_score(M_c) * valid_col.float()
        abs_sat = z_sat.abs()
        abs_m = z_m.abs()
        rob_list.extend(
            [
                z_sat / 5.0,
                abs_sat / 5.0,
                torch.log1p(abs_sat),
                z_m / 5.0,
                abs_m / 5.0,
                torch.log1p(abs_m),
            ]
        )
        raw_mod_priority = raw_mod_priority + 0.45 * torch.log1p(abs_sat) + 0.30 * torch.log1p(abs_m)
    rob_feats = torch.stack(rob_list, dim=-1)
    raw_mod_priority = torch.where(valid_col, raw_mod_priority, raw_mod_priority.new_full((), -1e4))
    return rob_feats, raw_mod_priority


@torch.no_grad()
def _robinson_confounder_mod_stats(
    X_s: torch.Tensor,
    t_res: torch.Tensor,
    y_res: torch.Tensor,
    valid_col: torch.Tensor,
    ref_cols: torch.Tensor,
):
    """O(R * N * P) exact centered confounder-modulated Robinson interaction scan (R <= 8).

    Rescues pure-epistatic confounder x modifier HTE interactions where marginal
    E[tau | X_m] == 0 so standard 1D Robinson scan misses X_m.
    """
    n, p = X_s.shape
    r_cnt = len(ref_cols)
    if r_cnt == 0 or p < 2:
        return X_s.new_zeros((p, 4)), X_s.new_zeros((p,))
    y_var = y_res.square().mean().clamp_min(1e-5)
    X_ref = X_s[:, ref_cols]  # (N, R)
    # Exact product mean mu_{r,j} = (1/N) sum_i X_{i,ref[r]} * X_{i,j}
    mu = (X_ref.T @ X_s) / max(float(n), 1.0)  # (R, P)
    ty = t_res * y_res
    t2 = t_res * t_res
    ty_sum = ty.sum()
    t2_sum = t2.sum()

    # Exact centered numerator & denominator via 3 fast (R, N) @ (N, P) GEMMs (zero (R, N, P) alloc)
    num = (X_ref * ty[:, None]).T @ X_s - mu * ty_sum  # (R, P)
    X_sq = X_s * X_s
    den_sq = (
        (X_ref * t_res[:, None]).square().T @ X_sq
        - 2.0 * mu * ((X_ref * t2[:, None]).T @ X_s)
        + (mu * mu) * t2_sum
    ).clamp_min(1e-6)
    z_abs = (num / (den_sq * y_var).sqrt()).abs().clamp(0.0, 25.0)  # (R, P)
    self_mask = ref_cols[:, None] == torch.arange(p, device=X_s.device)[None, :]
    z_abs = z_abs.masked_fill(self_mask, 0.0)
    z_abs = z_abs * valid_col.float().unsqueeze(0)

    max_z = z_abs.max(dim=0).values
    denom_r = (~self_mask).float().sum(dim=0).clamp_min(1.0)
    rms_z = (z_abs.square().sum(dim=0) / denom_r + 1e-8).sqrt() * valid_col.float()
    cm_feats = torch.stack(
        [max_z / 5.0, torch.log1p(max_z), rms_z / 5.0, torch.log1p(rms_z)], dim=-1
    )
    return cm_feats, 0.45 * torch.log1p(max_z)


class _DeepSetsColumnBlock(nn.Module):
    """O(P) permutation-equivariant global context exchange across all P columns."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(3 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        hn = self.norm(h)
        h_mean = hn.mean(dim=0, keepdim=True).expand_as(hn)
        h_max = hn.max(dim=0, keepdim=True).values.expand_as(hn)
        return h + self.mlp(torch.cat([hn, h_mean, h_max], dim=-1))


class _CorrBiasedColumnLayer(nn.Module):
    """Pairwise column self-attention on top-K candidates with empirical correlation bias."""

    def __init__(self, width: int, heads: int, *, corr_in_dim: int = 3):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.corr_in_dim = corr_in_dim
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out_proj = nn.Linear(width, width)
        self.corr_bias = nn.Sequential(nn.Linear(corr_in_dim, 16), nn.GELU(), nn.Linear(16, heads))
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, h: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        # h: (K, D), corr: (K, K) or (K, K, corr_in_dim)
        k, d = h.shape
        hn = self.norm1(h)
        q, kv_k, v = self.qkv(hn).chunk(3, dim=-1)
        q = q.view(k, self.heads, self.head_dim).transpose(0, 1)  # (H, K, Dh)
        kv_k = kv_k.view(k, self.heads, self.head_dim).transpose(0, 1)
        v = v.view(k, self.heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(q, kv_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if corr.ndim == 2:
            c_feats = torch.stack([corr, corr.abs(), corr * corr], dim=-1)  # (K, K, 3)
        else:
            c_feats = corr
        bias = self.corr_bias(c_feats).permute(2, 0, 1)  # (H, K, K)
        attn = F.softmax(scores + bias, dim=-1)
        mixed = torch.matmul(attn, v).transpose(0, 1).contiguous().view(k, d)
        h = h + self.out_proj(mixed)
        return h + self.ff(self.norm2(h))


class _TabularTwoWayLayer(nn.Module):
    """Lightweight two-way TabPFN layer: feature-slot attention + sample cross-attention."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.slot_norm = nn.LayerNorm(width)
        self.slot_attn = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.slot_ff_norm = nn.LayerNorm(width)
        self.slot_ff = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.sample_norm = nn.LayerNorm(width)
        self.sample_attn = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.sample_ff_norm = nn.LayerNorm(width)
        self.sample_ff = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def slot_mix(self, x: torch.Tensor) -> torch.Tensor:
        y = self.slot_norm(x)
        x = x + self.slot_attn(y, y, y, need_weights=False)[0]
        return x + self.slot_ff(self.slot_ff_norm(x))

    def sample_mix(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.sample_norm(x).transpose(0, 1)  # (S, N, D)
        attended = self.sample_attn(q, memory, memory, need_weights=False)[0]
        x = x + attended.transpose(0, 1)
        return x + self.sample_ff(self.sample_ff_norm(x))


class SparseTabularCausalPFN(nn.Module):
    """High-dimensional sparse tabular causal foundation model (scales to P = 2,000+)."""

    def __init__(self, cfg: TabularModelConfig = TabularModelConfig()):
        super().__init__()
        self.cfg = cfg
        c_dim = cfg.col_dim
        s_dim = cfg.study_width
        v3 = bool(cfg.use_v3_solver_basis)
        v4 = bool(cfg.use_v4_interactions)
        col_stat_dim = COL_STAT_DIM_V3 if v3 else COL_STAT_DIM
        rob_stat_dim = ROB_STAT_DIM_V4 if v4 else (ROB_STAT_DIM_V3 if v3 else ROB_STAT_DIM)
        corr_in_dim = 6 if v3 else 3
        basis_k = 4 if v3 else 2

        # 1. O(P) Round-1 Column Role Encoder & Screener
        self.col_in = nn.Sequential(
            nn.Linear(col_stat_dim, c_dim),
            nn.LayerNorm(c_dim),
            nn.GELU(),
            nn.Linear(c_dim, c_dim),
        )
        self.col_global_blocks = nn.ModuleList(
            [_DeepSetsColumnBlock(c_dim) for _ in range(3 if v3 else 2)]
        )
        self.nuis_screen_head = nn.Linear(c_dim, 1)

        # 2. Top-K_nuis Correlation-Biased Column Attention -> soft ARD nuisance gates
        self.nuis_col_layers = nn.ModuleList(
            [_CorrBiasedColumnLayer(c_dim, cfg.heads, corr_in_dim=corr_in_dim) for _ in range(cfg.col_attn_layers)]
        )
        n_nuis_gates = (3 if cfg.decouple_rlearner_propensity else 2) * basis_k
        self.nuis_gate_head = nn.Sequential(
            nn.LayerNorm(c_dim),
            nn.Linear(c_dim, c_dim // 2),
            nn.GELU(),
            nn.Linear(c_dim // 2, n_nuis_gates),
        )
        # Initialize linear gates moderately open, nonlinear gates slightly shrunk
        nn.init.constant_(self.nuis_gate_head[-1].bias, 0.0)
        if v4:
            n_pair_nuis = 3 if cfg.decouple_rlearner_propensity else 2
            self.nuis_pair_gate_head = nn.Sequential(
                nn.LayerNorm(2 * c_dim + NUIS_PAIR_STAT_DIM),
                nn.Linear(2 * c_dim + NUIS_PAIR_STAT_DIM, c_dim // 2),
                nn.GELU(),
                nn.Linear(c_dim // 2, n_pair_nuis),
            )
            nn.init.constant_(self.nuis_pair_gate_head[-1].bias, -1.5)

        # 3. Round-2 ALL-P Adjusted Robinson Updater -> Final 6-class Role Head + CATE Screener
        self.rob_update = nn.Sequential(
            nn.LayerNorm(c_dim + rob_stat_dim),
            nn.Linear(c_dim + rob_stat_dim, c_dim),
            nn.GELU(),
            nn.Linear(c_dim, c_dim),
        )
        self.rob_global_block = _DeepSetsColumnBlock(c_dim)
        if v3:
            self.rob_global_block2 = _DeepSetsColumnBlock(c_dim)
        self.role_head = nn.Sequential(nn.LayerNorm(c_dim), nn.Linear(c_dim, NUM_ROLES))
        self.mod_screen_head = nn.Linear(c_dim, 1)

        # 4. Top-K_cate Correlation-Biased Column Attention -> soft ARD CATE gates (tau)
        self.cate_col_layers = nn.ModuleList(
            [_CorrBiasedColumnLayer(c_dim, cfg.heads, corr_in_dim=corr_in_dim) for _ in range(cfg.col_attn_layers)]
        )
        self.cate_gate_head = nn.Sequential(
            nn.LayerNorm(c_dim), nn.Linear(c_dim, c_dim // 2), nn.GELU(), nn.Linear(c_dim // 2, basis_k)
        )
        nn.init.constant_(self.cate_gate_head[-1].bias, -0.5)
        if v4:
            self.cate_pair_gate_head = nn.Sequential(
                nn.LayerNorm(2 * c_dim + CATE_PAIR_STAT_DIM),
                nn.Linear(2 * c_dim + CATE_PAIR_STAT_DIM, c_dim // 2),
                nn.GELU(),
                nn.Linear(c_dim // 2, 1),
            )
            nn.init.constant_(self.cate_pair_gate_head[-1].bias, -1.5)

        # 5. Learnable LOO-CV Ridge penalty grids for e, m, tau
        k_grid = cfg.ridge_grid_size
        if v3:
            base_grid = torch.linspace(-0.5, 11.5, k_grid)
            tau_grid = torch.linspace(0.2, 12.0, k_grid)
        else:
            base_grid = torch.linspace(0.2, 0.2 + 1.5 * (k_grid - 1), k_grid)
            tau_grid = torch.linspace(0.8, 0.8 + 1.5 * (k_grid - 1), k_grid)
        self.log_lam_e = nn.Parameter(base_grid.clone())
        if cfg.decouple_rlearner_propensity:
            self.log_lam_er = nn.Parameter(base_grid.clone())
        self.log_lam_m = nn.Parameter(base_grid.clone())
        self.log_lam_tau = nn.Parameter(tau_grid.clone())
        self.log_cv_temp = nn.Parameter(torch.tensor(1.5))

        # 6. Calibration & Epistemic-Uncertainty Heads on ARD R-learner outputs + leverages
        cal_arm_h = 64 if v3 else 32
        cal_prop_h = 32 if v3 else 16
        self.arm_cal = nn.Sequential(nn.Linear(8, cal_arm_h), nn.GELU(), nn.Linear(cal_arm_h, 4))
        self.prop_cal = nn.Sequential(nn.Linear(4, cal_prop_h), nn.GELU(), nn.Linear(cal_prop_h, 1))
        nn.init.normal_(self.arm_cal[-1].weight, std=0.01)
        nn.init.zeros_(self.arm_cal[-1].bias)
        nn.init.normal_(self.prop_cal[-1].weight, std=0.01)
        nn.init.zeros_(self.prop_cal[-1].bias)

        # 7. Compact Residual Two-Way TabPFN Study Layers over top-attn_slots gated features
        cell_in_dim = 8 if v3 else 6
        self.cell_proj = nn.Sequential(
            nn.Linear(cell_in_dim, s_dim), nn.GELU(), nn.Linear(s_dim, s_dim)
        )
        self.col_to_study = nn.Linear(c_dim, s_dim)
        self.outcome_token = nn.Linear(6, s_dim)
        self.study_layers = nn.ModuleList(
            [_TabularTwoWayLayer(s_dim, cfg.heads) for _ in range(cfg.study_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(s_dim), nn.Linear(s_dim, cfg.bins))
        self.propensity_head = nn.Sequential(nn.LayerNorm(s_dim), nn.Linear(s_dim, 1))
        self.attn_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.prop_residual_scale = nn.Parameter(torch.tensor(0.1))

        centers = _sinh_grid(cfg.bins)
        spacings = centers[1:] - centers[:-1]
        local_width = torch.cat(
            [spacings[:1], 0.5 * (spacings[:-1] + spacings[1:]), spacings[-1:]]
        )
        self.register_buffer("centers", centers)
        self.register_buffer("bin_widths", local_width)

    @staticmethod
    def _empirical_corr(X_sel: torch.Tensor) -> torch.Tensor:
        Xn = F.normalize(X_sel - X_sel.mean(dim=0, keepdim=True), p=2, dim=0)
        return (Xn.T @ Xn).clamp(-1.0, 1.0)

    @staticmethod
    def _pairwise_corr_features(X_sel: torch.Tensor, M_sel: torch.Tensor) -> torch.Tensor:
        """6-d smooth pairwise correlation: zero-imputed + co-observed pairwise-complete + overlap."""
        obs = (1.0 - M_sel).clamp(0.0, 1.0)
        Xn = F.normalize(X_sel - X_sel.mean(dim=0, keepdim=True), p=2, dim=0)
        corr_x = (Xn.T @ Xn).clamp(-1.0, 1.0)
        co_n = obs.T @ obs
        num = X_sel.T @ X_sel
        den_i = (X_sel * X_sel).T @ obs
        den = (den_i * den_i.T).clamp_min(1e-6).sqrt()
        corr_pair = (num / den).clamp(-1.0, 1.0)
        conf = co_n / (co_n + 8.0)
        corr_coobs = corr_pair * conf
        overlap = (co_n / max(float(len(X_sel)), 1.0)).clamp(0.0, 1.0)
        return torch.stack(
            [corr_x, corr_x.abs(), corr_x * corr_x, corr_coobs, corr_coobs.abs(), overlap],
            dim=-1,
        )

    def _build_patient_slots(
        self,
        X_s: torch.Tensor,
        X2_s: torch.Tensor,
        M_s: torch.Tensor,
        mem: TabularMemory,
        *,
        X_sat: torch.Tensor | None = None,
        M_c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build (N, attn_slots, study_width) cell embeddings on top gated columns."""
        n = len(X_s)
        idx = mem.attn_idx
        k_act = len(idx)
        x_sel = X_s[:, idx]  # (N, k_act)
        x2_sel = X2_s[:, idx]
        m_sel = M_s[:, idx]
        g_feats = mem.attn_gate_feats.unsqueeze(0).expand(n, -1, -1)  # (N, k_act, 3)
        if self.cfg.use_v3_solver_basis and X_sat is not None and M_c is not None:
            cell_in = torch.cat(
                [
                    x_sel.unsqueeze(-1),
                    x2_sel.unsqueeze(-1),
                    X_sat[:, idx].unsqueeze(-1),
                    m_sel.unsqueeze(-1),
                    M_c[:, idx].unsqueeze(-1),
                    g_feats,
                ],
                dim=-1,
            )
        else:
            cell_in = torch.cat(
                [x_sel.unsqueeze(-1), x2_sel.unsqueeze(-1), m_sel.unsqueeze(-1), g_feats], dim=-1
            )
        col_ctx = self.col_to_study(mem.col_embs[idx]).unsqueeze(0)  # (1, k_act, s_dim)
        slots = self.cell_proj(cell_in) + col_ctx
        if k_act < self.cfg.attn_slots:
            pad = slots.new_zeros((n, self.cfg.attn_slots - k_act, self.cfg.study_width))
            slots = torch.cat([slots, pad], dim=1)
        return slots

    def _assemble_context(self, slots, t, y, scale):
        arm = t.long()
        standardized = (y - scale.rate[arm]) / scale.sd[arm]
        base_logit = scale.rate.logit()[arm]
        log_n = torch.log1p(scale.counts[arm])
        ones = torch.ones_like(t)
        token_in = torch.stack([t, standardized, base_logit, log_n, ones, ones], dim=-1)
        tok = self.outcome_token(token_in).unsqueeze(1)
        return torch.cat([slots, tok], dim=1)

    def _assemble_arm_queries(self, slots, scale):
        n_q = len(slots)
        s_exp = slots[:, None, :, :].expand(-1, 2, -1, -1)
        arms = slots.new_tensor([0.0, 1.0]).expand(n_q, 2)
        zeros = torch.zeros_like(arms)
        ones = torch.ones_like(arms)
        base_logit = scale.rate.logit()[None, :].expand(n_q, 2)
        log_n = torch.log1p(scale.counts)[None, :].expand(n_q, 2)
        token_in = torch.stack([arms, zeros, base_logit, log_n, zeros, ones], dim=-1)
        tok = self.outcome_token(token_in).unsqueeze(2)
        return torch.cat([s_exp, tok], dim=2).flatten(0, 1)

    def _assemble_propensity_queries(self, slots, scale):
        n_q = len(slots)
        total_n = scale.counts.sum()
        tx_rate = (scale.counts[1] + 0.5) / (total_n + 1.0)
        half = slots.new_full((n_q,), 0.5)
        zeros = slots.new_zeros((n_q,))
        base_logit = tx_rate.logit().expand(n_q)
        log_n = torch.log1p(total_n).expand(n_q)
        token_in = torch.stack([half, zeros, base_logit, log_n, zeros, zeros], dim=-1)
        tok = self.outcome_token(token_in).unsqueeze(1)
        return torch.cat([slots, tok], dim=1)

    def _attend_queries(self, queries, attn_memories):
        use_bf16 = queries.is_cuda and torch.cuda.is_bf16_supported()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            for layer, memory in zip(self.study_layers, attn_memories, strict=True):
                queries = layer.slot_mix(queries)
                queries = layer.sample_mix(queries, memory)
        return queries[:, -1, :].float()

    @staticmethod
    @torch.no_grad()
    def _nuis_pair_stats(
        P_nuis: torch.Tensor,
        t_f: torch.Tensor,
        y_f: torch.Tensor,
        corr_ab: torch.Tensor,
        p_valid: torch.Tensor,
    ) -> torch.Tensor:
        """6-d strictly symmetric empirical descriptors for Round-1 nuisance pairs."""
        m0 = t_f < 0.5
        m1 = ~m0
        n0 = m0.float().sum().clamp_min(2.0)
        n1 = m1.float().sum().clamp_min(2.0)
        r_t, t_t = _corr_and_tstat(P_nuis, t_f)
        r_y0, t_y0 = _corr_and_tstat(P_nuis, y_f, mask=m0)
        r_y1, t_y1 = _corr_and_tstat(P_nuis, y_f, mask=m1)
        if int(m0.sum().item()) < 3:
            r_y0.zero_(); t_y0.zero_()
        if int(m1.sum().item()) < 3:
            r_y1.zero_(); t_y1.zero_()
        se_diff = (1.0 / (n0 - 1.0).clamp_min(1.0) + 1.0 / (n1 - 1.0).clamp_min(1.0)).sqrt()
        z_inter = ((r_y1 - r_y0) / se_diff).clamp(-25.0, 25.0)
        stats = torch.stack(
            [r_t, t_t / 5.0, t_y0 / 5.0, t_y1 / 5.0, z_inter / 5.0, corr_ab], dim=-1
        )
        return stats * p_valid.unsqueeze(-1)

    @staticmethod
    @torch.no_grad()
    def _cate_pair_stats(
        P_cate: torch.Tensor,
        t_res: torch.Tensor,
        y_res: torch.Tensor,
        corr_ab: torch.Tensor,
        p_valid: torch.Tensor,
    ) -> torch.Tensor:
        """4-d strictly symmetric Robinson interaction descriptors for Round-2 CATE pairs."""
        y_var = y_res.square().mean().clamp_min(1e-5)
        zt = P_cate * t_res[:, None]
        z_rob = (
            (zt * y_res[:, None]).sum(dim=0) / (zt.square().sum(dim=0).clamp_min(1e-6) * y_var).sqrt()
        ).clamp(-25.0, 25.0) * p_valid
        abs_z = z_rob.abs()
        stats = torch.stack([z_rob / 5.0, abs_z / 5.0, torch.log1p(abs_z), corr_ab], dim=-1)
        return stats * p_valid.unsqueeze(-1)

    def condition(
        self,
        X_ctx: torch.Tensor,
        t_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
        *,
        M_ctx: torch.Tensor | None = None,
    ) -> tuple[TabularMemory, ArmScale]:
        """Condition on observational context table ``(X_ctx, t_ctx, y_ctx)`` (zero oracle info)."""
        device_type = X_ctx.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            t_f = t_ctx.float()
            y_f = y_ctx.float()
            n_ctx, p = X_ctx.shape
            scale = ArmScale.from_context(t_f, y_f)

            # 1. Context-only standardization & O(N * P) observational column stats
            v3 = bool(self.cfg.use_v3_solver_basis)
            v4 = bool(self.cfg.use_v4_interactions)
            X_s, X2_s, M_s, col_mean, col_sd, x2_mean, x2_sd, valid_col = _standardize_context(
                X_ctx, M_ctx
            )
            X_sat = M_c = xsat_mean = xsat_sd = miss_mean = miss_sd = miss_weight = None
            if v3:
                obs_ctx = torch.isfinite(X_ctx.float()).float()
                X_sat, M_c, xsat_mean, xsat_sd, miss_mean, miss_sd, miss_weight = (
                    _compute_v3_basis_context(X_s, M_s, obs_ctx, valid_col)
                )
            col_stats, raw_nuis_prio = _column_sufficient_stats(
                X_s, X2_s, M_s, t_f, y_f, valid_col, X_sat=X_sat, M_c=M_c
            )

            # 2. O(P) Round-1 Column Role Encoder & Nuisance Screening
            h_col = self.col_in(col_stats)
            for blk in self.col_global_blocks:
                h_col = blk(h_col)
            nuis_screen_logits = self.nuis_screen_head(h_col).squeeze(-1)
            nuis_score = raw_nuis_prio + 1.5 * nuis_screen_logits
            nuis_score = torch.where(valid_col, nuis_score, nuis_score.new_full((), -1e5))

            k1 = min(p, self.cfg.top_k_nuis)
            nuis_idx = torch.topk(nuis_score, k1, largest=True).indices  # canonical score order

            # 3. Correlation-Biased Column Attention on Top-K1 Nuisance Columns -> ARD Gates
            h_nuis = h_col[nuis_idx]
            X_nuis = X_s[:, nuis_idx]
            X2_nuis = X2_s[:, nuis_idx]
            corr_nuis = (
                self._pairwise_corr_features(X_nuis, M_s[:, nuis_idx])
                if v3
                else self._empirical_corr(X_nuis)
            )
            for layer in self.nuis_col_layers:
                h_nuis = layer(h_nuis, corr_nuis)
            nuis_gates = torch.sigmoid(self.nuis_gate_head(h_nuis)).clamp(1e-3, 1.0)
            v_nuis = valid_col[nuis_idx].float()

            g_e_sat = g_e_miss = g_er_sat = g_er_miss = g_m_sat = g_m_miss = None
            if v3:
                if self.cfg.decouple_rlearner_propensity:
                    (
                        g_e_lin, g_e_nl, g_e_sat, g_e_miss,
                        g_er_lin, g_er_nl, g_er_sat, g_er_miss,
                        g_m_lin, g_m_nl, g_m_sat, g_m_miss,
                    ) = nuis_gates.unbind(dim=-1)
                    g_er_lin = g_er_lin * v_nuis
                    g_er_nl = g_er_nl * v_nuis
                    g_er_sat = g_er_sat * v_nuis
                    g_er_miss = g_er_miss * v_nuis
                else:
                    (
                        g_e_lin, g_e_nl, g_e_sat, g_e_miss,
                        g_m_lin, g_m_nl, g_m_sat, g_m_miss,
                    ) = nuis_gates.unbind(dim=-1)
                g_e_lin = g_e_lin * v_nuis
                g_e_nl = g_e_nl * v_nuis
                g_e_sat = g_e_sat * v_nuis
                g_e_miss = g_e_miss * v_nuis
                g_m_lin = g_m_lin * v_nuis
                g_m_nl = g_m_nl * v_nuis
                g_m_sat = g_m_sat * v_nuis
                g_m_miss = g_m_miss * v_nuis
            else:
                if self.cfg.decouple_rlearner_propensity:
                    g_e_lin, g_e_nl, g_er_lin, g_er_nl, g_m_lin, g_m_nl = nuis_gates.unbind(dim=-1)
                    g_er_lin = g_er_lin * v_nuis
                    g_er_nl = g_er_nl * v_nuis
                else:
                    g_e_lin, g_e_nl, g_m_lin, g_m_nl = nuis_gates.unbind(dim=-1)
                g_e_lin = g_e_lin * v_nuis
                g_e_nl = g_e_nl * v_nuis
                g_m_lin = g_m_lin * v_nuis
                g_m_nl = g_m_nl * v_nuis

            # Optional v4 Round-1 symmetric pairwise interaction basis on top pair_top_k confounders/prognostics
            nuis_pair_a = nuis_pair_b = nuis_pair_mean = nuis_pair_sd = nuis_pair_weight = nuis_pair_valid = None
            g_pe = g_per = g_pm = P_nuis = None
            g_m_r1_max = g_er_r1_max = None
            if v4:
                g_m_r1_max = torch.stack([g_m_lin, g_m_nl, g_m_sat, g_m_miss], dim=-1).max(dim=-1).values
                g_er_r1_max = (
                    torch.stack([g_er_lin, g_er_nl, g_er_sat, g_er_miss], dim=-1).max(dim=-1).values
                    if self.cfg.decouple_rlearner_propensity
                    else torch.stack([g_e_lin, g_e_nl, g_e_sat, g_e_miss], dim=-1).max(dim=-1).values
                )
                if k1 >= 2:
                    r1_pair_prio = (
                        g_m_r1_max.detach()
                        + 0.35 * g_er_r1_max.detach()
                        + 1e-3 * raw_nuis_prio[nuis_idx].clamp(-10.0, 10.0)
                    )
                    r1_pair_prio = torch.where(valid_col[nuis_idx], r1_pair_prio, r1_pair_prio.new_full((), -1e5))
                    k_pn = min(k1, self.cfg.pair_top_k)
                    pn_loc = torch.topk(r1_pair_prio, k_pn, largest=True).indices
                    pa_i, pb_i = torch.triu_indices(k_pn, k_pn, offset=1, device=X_s.device)
                    pa_nuis = pn_loc[pa_i]
                    pb_nuis = pn_loc[pb_i]
                    nuis_pair_a = nuis_idx[pa_nuis]
                    nuis_pair_b = nuis_idx[pb_nuis]
                    P_nuis, nuis_pair_mean, nuis_pair_sd, nuis_pair_weight, nuis_pair_valid = (
                        _build_pairs_context(X_s, valid_col, nuis_pair_a, nuis_pair_b)
                    )
                    corr_ab_n = corr_nuis[pa_nuis, pb_nuis, 0]
                    nuis_p_stats = self._nuis_pair_stats(P_nuis, t_f, y_f, corr_ab_n, nuis_pair_valid)
                    ha = h_nuis[pa_nuis]
                    hb = h_nuis[pb_nuis]
                    nuis_p_in = torch.cat(
                        [ha + hb, ((ha - hb).square() + 1e-6).sqrt(), nuis_p_stats], dim=-1
                    )
                    pair_n_gates = torch.sigmoid(self.nuis_pair_gate_head(nuis_p_in)).clamp(1e-3, 1.0)
                    if self.cfg.decouple_rlearner_propensity:
                        g_pe, g_per, g_pm = pair_n_gates.unbind(dim=-1)
                        g_per = g_per * nuis_pair_valid
                    else:
                        g_pe, g_pm = pair_n_gates.unbind(dim=-1)
                        g_per = g_pe * nuis_pair_valid
                    g_pe = g_pe * nuis_pair_valid
                    g_pm = g_pm * nuis_pair_valid

            # 4. Differentiable ARD Closed-Form LOO-CV Nuisances (e_loo, e_r_loo, m_loo)
            ones = torch.ones((n_ctx, 1), device=X_s.device, dtype=torch.float32)
            lam_hi = 5e5 if v3 else 1e5
            if v3:
                Xsat_nuis = X_sat[:, nuis_idx]
                Mc_nuis = M_c[:, nuis_idx]
                e_parts = [
                    ones,
                    X_nuis * g_e_lin[None, :],
                    X2_nuis * g_e_nl[None, :],
                    Xsat_nuis * g_e_sat[None, :],
                    Mc_nuis * g_e_miss[None, :],
                ]
                m_parts = [
                    ones,
                    X_nuis * g_m_lin[None, :],
                    X2_nuis * g_m_nl[None, :],
                    Xsat_nuis * g_m_sat[None, :],
                    Mc_nuis * g_m_miss[None, :],
                ]
                if P_nuis is not None:
                    e_parts.append(P_nuis * g_pe[None, :])
                    m_parts.append(P_nuis * g_pm[None, :])
                A_e = torch.cat(e_parts, dim=1)
                A_m = torch.cat(m_parts, dim=1)
            else:
                A_e = torch.cat([ones, X_nuis * g_e_lin[None, :], X2_nuis * g_e_nl[None, :]], dim=1)
                A_m = torch.cat([ones, X_nuis * g_m_lin[None, :], X2_nuis * g_m_nl[None, :]], dim=1)
            cv_temp = F.softplus(self.log_cv_temp.float()) + 0.1

            lam_e = self.log_lam_e.float().exp().clamp(0.05, lam_hi)
            prop_fit, e_loo = _fit_ard_loo_cv_ridge(A_e, t_f, lam_e, cv_temp)
            e_loo = e_loo.clamp(0.02, 0.98)

            if self.cfg.decouple_rlearner_propensity:
                if v3:
                    er_parts = [
                        ones,
                        X_nuis * g_er_lin[None, :],
                        X2_nuis * g_er_nl[None, :],
                        Xsat_nuis * g_er_sat[None, :],
                        Mc_nuis * g_er_miss[None, :],
                    ]
                    if P_nuis is not None:
                        er_parts.append(P_nuis * g_per[None, :])
                    A_er = torch.cat(er_parts, dim=1)
                else:
                    A_er = torch.cat(
                        [ones, X_nuis * g_er_lin[None, :], X2_nuis * g_er_nl[None, :]], dim=1
                    )
                lam_er = self.log_lam_er.float().exp().clamp(0.05, lam_hi)
                r_prop_fit, e_r_loo = _fit_ard_loo_cv_ridge(A_er, t_f, lam_er, cv_temp)
                e_r_loo = e_r_loo.clamp(0.02, 0.98)
            else:
                g_er_lin, g_er_nl = g_e_lin, g_e_nl
                g_er_sat, g_er_miss = g_e_sat, g_e_miss
                r_prop_fit, e_r_loo = prop_fit, e_loo

            lam_m = self.log_lam_m.float().exp().clamp(0.05, lam_hi)
            out_fit, m_loo = _fit_ard_loo_cv_ridge(A_m, y_f, lam_m, cv_temp)

            # 5. Round-2 ALL-P Adjusted Robinson Interaction Scan -> Role Posteriors & CATE Set
            t_res = t_f - e_r_loo
            y_res = y_f - m_loo
            rob_feats, raw_mod_prio = _robinson_all_p_stats(
                X_s, X2_s, t_res, y_res, valid_col, X_sat=X_sat, M_c=M_c
            )
            if v4:
                r_ref = min(8, k1)
                ref_score = (
                    g_m_r1_max.detach()
                    + 0.5 * g_er_r1_max.detach()
                    + 1e-3 * raw_nuis_prio[nuis_idx].clamp(-10.0, 10.0)
                )
                ref_score = torch.where(valid_col[nuis_idx], ref_score, ref_score.new_full((), -1e5))
                ref_cols = nuis_idx[torch.topk(ref_score, r_ref, largest=True).indices]
                cm_feats, cm_prio = _robinson_confounder_mod_stats(
                    X_s, t_res, y_res, valid_col, ref_cols
                )
                rob_feats = torch.cat([rob_feats, cm_feats], dim=-1)
                raw_mod_prio = torch.where(
                    valid_col, raw_mod_prio + cm_prio, raw_mod_prio.new_full((), -1e4)
                )
            h_col2 = h_col + self.rob_update(torch.cat([h_col, rob_feats], dim=-1))
            h_col2 = self.rob_global_block(h_col2)
            if v3:
                h_col2 = self.rob_global_block2(h_col2)
            role_logits = self.role_head(h_col2)  # (P, 6)

            mod_screen_logits = self.mod_screen_head(h_col2).squeeze(-1)
            mod_score = raw_mod_prio + 1.5 * mod_screen_logits
            mod_score = torch.where(valid_col, mod_score, mod_score.new_full((), -1e5))
            k2 = min(p, self.cfg.top_k_cate)
            cate_idx = torch.topk(mod_score, k2, largest=True).indices

            # 6. Correlation-Biased Attention on Top-K2 CATE Columns -> ARD Robinson R-Learner
            h_cate = h_col2[cate_idx]
            X_cate = X_s[:, cate_idx]
            X2_cate = X2_s[:, cate_idx]
            corr_cate = (
                self._pairwise_corr_features(X_cate, M_s[:, cate_idx])
                if v3
                else self._empirical_corr(X_cate)
            )
            for layer in self.cate_col_layers:
                h_cate = layer(h_cate, corr_cate)
            cate_gates = torch.sigmoid(self.cate_gate_head(h_cate)).clamp(1e-3, 1.0)
            v_cate = valid_col[cate_idx].float()
            g_tau_sat = g_tau_miss = None
            cate_pair_a = cate_pair_b = cate_pair_mean = cate_pair_sd = cate_pair_weight = cate_pair_valid = None
            g_ptau = P_cate = None

            full_ge = torch.zeros(p, device=X_s.device, dtype=torch.float32)
            full_gm = torch.zeros(p, device=X_s.device, dtype=torch.float32)
            full_gt = torch.zeros(p, device=X_s.device, dtype=torch.float32)
            if v3:
                g_tau_lin, g_tau_nl, g_tau_sat, g_tau_miss = cate_gates.unbind(dim=-1)
                g_tau_lin = g_tau_lin * v_cate
                g_tau_nl = g_tau_nl * v_cate
                g_tau_sat = g_tau_sat * v_cate
                g_tau_miss = g_tau_miss * v_cate
                full_ge[nuis_idx] = torch.stack([g_er_lin, g_er_nl, g_er_sat, g_er_miss], dim=-1).max(dim=-1).values
                full_gm[nuis_idx] = torch.stack([g_m_lin, g_m_nl, g_m_sat, g_m_miss], dim=-1).max(dim=-1).values
                full_gt[cate_idx] = torch.stack([g_tau_lin, g_tau_nl, g_tau_sat, g_tau_miss], dim=-1).max(dim=-1).values
                tau_parts = [
                    ones,
                    X_cate * g_tau_lin[None, :],
                    X2_cate * g_tau_nl[None, :],
                    X_sat[:, cate_idx] * g_tau_sat[None, :],
                    M_c[:, cate_idx] * g_tau_miss[None, :],
                ]
                if v4 and p >= 2:
                    # Select top pair_top_k across modifiers, dual_role, confounders, & prognostics
                    pair_tau_prio = (
                        1.5 * full_gt.detach()
                        + 0.9 * full_gm.detach()
                        + 0.5 * full_ge.detach()
                        + 1e-3 * (raw_nuis_prio.clamp(-10.0, 10.0) + raw_mod_prio.clamp(-10.0, 10.0))
                    )
                    pair_tau_prio = torch.where(valid_col, pair_tau_prio, pair_tau_prio.new_full((), -1e5))
                    k_pc = min(p, self.cfg.pair_top_k)
                    cp_nodes = torch.topk(pair_tau_prio, k_pc, largest=True).indices
                    ca_i, cb_i = torch.triu_indices(k_pc, k_pc, offset=1, device=X_s.device)
                    cate_pair_a = cp_nodes[ca_i]
                    cate_pair_b = cp_nodes[cb_i]
                    P_cate, cate_pair_mean, cate_pair_sd, cate_pair_weight, cate_pair_valid = (
                        _build_pairs_context(X_s, valid_col, cate_pair_a, cate_pair_b)
                    )
                    corr_cp = self._empirical_corr(X_s[:, cp_nodes])
                    corr_ab_c = corr_cp[ca_i, cb_i]
                    cate_p_stats = self._cate_pair_stats(
                        P_cate, t_res, y_res, corr_ab_c, cate_pair_valid
                    )
                    h2a = h_col2[cate_pair_a]
                    h2b = h_col2[cate_pair_b]
                    cate_p_in = torch.cat(
                        [h2a + h2b, ((h2a - h2b).square() + 1e-6).sqrt(), cate_p_stats], dim=-1
                    )
                    g_ptau = (
                        torch.sigmoid(self.cate_pair_gate_head(cate_p_in)).squeeze(-1).clamp(1e-3, 1.0)
                        * cate_pair_valid
                    )
                    tau_parts.append(P_cate * g_ptau[None, :])
                A_tau = torch.cat(tau_parts, dim=1)
            else:
                g_tau_lin, g_tau_nl = cate_gates.unbind(dim=-1)
                g_tau_lin = g_tau_lin * v_cate
                g_tau_nl = g_tau_nl * v_cate
                full_ge[nuis_idx] = torch.maximum(g_er_lin, g_er_nl)
                full_gm[nuis_idx] = torch.maximum(g_m_lin, g_m_nl)
                full_gt[cate_idx] = torch.maximum(g_tau_lin, g_tau_nl)
                A_tau = torch.cat(
                    [ones, X_cate * g_tau_lin[None, :], X2_cate * g_tau_nl[None, :]], dim=1
                )
            B_ctx = t_res.unsqueeze(-1) * A_tau
            lam_tau = self.log_lam_tau.float().exp().clamp(0.05, lam_hi)
            cate_fit, _ = _fit_ard_loo_cv_ridge(B_ctx, y_res, lam_tau, cv_temp, intercept_reg=0.05)

            # 7. Select top attn_slots columns across (e_r, m, tau) gates for residual Two-Way TabPFN
            if v4 and g_ptau is not None and len(g_ptau) > 0:
                pair_gt_node = torch.zeros(p, device=X_s.device, dtype=torch.float32)
                pair_gt_node.scatter_reduce_(0, cate_pair_a, g_ptau.detach(), reduce="amax", include_self=True)
                pair_gt_node.scatter_reduce_(0, cate_pair_b, g_ptau.detach(), reduce="amax", include_self=True)
                full_gt = torch.maximum(full_gt, pair_gt_node)
            attn_prio = (
                full_gm + 1.5 * full_gt + 0.5 * full_ge + 1e-3 * (raw_nuis_prio.clamp(-10, 10) + raw_mod_prio.clamp(-10, 10))
            )
            attn_prio = torch.where(valid_col, attn_prio, attn_prio.new_full((), -1e5))
            k_attn = min(p, self.cfg.attn_slots)
            attn_idx = torch.topk(attn_prio, k_attn, largest=True).indices
            attn_gate_feats = torch.stack(
                [full_ge[attn_idx], full_gm[attn_idx], full_gt[attn_idx]], dim=-1
            )
            if self.cfg.decouple_rlearner_propensity:
                # Keep prop_loss (via attn_prop) completely decoupled from g_er
                attn_gate_feats = attn_gate_feats.detach()

            tmp_mem = TabularMemory(
                col_mean=col_mean,
                col_sd=col_sd,
                x2_mean=x2_mean,
                x2_sd=x2_sd,
                valid_col=valid_col,
                col_embs=h_col2,
                role_logits=role_logits,
                nuis_screen_logits=nuis_screen_logits,
                mod_screen_logits=mod_screen_logits,
                nuis_idx=nuis_idx,
                cate_idx=cate_idx,
                attn_idx=attn_idx,
                g_e_lin=g_e_lin,
                g_e_nl=g_e_nl,
                g_er_lin=g_er_lin,
                g_er_nl=g_er_nl,
                g_m_lin=g_m_lin,
                g_m_nl=g_m_nl,
                g_tau_lin=g_tau_lin,
                g_tau_nl=g_tau_nl,
                attn_gate_feats=attn_gate_feats,
                propensity_fit=prop_fit,
                rlearner_prop_fit=r_prop_fit,
                outcome_fit=out_fit,
                cate_fit=cate_fit,
                attn_memories=[],
                n_ctx=n_ctx,
                n_features=p,
                xsat_mean=xsat_mean,
                xsat_sd=xsat_sd,
                miss_mean=miss_mean,
                miss_sd=miss_sd,
                miss_weight=miss_weight,
                g_e_sat=g_e_sat,
                g_e_miss=g_e_miss,
                g_er_sat=g_er_sat,
                g_er_miss=g_er_miss,
                g_m_sat=g_m_sat,
                g_m_miss=g_m_miss,
                g_tau_sat=g_tau_sat,
                g_tau_miss=g_tau_miss,
                nuis_pair_a=nuis_pair_a,
                nuis_pair_b=nuis_pair_b,
                nuis_pair_mean=nuis_pair_mean,
                nuis_pair_sd=nuis_pair_sd,
                nuis_pair_weight=nuis_pair_weight,
                nuis_pair_valid=nuis_pair_valid,
                g_pe=g_pe,
                g_per=g_per,
                g_pm=g_pm,
                cate_pair_a=cate_pair_a,
                cate_pair_b=cate_pair_b,
                cate_pair_mean=cate_pair_mean,
                cate_pair_sd=cate_pair_sd,
                cate_pair_weight=cate_pair_weight,
                cate_pair_valid=cate_pair_valid,
                g_ptau=g_ptau,
            )

            ctx_slots = self._build_patient_slots(X_s, X2_s, M_s, tmp_mem, X_sat=X_sat, M_c=M_c)
            context = self._assemble_context(ctx_slots, t_f, y_f, scale)
            attn_memories = []
            use_bf16 = context.is_cuda and torch.cuda.is_bf16_supported()
            use_ckpt = self.training and v4 and n_ctx > 4000
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                for i, layer in enumerate(self.study_layers):
                    if use_ckpt:
                        context = checkpoint(layer.slot_mix, context, use_reentrant=False)
                    else:
                        context = layer.slot_mix(context)
                    memory = layer.sample_norm(context).transpose(0, 1)  # (S, N_ctx, D)
                    attn_memories.append(memory)
                    if i + 1 < len(self.study_layers):
                        if use_ckpt:
                            context = checkpoint(
                                layer.sample_mix, context, memory, use_reentrant=False
                            )
                        else:
                            context = layer.sample_mix(context, memory)
            tmp_mem.attn_memories = attn_memories
            return tmp_mem, scale

    def _query_solver_stats(
        self, X_q: torch.Tensor, M_q: torch.Tensor | None, mem: TabularMemory
    ):
        v3 = bool(self.cfg.use_v3_solver_basis)
        X_s, X2_s, M_s = _apply_standardize(
            X_q, M_q, mem.col_mean, mem.col_sd, mem.x2_mean, mem.x2_sd, mem.valid_col
        )
        X_sat = M_c = None
        if v3:
            obs_q = torch.isfinite(X_q.float()).float()
            X_sat, M_c = _apply_v3_basis(
                X_s,
                M_s,
                obs_q,
                mem.valid_col,
                mem.xsat_mean,
                mem.xsat_sd,
                mem.miss_mean,
                mem.miss_sd,
                mem.miss_weight,
            )
        n_q = len(X_s)
        ones_q = torch.ones((n_q, 1), device=X_s.device, dtype=torch.float32)

        X_n = X_s[:, mem.nuis_idx]
        X2_n = X2_s[:, mem.nuis_idx]
        X_c = X_s[:, mem.cate_idx]
        X2_c = X2_s[:, mem.cate_idx]
        P_nuis_q = P_cate_q = None
        if mem.nuis_pair_a is not None:
            P_nuis_q = _apply_pairs(
                X_s,
                mem.nuis_pair_a,
                mem.nuis_pair_b,
                mem.nuis_pair_mean,
                mem.nuis_pair_sd,
                mem.nuis_pair_weight,
                mem.nuis_pair_valid,
            )
        if mem.cate_pair_a is not None:
            P_cate_q = _apply_pairs(
                X_s,
                mem.cate_pair_a,
                mem.cate_pair_b,
                mem.cate_pair_mean,
                mem.cate_pair_sd,
                mem.cate_pair_weight,
                mem.cate_pair_valid,
            )
        if v3:
            Xsat_n = X_sat[:, mem.nuis_idx]
            Mc_n = M_c[:, mem.nuis_idx]
            e_parts_q = [
                ones_q,
                X_n * mem.g_e_lin[None, :],
                X2_n * mem.g_e_nl[None, :],
                Xsat_n * mem.g_e_sat[None, :],
                Mc_n * mem.g_e_miss[None, :],
            ]
            m_parts_q = [
                ones_q,
                X_n * mem.g_m_lin[None, :],
                X2_n * mem.g_m_nl[None, :],
                Xsat_n * mem.g_m_sat[None, :],
                Mc_n * mem.g_m_miss[None, :],
            ]
            t_parts_q = [
                ones_q,
                X_c * mem.g_tau_lin[None, :],
                X2_c * mem.g_tau_nl[None, :],
                X_sat[:, mem.cate_idx] * mem.g_tau_sat[None, :],
                M_c[:, mem.cate_idx] * mem.g_tau_miss[None, :],
            ]
            if P_nuis_q is not None:
                e_parts_q.append(P_nuis_q * mem.g_pe[None, :])
                m_parts_q.append(P_nuis_q * mem.g_pm[None, :])
            if P_cate_q is not None:
                t_parts_q.append(P_cate_q * mem.g_ptau[None, :])
            A_e_q = torch.cat(e_parts_q, dim=1)
            A_m_q = torch.cat(m_parts_q, dim=1)
            A_t_q = torch.cat(t_parts_q, dim=1)
        else:
            A_e_q = torch.cat([ones_q, X_n * mem.g_e_lin[None, :], X2_n * mem.g_e_nl[None, :]], dim=1)
            A_m_q = torch.cat([ones_q, X_n * mem.g_m_lin[None, :], X2_n * mem.g_m_nl[None, :]], dim=1)
            A_t_q = torch.cat(
                [ones_q, X_c * mem.g_tau_lin[None, :], X2_c * mem.g_tau_nl[None, :]], dim=1
            )

        e_q, h_e = _predict_ridge(mem.propensity_fit, A_e_q)
        m_q, h_m = _predict_ridge(mem.outcome_fit, A_m_q)
        tau_q, h_tau = _predict_ridge(mem.cate_fit, A_t_q)
        e_prob = e_q.clamp(0.01, 0.99)
        if self.cfg.decouple_rlearner_propensity:
            if v3:
                er_parts_q = [
                    ones_q,
                    X_n * mem.g_er_lin[None, :],
                    X2_n * mem.g_er_nl[None, :],
                    Xsat_n * mem.g_er_sat[None, :],
                    Mc_n * mem.g_er_miss[None, :],
                ]
                if P_nuis_q is not None:
                    er_parts_q.append(P_nuis_q * mem.g_per[None, :])
                A_er_q = torch.cat(er_parts_q, dim=1)
            else:
                A_er_q = torch.cat(
                    [ones_q, X_n * mem.g_er_lin[None, :], X2_n * mem.g_er_nl[None, :]], dim=1
                )
            e_r_q, _ = _predict_ridge(mem.rlearner_prop_fit, A_er_q)
            e_r_prob = e_r_q.clamp(0.01, 0.99)
        else:
            e_r_prob = e_prob
        y0_raw = (m_q - e_r_prob * tau_q).clamp(0.01, 0.99)
        y1_raw = (m_q + (1.0 - e_r_prob) * tau_q).clamp(0.01, 0.99)
        return X_s, X2_s, M_s, X_sat, M_c, y0_raw, y1_raw, m_q, e_prob, e_r_prob, tau_q, h_m, h_tau, h_e

    def query(
        self,
        X_q: torch.Tensor,
        mem: TabularMemory,
        scale: ArmScale,
        *,
        M_q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict arm risk histogram logits ``(N_q, 2, bins)`` and propensity logits ``(N_q,)``."""
        device_type = X_q.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            X_s, X2_s, M_s, X_sat, M_c, y0_raw, y1_raw, m_q, e_prob, e_r_prob, tau_q, h_m, h_tau, h_e = (
                self._query_solver_stats(X_q, M_q, mem)
            )
            n_q = len(X_s)
            log_n = y0_raw.new_full((n_q,), math.log1p(mem.n_ctx))
            arm_feats = torch.stack(
                [y0_raw, y1_raw, m_q, e_r_prob, tau_q, h_m.sqrt(), h_tau.sqrt(), log_n], dim=-1
            )
            cal = self.arm_cal(arm_feats)
            delta_shift = cal[:, :2]
            log_sigma = cal[:, 2:]
            base_delta = scale.logit_offsets(torch.stack([y0_raw, y1_raw], dim=1))
            center_delta = (base_delta + delta_shift).clamp(self.centers[0], self.centers[-1])
            sigma = F.softplus(log_sigma + 0.5) + 0.15
            solver_bin_logits = -0.5 * (
                (self.centers[None, None, :] - center_delta[..., None]) / sigma[..., None]
            ).square()

            e_logit = e_prob.logit()
            prop_feats = torch.stack([e_prob, e_logit, h_e.sqrt(), log_n], dim=-1)
            base_prop_logit = e_logit + self.prop_cal(prop_feats).squeeze(-1)

            q_slots = self._build_patient_slots(X_s, X2_s, M_s, mem, X_sat=X_sat, M_c=M_c)
            arm_q = self._assemble_arm_queries(q_slots, scale)
            arm_pooled = self._attend_queries(arm_q, mem.attn_memories)
            attn_logits = self.head(arm_pooled).reshape(n_q, 2, -1)

            prop_q = self._assemble_propensity_queries(q_slots, scale)
            prop_pooled = self._attend_queries(prop_q, mem.attn_memories)
            attn_prop = self.propensity_head(prop_pooled).squeeze(-1)

            outcome_logits = solver_bin_logits + self.attn_residual_scale.float() * attn_logits
            prop_logits = base_prop_logit + self.prop_residual_scale.float() * attn_prop
            return outcome_logits, prop_logits

    def forward(
        self,
        X_ctx: torch.Tensor,
        t_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
        X_q: torch.Tensor,
        *,
        M_ctx: torch.Tensor | None = None,
        M_q: torch.Tensor | None = None,
    ):
        mem, scale = self.condition(X_ctx, t_ctx, y_ctx, M_ctx=M_ctx)
        outcome_logits, prop_logits = self.query(X_q, mem, scale, M_q=M_q)
        return outcome_logits, prop_logits, scale, mem

    def risk_stats(self, logits: torch.Tensor, scale: ArmScale):
        probs = logits.float().softmax(dim=-1)
        values = scale.bin_risks(self.centers).float()
        mean = (probs * values).sum(dim=-1)
        var = (probs * (values - mean[..., None]).square()).sum(dim=-1).clamp_min(0.0)
        return mean, var.sqrt()

    def risks(self, logits: torch.Tensor, scale: ArmScale):
        mean, _ = self.risk_stats(logits, scale)
        return mean

    def loss_components(
        self,
        outcome_logits: torch.Tensor,
        prop_logits: torch.Tensor,
        scale: ArmScale,
        oracle_risks: torch.Tensor,
        true_propensity: torch.Tensor,
        *,
        role_logits: torch.Tensor | None = None,
        nuis_screen_logits: torch.Tensor | None = None,
        mod_screen_logits: torch.Tensor | None = None,
        true_roles: torch.Tensor | None = None,
    ):
        """Histogram NLL + direct CATE MSE + propensity BCE + balanced column role & screener loss."""
        logits = outcome_logits.float()
        oracle = oracle_risks.float()
        delta = scale.logit_offsets(oracle).clamp(self.centers[0], self.centers[-1])
        nearest = (self.centers - delta[..., None]).abs().argmin(dim=-1)
        sigma = self.bin_widths[nearest].unsqueeze(-1)
        targets = (-0.5 * ((self.centers - delta[..., None]) / sigma).square()).softmax(dim=-1)
        outcome_loss = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

        pred_risks = self.risks(logits, scale)
        pred_ite = pred_risks[:, 1] - pred_risks[:, 0]
        true_ite = oracle[:, 1] - oracle[:, 0]
        cate_loss = (pred_ite - true_ite).square().mean()

        prop_loss = F.binary_cross_entropy_with_logits(
            prop_logits.float(), true_propensity.float()
        )
        total = (
            outcome_loss
            + self.cfg.cate_loss_weight * cate_loss
            + self.cfg.propensity_loss_weight * prop_loss
        )

        def _balanced_bce(sc_logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
            per = F.binary_cross_entropy_with_logits(
                sc_logits.float(), pos_mask.float(), reduction="none"
            )
            has_pos = bool(pos_mask.any().item())
            has_neg = bool((~pos_mask).any().item())
            if has_pos and has_neg:
                return 0.5 * (per[pos_mask].mean() + per[~pos_mask].mean())
            return per.mean()

        role_loss = logits.new_zeros(())
        if role_logits is not None and true_roles is not None and self.cfg.role_loss_weight > 0:
            # Upweight rare non-noise roles cleanly so 99% noise doesn't collapse role logits to noise
            r_logits = role_logits.float()
            r_tgt = true_roles.long()
            is_noise = r_tgt == 0
            per_col = F.cross_entropy(r_logits, r_tgt, reduction="none")
            noise_loss = per_col[is_noise].mean() if bool(is_noise.any().item()) else logits.new_zeros(())
            sig_loss = per_col[~is_noise].mean() if bool((~is_noise).any().item()) else logits.new_zeros(())
            role_loss = 0.5 * (noise_loss + sig_loss)
            # Direct supervision for Round-1 nuisance screener (keeps confounders/prognostics/modifiers,
            # filters noise & pure instruments!) and Round-2 modifier screener (modifiers + dual_role).
            if nuis_screen_logits is not None:
                role_loss = role_loss + 0.35 * _balanced_bce(nuis_screen_logits, r_tgt >= 2)
            if mod_screen_logits is not None:
                is_mod = (r_tgt == 4) | (r_tgt == 5)
                role_loss = role_loss + 0.35 * _balanced_bce(mod_screen_logits, is_mod)
            total = total + self.cfg.role_loss_weight * role_loss

        return total, {
            "outcome_loss": outcome_loss.detach(),
            "cate_loss": cate_loss.detach(),
            "propensity_loss": prop_loss.detach(),
            "role_loss": role_loss.detach(),
        }

    @torch.no_grad()
    def column_role_report(
        self,
        mem: TabularMemory,
        feature_names: list[str] | None = None,
        *,
        calibrated: bool = True,
    ) -> list[dict]:
        """Return interpretable per-column Bayesian role posteriors and ARD gate weights.

        When ``calibrated=True`` (default), applies Normalized Saerens EM
        (:func:`calibrate_role_probabilities`) so probabilities reflect the cohort's
        empirical sparsity instead of the 50/50 balanced training prior while preserving
        exact column ranking.
        """
        p = mem.n_features
        names = feature_names if feature_names is not None else [f"x_{i}" for i in range(p)]
        raw_probs = F.softmax(mem.role_logits.float(), dim=-1).cpu().numpy()
        if calibrated:
            probs, _, _ = calibrate_role_probabilities(mem.role_logits)
        else:
            probs = raw_probs.astype(np.float64)
        ge = torch.zeros(p, device=mem.col_mean.device)
        gm = torch.zeros(p, device=mem.col_mean.device)
        gt = torch.zeros(p, device=mem.col_mean.device)
        ge[mem.nuis_idx] = mem.max_gate("er")
        gm[mem.nuis_idx] = mem.max_gate("m")
        gt[mem.cate_idx] = mem.max_gate("tau")
        if mem.nuis_pair_a is not None and mem.g_per is not None and mem.g_pm is not None:
            ge.scatter_reduce_(0, mem.nuis_pair_a, mem.g_per, reduce="amax", include_self=True)
            ge.scatter_reduce_(0, mem.nuis_pair_b, mem.g_per, reduce="amax", include_self=True)
            gm.scatter_reduce_(0, mem.nuis_pair_a, mem.g_pm, reduce="amax", include_self=True)
            gm.scatter_reduce_(0, mem.nuis_pair_b, mem.g_pm, reduce="amax", include_self=True)
        if mem.cate_pair_a is not None and mem.g_ptau is not None:
            gt.scatter_reduce_(0, mem.cate_pair_a, mem.g_ptau, reduce="amax", include_self=True)
            gt.scatter_reduce_(0, mem.cate_pair_b, mem.g_ptau, reduce="amax", include_self=True)
        ge_np, gm_np, gt_np = ge.cpu().numpy(), gm.cpu().numpy(), gt.cpu().numpy()
        rows = []
        for j in range(p):
            p_sig = float(1.0 - probs[j, 0])
            pred_role = (
                ROLE_NAMES[1 + int(probs[j, 1:].argmax())] if p_sig >= 0.5 else "noise"
            )
            p_conf = float(probs[j, ROLE_NAMES.index("confounder")])
            p_mod = float(probs[j, ROLE_NAMES.index("modifier")])
            p_dual = float(probs[j, ROLE_NAMES.index("dual_role")])
            row = {
                "feature": names[j],
                "predicted_role": pred_role,
                "p_signal": p_sig,
                "p_any_confounder": p_conf + p_dual,
                "p_any_modifier": p_mod + p_dual,
                "gate_propensity": float(ge_np[j]),
                "gate_outcome": float(gm_np[j]),
                "gate_cate": float(gt_np[j]),
                "raw_signal_score": float(1.0 - raw_probs[j, 0]),
            }
            for r_i, r_name in enumerate(ROLE_NAMES):
                row[f"p_{r_name}"] = float(probs[j, r_i])
            rows.append(row)
        rows.sort(
            key=lambda r: (r["p_signal"], max(r["gate_outcome"], r["gate_cate"])),
            reverse=True,
        )
        return rows


def calibrate_role_probabilities(
    role_logits: torch.Tensor | np.ndarray, *, max_iter: int = 50
) -> tuple[np.ndarray, float, float]:
    """Calibrate balanced-loss 6-class ``role_logits`` to empirical cohort sparsity.

    Because training balances noise and non-noise columns 50/50 so 99% noise does not
    collapse logits to all-noise, raw softmax probabilities over-call non-noise at large P.
    This uses Normalized Saerens EM (enforcing the exact likelihood-ratio identity
    ``E[LR | noise] == 1`` with zero Gaussian assumption on the null tail) to recover
    cohort-calibrated posterior probabilities while preserving exact column ranking.

    Returns ``(probs (P, 6), estimated_signal_prior, c_null)``.
    """
    logits_np = (
        role_logits.detach().float().cpu().numpy()
        if isinstance(role_logits, torch.Tensor)
        else np.asarray(role_logits, dtype=np.float64)
    )
    p = len(logits_np)
    if p == 0:
        return np.empty((0, NUM_ROLES), dtype=np.float64), 0.0, 1.0

    l0 = logits_np[:, 0].astype(np.float64)
    ls_max = logits_np[:, 1:].max(axis=1).astype(np.float64)
    l_sig = ls_max + np.log(
        np.exp(logits_np[:, 1:].astype(np.float64) - ls_max[:, None]).sum(axis=1)
    )
    log_lr = np.clip(l_sig - l0, -25.0, 25.0)
    lr_raw = np.exp(log_lr)

    pi = min(0.25, max(10.0 / max(p, 1), 1.0 / max(p, 1)))
    c_null = 1.0
    for _ in range(max_iter):
        lr = lr_raw / max(c_null, 1e-4)
        num = pi * lr
        p_sig = num / (1.0 - pi + num)
        w0 = float((1.0 - p_sig).sum())
        if w0 > 1.0:
            c_new = float(np.clip(((1.0 - p_sig) * lr_raw).sum() / w0, 0.25, 2.0))
            c_null = 0.7 * c_new + 0.3 * c_null
        pi_new = float(np.clip((p_sig.sum() + 0.5) / (p + 1.0), 0.25 / p, 0.75))
        if abs(pi_new - pi) < 1e-6:
            pi = pi_new
            break
        pi = pi_new

    lr = lr_raw / max(c_null, 1e-4)
    p_sig = (pi * lr) / (1.0 - pi + pi * lr)
    cond = np.exp(logits_np[:, 1:].astype(np.float64) - l_sig[:, None])
    probs = np.empty((p, NUM_ROLES), dtype=np.float64)
    probs[:, 0] = 1.0 - p_sig
    probs[:, 1:] = p_sig[:, None] * cond
    return probs, float(pi), float(c_null)


def format_role_summary(report: list[dict], *, top_k: int = 15) -> str:
    """Format a compact stdout summary of top confounders, modifiers, and instruments."""
    p_total = len(report)
    if p_total == 0:
        return "No candidate features in role report."
    est_active = sum(float(r.get("p_signal", 1.0 - r.get("p_noise", 1.0))) for r in report)
    n_called = sum(1 for r in report if r.get("predicted_role") != "noise")
    lines = [
        f"=== SparseTabularCausalPFN Feature Role Summary (P={p_total}) ===",
        f"Estimated active features: ~{est_active:.1f} / {p_total} "
        f"({100.0 * (1.0 - est_active / max(p_total, 1)):.1f}% noise) | "
        f"features with p_signal >= 0.50: {n_called}",
    ]

    def _section(title: str, rows: list[dict], cols: list[tuple[str, str]]):
        lines.append("")
        lines.append(title)
        hdr = f"  {'Feature':<36s}" + "".join(f" {lbl:>9s}" for lbl, _ in cols) + "  Role"
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))
        for r in rows[:top_k]:
            fname = str(r["feature"])
            if len(fname) > 36:
                fname = fname[:33] + "..."
            vals = "".join(f" {float(r.get(key, 0.0)):9.3f}" for _, key in cols)
            lines.append(f"  {fname:<36s}{vals}  {r.get('predicted_role', 'noise')}")

    conf_rows = sorted(
        report,
        key=lambda r: (
            float(r.get("p_any_confounder", r.get("p_confounder", 0.0) + r.get("p_dual_role", 0.0))),
            float(r.get("gate_outcome", 0.0)) + float(r.get("gate_propensity", 0.0)),
        ),
        reverse=True,
    )
    _section(
        f"Top {min(top_k, p_total)} Suspected Confounders (ranked by p_confounder + p_dual_role):",
        conf_rows,
        [
            ("p_any_c", "p_any_confounder"),
            ("p_conf", "p_confounder"),
            ("p_dual", "p_dual_role"),
            ("gate_e", "gate_propensity"),
            ("gate_m", "gate_outcome"),
        ],
    )

    mod_rows = sorted(
        report,
        key=lambda r: (
            float(r.get("p_any_modifier", r.get("p_modifier", 0.0) + r.get("p_dual_role", 0.0))),
            float(r.get("gate_cate", 0.0)),
        ),
        reverse=True,
    )
    _section(
        f"Top {min(top_k, p_total)} Suspected Effect Modifiers (ranked by p_modifier + p_dual_role):",
        mod_rows,
        [
            ("p_any_m", "p_any_modifier"),
            ("p_mod", "p_modifier"),
            ("p_dual", "p_dual_role"),
            ("gate_tau", "gate_cate"),
            ("p_sig", "p_signal"),
        ],
    )

    inst_rows = sorted(
        report,
        key=lambda r: float(r.get("p_instrument", 0.0)),
        reverse=True,
    )
    _section(
        f"Top {min(min(top_k, 10), p_total)} Suspected Pure Instruments (suppressed in nuisance gates):",
        inst_rows[: min(top_k, 10)],
        [
            ("p_inst", "p_instrument"),
            ("gate_e", "gate_propensity"),
            ("gate_m", "gate_outcome"),
            ("p_sig", "p_signal"),
        ],
    )
    return "\n".join(lines)
