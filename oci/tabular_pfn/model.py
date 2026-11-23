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

from oci.longitudinal.model import ArmScale, RidgeFit, _predict_ridge, _sinh_grid

from .prior import ROLE_NAMES

CHECKPOINT_FORMAT_VERSION = 1
COL_STAT_DIM = 24
ROB_STAT_DIM = 6
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
            or self.ridge_grid_size < 2
            or self.bins < 3
            or self.cate_loss_weight < 0
            or self.propensity_loss_weight < 0
            or self.role_loss_weight < 0
        ):
            raise ValueError("Invalid tabular PFN model configuration")
        if self.col_dim % self.heads or self.study_width % self.heads:
            raise ValueError("col_dim and study_width must be divisible by heads")


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
    g_m_lin: torch.Tensor  # (K1,)
    g_m_nl: torch.Tensor  # (K1,)
    g_tau_lin: torch.Tensor  # (K2,)
    g_tau_nl: torch.Tensor  # (K2,)
    attn_gate_feats: torch.Tensor  # (K_attn, 3)
    propensity_fit: RidgeFit
    outcome_fit: RidgeFit
    cate_fit: RidgeFit
    attn_memories: list
    n_ctx: int
    n_features: int


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
):
    """Compute 24-d strictly observational per-column descriptors in O(N * P) time."""
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
    if int(m0.sum().item()) < 3:
        r_y0.zero_(); t_y0.zero_(); r_y0_2.zero_()
    if int(m1.sum().item()) < 3:
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

    stats = torch.stack(
        [
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
        ],
        dim=-1,
    )
    stats = stats * valid_col.float().unsqueeze(-1)
    # Also return raw step-0 priority signals (zero-shot recall guarantee on step 0!)
    raw_nuis_priority = (torch.log1p(y_cond_t) + 0.35 * torch.log1p(abs_t_t) + 0.5 * torch.log1p(abs_z_inter))
    raw_nuis_priority = torch.where(valid_col, raw_nuis_priority, raw_nuis_priority.new_full((), -1e4))
    return stats, raw_nuis_priority


@torch.no_grad()
def _robinson_all_p_stats(
    X_s: torch.Tensor,
    X2_s: torch.Tensor,
    t_res: torch.Tensor,
    y_res: torch.Tensor,
    valid_col: torch.Tensor,
):
    """Scan ALL P columns against adjusted LOO residuals (t - e_loo, y - m_loo) in O(N * P)."""
    n, p = X_s.shape
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
    rob_feats = torch.stack(
        [
            z_lin / 5.0,
            abs_lin / 5.0,
            torch.log1p(abs_lin),
            z_nl / 5.0,
            abs_nl / 5.0,
            torch.log1p(abs_nl),
        ],
        dim=-1,
    )
    raw_mod_priority = torch.log1p(abs_lin) + 0.6 * torch.log1p(abs_nl)
    raw_mod_priority = torch.where(valid_col, raw_mod_priority, raw_mod_priority.new_full((), -1e4))
    return rob_feats, raw_mod_priority


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

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out_proj = nn.Linear(width, width)
        self.corr_bias = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, heads))
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, h: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        # h: (K, D), corr: (K, K)
        k, d = h.shape
        hn = self.norm1(h)
        q, kv_k, v = self.qkv(hn).chunk(3, dim=-1)
        q = q.view(k, self.heads, self.head_dim).transpose(0, 1)  # (H, K, Dh)
        kv_k = kv_k.view(k, self.heads, self.head_dim).transpose(0, 1)
        v = v.view(k, self.heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(q, kv_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        c_feats = torch.stack([corr, corr.abs(), corr * corr], dim=-1)  # (K, K, 3)
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

        # 1. O(P) Round-1 Column Role Encoder & Screener
        self.col_in = nn.Sequential(
            nn.Linear(COL_STAT_DIM, c_dim),
            nn.LayerNorm(c_dim),
            nn.GELU(),
            nn.Linear(c_dim, c_dim),
        )
        self.col_global_blocks = nn.ModuleList([_DeepSetsColumnBlock(c_dim) for _ in range(2)])
        self.nuis_screen_head = nn.Linear(c_dim, 1)

        # 2. Top-K_nuis Correlation-Biased Column Attention -> soft ARD nuisance gates (e, m)
        self.nuis_col_layers = nn.ModuleList(
            [_CorrBiasedColumnLayer(c_dim, cfg.heads) for _ in range(cfg.col_attn_layers)]
        )
        self.nuis_gate_head = nn.Sequential(
            nn.LayerNorm(c_dim), nn.Linear(c_dim, c_dim // 2), nn.GELU(), nn.Linear(c_dim // 2, 4)
        )
        # Initialize linear gates moderately open, nonlinear gates slightly shrunk
        nn.init.constant_(self.nuis_gate_head[-1].bias, 0.0)

        # 3. Round-2 ALL-P Adjusted Robinson Updater -> Final 6-class Role Head + CATE Screener
        self.rob_update = nn.Sequential(
            nn.LayerNorm(c_dim + ROB_STAT_DIM),
            nn.Linear(c_dim + ROB_STAT_DIM, c_dim),
            nn.GELU(),
            nn.Linear(c_dim, c_dim),
        )
        self.rob_global_block = _DeepSetsColumnBlock(c_dim)
        self.role_head = nn.Sequential(nn.LayerNorm(c_dim), nn.Linear(c_dim, NUM_ROLES))
        self.mod_screen_head = nn.Linear(c_dim, 1)

        # 4. Top-K_cate Correlation-Biased Column Attention -> soft ARD CATE gates (tau)
        self.cate_col_layers = nn.ModuleList(
            [_CorrBiasedColumnLayer(c_dim, cfg.heads) for _ in range(cfg.col_attn_layers)]
        )
        self.cate_gate_head = nn.Sequential(
            nn.LayerNorm(c_dim), nn.Linear(c_dim, c_dim // 2), nn.GELU(), nn.Linear(c_dim // 2, 2)
        )
        nn.init.constant_(self.cate_gate_head[-1].bias, -0.5)

        # 5. Learnable LOO-CV Ridge penalty grids for e, m, tau
        k_grid = cfg.ridge_grid_size
        base_grid = torch.linspace(0.2, 0.2 + 1.5 * (k_grid - 1), k_grid)
        tau_grid = torch.linspace(0.8, 0.8 + 1.5 * (k_grid - 1), k_grid)
        self.log_lam_e = nn.Parameter(base_grid.clone())
        self.log_lam_m = nn.Parameter(base_grid.clone())
        self.log_lam_tau = nn.Parameter(tau_grid.clone())
        self.log_cv_temp = nn.Parameter(torch.tensor(1.5))

        # 6. Calibration & Epistemic-Uncertainty Heads on ARD R-learner outputs + leverages
        self.arm_cal = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 4))
        self.prop_cal = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.normal_(self.arm_cal[-1].weight, std=0.01)
        nn.init.zeros_(self.arm_cal[-1].bias)
        nn.init.normal_(self.prop_cal[-1].weight, std=0.01)
        nn.init.zeros_(self.prop_cal[-1].bias)

        # 7. Compact Residual Two-Way TabPFN Study Layers over top-attn_slots gated features
        self.cell_proj = nn.Sequential(
            nn.Linear(6, s_dim), nn.GELU(), nn.Linear(s_dim, s_dim)
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

    def _build_patient_slots(
        self,
        X_s: torch.Tensor,
        X2_s: torch.Tensor,
        M_s: torch.Tensor,
        mem: TabularMemory,
    ) -> torch.Tensor:
        """Build (N, attn_slots, study_width) cell embeddings on top gated columns."""
        n = len(X_s)
        idx = mem.attn_idx
        k_act = len(idx)
        x_sel = X_s[:, idx]  # (N, k_act)
        x2_sel = X2_s[:, idx]
        m_sel = M_s[:, idx]
        g_feats = mem.attn_gate_feats.unsqueeze(0).expand(n, -1, -1)  # (N, k_act, 3)
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
        for layer, memory in zip(self.study_layers, attn_memories, strict=True):
            queries = layer.slot_mix(queries)
            queries = layer.sample_mix(queries, memory)
        return queries[:, -1, :]

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
            X_s, X2_s, M_s, col_mean, col_sd, x2_mean, x2_sd, valid_col = _standardize_context(
                X_ctx, M_ctx
            )
            col_stats, raw_nuis_prio = _column_sufficient_stats(
                X_s, X2_s, M_s, t_f, y_f, valid_col
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
            corr_nuis = self._empirical_corr(X_nuis)
            for layer in self.nuis_col_layers:
                h_nuis = layer(h_nuis, corr_nuis)
            nuis_gates = torch.sigmoid(self.nuis_gate_head(h_nuis)).clamp(1e-3, 1.0)
            g_e_lin, g_e_nl, g_m_lin, g_m_nl = nuis_gates.unbind(dim=-1)
            v_nuis = valid_col[nuis_idx].float()
            g_e_lin = g_e_lin * v_nuis
            g_e_nl = g_e_nl * v_nuis
            g_m_lin = g_m_lin * v_nuis
            g_m_nl = g_m_nl * v_nuis

            # 4. Differentiable ARD Closed-Form LOO-CV Nuisances (e_loo, m_loo)
            ones = torch.ones((n_ctx, 1), device=X_s.device, dtype=torch.float32)
            A_e = torch.cat([ones, X_nuis * g_e_lin[None, :], X2_nuis * g_e_nl[None, :]], dim=1)
            A_m = torch.cat([ones, X_nuis * g_m_lin[None, :], X2_nuis * g_m_nl[None, :]], dim=1)
            cv_temp = F.softplus(self.log_cv_temp.float()) + 0.1

            lam_e = self.log_lam_e.float().exp().clamp(0.05, 1e5)
            prop_fit, e_loo = _fit_ard_loo_cv_ridge(A_e, t_f, lam_e, cv_temp)
            e_loo = e_loo.clamp(0.02, 0.98)

            lam_m = self.log_lam_m.float().exp().clamp(0.05, 1e5)
            out_fit, m_loo = _fit_ard_loo_cv_ridge(A_m, y_f, lam_m, cv_temp)

            # 5. Round-2 ALL-P Adjusted Robinson Interaction Scan -> Role Posteriors & CATE Set
            t_res = t_f - e_loo
            y_res = y_f - m_loo
            rob_feats, raw_mod_prio = _robinson_all_p_stats(X_s, X2_s, t_res, y_res, valid_col)
            h_col2 = h_col + self.rob_update(torch.cat([h_col, rob_feats], dim=-1))
            h_col2 = self.rob_global_block(h_col2)
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
            corr_cate = self._empirical_corr(X_cate)
            for layer in self.cate_col_layers:
                h_cate = layer(h_cate, corr_cate)
            cate_gates = torch.sigmoid(self.cate_gate_head(h_cate)).clamp(1e-3, 1.0)
            g_tau_lin, g_tau_nl = cate_gates.unbind(dim=-1)
            v_cate = valid_col[cate_idx].float()
            g_tau_lin = g_tau_lin * v_cate
            g_tau_nl = g_tau_nl * v_cate

            A_tau = torch.cat(
                [ones, X_cate * g_tau_lin[None, :], X2_cate * g_tau_nl[None, :]], dim=1
            )
            B_ctx = t_res.unsqueeze(-1) * A_tau
            lam_tau = self.log_lam_tau.float().exp().clamp(0.05, 1e5)
            cate_fit, _ = _fit_ard_loo_cv_ridge(B_ctx, y_res, lam_tau, cv_temp, intercept_reg=0.05)

            # 7. Select top attn_slots columns across (e, m, tau) gates for residual Two-Way TabPFN
            full_ge = torch.zeros(p, device=X_s.device, dtype=torch.float32)
            full_gm = torch.zeros(p, device=X_s.device, dtype=torch.float32)
            full_gt = torch.zeros(p, device=X_s.device, dtype=torch.float32)
            full_ge[nuis_idx] = torch.maximum(g_e_lin, g_e_nl)
            full_gm[nuis_idx] = torch.maximum(g_m_lin, g_m_nl)
            full_gt[cate_idx] = torch.maximum(g_tau_lin, g_tau_nl)
            attn_prio = (
                full_gm + 1.5 * full_gt + 0.5 * full_ge + 1e-3 * (raw_nuis_prio.clamp(-10, 10) + raw_mod_prio.clamp(-10, 10))
            )
            attn_prio = torch.where(valid_col, attn_prio, attn_prio.new_full((), -1e5))
            k_attn = min(p, self.cfg.attn_slots)
            attn_idx = torch.topk(attn_prio, k_attn, largest=True).indices
            attn_gate_feats = torch.stack(
                [full_ge[attn_idx], full_gm[attn_idx], full_gt[attn_idx]], dim=-1
            )

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
                g_m_lin=g_m_lin,
                g_m_nl=g_m_nl,
                g_tau_lin=g_tau_lin,
                g_tau_nl=g_tau_nl,
                attn_gate_feats=attn_gate_feats,
                propensity_fit=prop_fit,
                outcome_fit=out_fit,
                cate_fit=cate_fit,
                attn_memories=[],
                n_ctx=n_ctx,
                n_features=p,
            )

            ctx_slots = self._build_patient_slots(X_s, X2_s, M_s, tmp_mem)
            context = self._assemble_context(ctx_slots, t_f, y_f, scale)
            attn_memories = []
            for i, layer in enumerate(self.study_layers):
                context = layer.slot_mix(context)
                memory = layer.sample_norm(context).transpose(0, 1)  # (S, N_ctx, D)
                attn_memories.append(memory)
                if i + 1 < len(self.study_layers):
                    context = layer.sample_mix(context, memory)
            tmp_mem.attn_memories = attn_memories
            return tmp_mem, scale

    def _query_solver_stats(
        self, X_q: torch.Tensor, M_q: torch.Tensor | None, mem: TabularMemory
    ):
        X_s, X2_s, M_s = _apply_standardize(
            X_q, M_q, mem.col_mean, mem.col_sd, mem.x2_mean, mem.x2_sd, mem.valid_col
        )
        n_q = len(X_s)
        ones_q = torch.ones((n_q, 1), device=X_s.device, dtype=torch.float32)

        X_n = X_s[:, mem.nuis_idx]
        X2_n = X2_s[:, mem.nuis_idx]
        A_e_q = torch.cat([ones_q, X_n * mem.g_e_lin[None, :], X2_n * mem.g_e_nl[None, :]], dim=1)
        A_m_q = torch.cat([ones_q, X_n * mem.g_m_lin[None, :], X2_n * mem.g_m_nl[None, :]], dim=1)

        X_c = X_s[:, mem.cate_idx]
        X2_c = X2_s[:, mem.cate_idx]
        A_t_q = torch.cat(
            [ones_q, X_c * mem.g_tau_lin[None, :], X2_c * mem.g_tau_nl[None, :]], dim=1
        )

        e_q, h_e = _predict_ridge(mem.propensity_fit, A_e_q)
        m_q, h_m = _predict_ridge(mem.outcome_fit, A_m_q)
        tau_q, h_tau = _predict_ridge(mem.cate_fit, A_t_q)
        e_prob = e_q.clamp(0.01, 0.99)
        y0_raw = (m_q - e_prob * tau_q).clamp(0.01, 0.99)
        y1_raw = (m_q + (1.0 - e_prob) * tau_q).clamp(0.01, 0.99)
        return X_s, X2_s, M_s, y0_raw, y1_raw, m_q, e_prob, tau_q, h_m, h_tau, h_e

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
            X_s, X2_s, M_s, y0_raw, y1_raw, m_q, e_prob, tau_q, h_m, h_tau, h_e = (
                self._query_solver_stats(X_q, M_q, mem)
            )
            n_q = len(X_s)
            log_n = y0_raw.new_full((n_q,), math.log1p(mem.n_ctx))
            arm_feats = torch.stack(
                [y0_raw, y1_raw, m_q, e_prob, tau_q, h_m.sqrt(), h_tau.sqrt(), log_n], dim=-1
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

            q_slots = self._build_patient_slots(X_s, X2_s, M_s, mem)
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
        ge[mem.nuis_idx] = torch.maximum(mem.g_e_lin, mem.g_e_nl)
        gm[mem.nuis_idx] = torch.maximum(mem.g_m_lin, mem.g_m_nl)
        gt[mem.cate_idx] = torch.maximum(mem.g_tau_lin, mem.g_tau_nl)
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
