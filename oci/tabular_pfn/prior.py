"""Vectorized OCI-aligned multi-DGP high-dimensional sparse tabular causal prior.

Generates synthetic observational studies with N up to 10,000 and P up to 2,000+
candidate features by mixing three fast (<25 ms/study) OCI-aligned DGP families:

  1. ``sparse_oci`` (default 50%):
     High-dimensional all-evidence clinical tables with 50–99.5% noise, explicit
     6-class causal roles (``noise``, ``instrument``, ``prognostic``, ``confounder``,
     ``modifier``, ``dual_role``), multi-level mutually exclusive Dirichlet one-hot
     categorical blocks, collinear noisy proxies, block-correlated factor noise,
     linear/threshold/quadratic/hinge/saturating + pairwise interactions, optional
     unmeasured confounding (Z-bias!), Stage-1 LLM extraction noise & proxy-only
     confounders, MCAR/MAR/MNAR + 3-state EHR documentation, occasional RCTs,
     strictly null-HTE studies, and both ``logOR`` and additive ``RD`` scales.
  2. ``reequation_calc`` (default 25%):
     Exact vectorized implementation of OCI's ``sample_size_calculator/engine.R``
     and ``synthetic_data/reequation.py`` tabular DGP family (exchangeable Gaussian
     copula, uniform-CDF / log-normal continuous, Beta-prevalence binary, Dirichlet
     multi-level categorical contrasts, ``none``/``threshold``/``quadratic``
     nonlinearities, optional unmeasured confounders, ``logOR`` and ``RD`` scales)
     embedded inside ``P = 20..2,000`` realistic Stage-2 noise/proxy/instrument cols.
  3. ``nemotron_design`` (default 25% when bundled ``nemotron_5000.json.gz`` exists):
     Direct tabular episodes drawn from the 4,000 ``split == "train"`` oncology
     study designs in ``synthetic_data/designs/nemotron_5000.json.gz`` (real clinical
     lab/vital/PRO measurement moments, real cancer-specific note-finding prevalences
     and plausible causal role menus, pairwise outcome interactions, Stage-1
     extraction error, and MNAR/MAR missingness) embedded inside ``P = 20..2,000``
     candidate columns. The 1,000 ``split == "held_out"`` designs (including 3 unseen
     cancers) are strictly excluded from training and reserved for zero-shot eval.
"""

from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri

ROLE_NAMES = (
    "noise",
    "instrument",
    "prognostic",
    "confounder",
    "modifier",
    "dual_role",
)
ROLE_TO_ID = {name: idx for idx, name in enumerate(ROLE_NAMES)}
DGP_FAMILIES = ("sparse_oci", "reequation_calc", "nemotron_design")

_DEFAULT_DESIGN_BANK = (
    Path(__file__).resolve().parents[2] / "synthetic_data" / "designs" / "nemotron_5000.json.gz"
)
_DESIGN_BANK_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}

# Compact copy of clinical measurement reference moments (aligned with synthetic_data/study_designs.py)
_LAB_REF = {
    "platelets": (250.0, 35.0),
    "hemoglobin": (13.0, 1.0),
    "wbc": (7.2, 1.5),
    "neutrophils_abs": (4.2, 1.5),
    "lymphocytes_abs": (2.0, 0.6),
    "creatinine": (1.0, 0.28),
    "sodium": (139.0, 2.0),
    "potassium": (4.2, 0.3),
    "bun": (17.0, 3.0),
    "glucose": (99.0, 13.0),
}
_LAB_CHG = {
    "platelets": 42.0,
    "hemoglobin": 0.85,
    "wbc": 1.4,
    "neutrophils_abs": 1.1,
    "lymphocytes_abs": 0.45,
    "creatinine": 0.14,
    "sodium": 2.5,
    "potassium": 0.37,
    "bun": 3.0,
    "glucose": 18.0,
}
_VITAL_REF = {
    "weight_kg": (80.0, 18.0),
    "systolic_mmhg": (124.0, 10.0),
    "pulse_bpm": (78.0, 8.0),
    "spo2_pct": (97.0, 1.8),
    "respirations_per_min": (16.0, 2.0),
}
_VITAL_CHG = {
    "weight_kg": 3.5,
    "systolic_mmhg": 8.2,
    "pulse_bpm": 7.2,
    "spo2_pct": 1.5,
    "respirations_per_min": 1.6,
}
_MEASUREMENT_MOMENTS: dict[str, tuple[float, float]] = {
    "latest_ecog": (1.1, 0.85),
    "recent_ed_visits": (0.7, 0.85),
    "log_nlr": (0.75, 0.55),
    "creatinine_slope": (0.03, 0.19),
    "weight_loss_per_year": (3.2, 5.5),
}
for _src, _ref, _chg in (("lab", _LAB_REF, _LAB_CHG), ("vital", _VITAL_REF, _VITAL_CHG)):
    for _fld, (_mu, _sd) in _ref.items():
        for _sum in ("last", "mean", "change"):
            _MEASUREMENT_MOMENTS[f"{_src}.{_fld}.{_sum}"] = (
                0.0 if _sum == "change" else _mu,
                _chg[_fld] if _sum == "change" else _sd,
            )


@dataclass(frozen=True)
class PriorConfig:
    min_features: int = 20
    max_features: int = 2000
    min_active: int = 2
    max_active: int = 18
    null_hte_prob: float = 0.25
    instrument_prob: float = 0.70
    max_instruments: int = 8
    max_proxies: int = 24
    missing_study_prob: float = 0.75
    max_missing_rate: float = 0.45
    factor_noise_rank: int = 12
    # OCI-aligned multi-DGP mixture settings
    family_weights: tuple[float, float, float] = (0.50, 0.25, 0.25)
    design_bank_path: str | None = None
    design_split: str = "train"
    unmeasured_conf_prob: float = 0.25
    extraction_noise_prob: float = 0.35
    mnar_prob: float = 0.50
    rd_scale_prob: float = 0.25
    rct_prob: float = 0.08
    max_treatment_logit_std: float = 2.8
    categorical_block_prob: float = 0.70

    def __post_init__(self):
        if isinstance(self.family_weights, list):
            object.__setattr__(self, "family_weights", tuple(float(w) for w in self.family_weights))
        if len(self.family_weights) != len(DGP_FAMILIES) or any(w < 0 for w in self.family_weights):
            raise ValueError("family_weights must be 3 non-negative floats")
        if sum(self.family_weights) <= 0:
            raise ValueError("At least one family weight must be positive")
        if self.design_split not in ("train", "held_out"):
            raise ValueError("design_split must be 'train' or 'held_out'")


def _resolve_design_bank_path(path: str | None) -> Path | None:
    if path is not None:
        p = Path(path)
        return p if p.is_file() else None
    return _DEFAULT_DESIGN_BANK if _DEFAULT_DESIGN_BANK.is_file() else None


def _load_design_split(path: str | None, split: str) -> list[dict[str, Any]]:
    """Load and cache compact per-design specs strictly for ``split`` ('train' or 'held_out')."""
    resolved = _resolve_design_bank_path(path)
    if resolved is None:
        return []
    key = (str(resolved), split)
    cached = _DESIGN_BANK_CACHE.get(key)
    if cached is not None:
        return cached
    with gzip.open(resolved, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    compact: list[dict[str, Any]] = []
    for d in raw.get("designs", ()):
        q = d.get("question", {})
        if q.get("split") != split:
            continue
        cands = d.get("candidates", {})
        meas_list = []
        for m in cands.get("measurements", ()):
            name = str(m.get("name", ""))
            mu, sd = _MEASUREMENT_MOMENTS.get(name, (1.2, 0.95) if name.startswith("pro.") else (0.0, 1.0))
            meas_list.append((name, float(mu), float(sd), tuple(m.get("roles", ("unused",)))))
        find_list = []
        for f in cands.get("note_findings", ()):
            prev = float(np.clip(float(f.get("prevalence", 0.2)), 0.02, 0.85))
            find_list.append((prev, tuple(f.get("roles", ("unused",)))))
        if meas_list or find_list:
            compact.append(
                {
                    "id": str(d.get("id", "")),
                    "cancer": str(q.get("cancer", "")),
                    "measurements": meas_list,
                    "findings": find_list,
                }
            )
    _DESIGN_BANK_CACHE[key] = compact
    return compact


def _expit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(linpred: np.ndarray, target_rate: float) -> float:
    """Solve mean(expit(linpred + b)) == target_rate via safeguarded bisection."""
    lo, hi = -14.0, 14.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if float(_expit(linpred + mid).mean()) < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _standardize_vec(v: np.ndarray) -> np.ndarray:
    sd = float(v.std())
    if sd > 1e-6:
        return ((v - float(v.mean())) / sd).astype(np.float32)
    return np.zeros_like(v, dtype=np.float32)


def _apply_nonlinear(z: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Apply a realistic clinical transform and re-standardize to unit scale."""
    if mode == "linear" or mode == "none":
        out = z
    elif mode == "threshold":
        thresh = float(rng.uniform(-0.8, 1.0))
        out = (z > thresh).astype(np.float32)
    elif mode == "quadratic":
        out = z * z - 1.0
    elif mode == "hinge":
        thresh = float(rng.uniform(-0.5, 0.8))
        out = np.maximum(z - thresh, 0.0)
    elif mode == "saturating":
        out = np.tanh(float(rng.uniform(0.6, 1.8)) * z)
    else:
        out = z
    return _standardize_vec(out)


def _generate_factor_latents(
    n_total: int, p: int, rank: int, rng: np.random.Generator
) -> np.ndarray:
    """Fast O(N * P * rank) block-correlated Gaussian latent features."""
    if p <= 0:
        return np.empty((n_total, 0), dtype=np.float32)
    r = min(rank, max(2, p // 4))
    factors = rng.standard_normal((n_total, r)).astype(np.float32)
    loadings = rng.standard_normal((r, p)).astype(np.float32) / math.sqrt(r)
    strength = float(rng.uniform(0.15, 0.65))
    loadings *= (rng.random((r, p)) < float(rng.uniform(0.25, 0.70))).astype(np.float32)
    Z = strength * (factors @ loadings) + math.sqrt(
        max(1.0 - strength * strength, 0.1)
    ) * rng.standard_normal((n_total, p)).astype(np.float32)
    Z -= Z.mean(axis=0, keepdims=True)
    Z /= np.maximum(Z.std(axis=0, keepdims=True), 1e-5)
    return Z


def _realize_column_types(
    Z: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized conversion of latent N(0, 1) columns to mixed clinical types."""
    n, p = Z.shape
    if p == 0:
        return np.empty((n, 0), dtype=np.float32), np.zeros(0, dtype=bool)
    X = Z.astype(np.float32, copy=True)
    is_binary = np.zeros(p, dtype=bool)

    # 45% binary, 15% ordinal (0..4), 25% Gaussian, 15% skewed log-normal
    u = rng.random(p)
    bin_cols = np.where(u < 0.45)[0]
    ord_cols = np.where((u >= 0.45) & (u < 0.60))[0]
    log_cols = np.where(u >= 0.85)[0]

    if len(bin_cols):
        prev = np.exp(rng.uniform(math.log(0.03), math.log(0.55), size=len(bin_cols)))
        thresh = ndtri(1.0 - prev).astype(np.float32)
        X[:, bin_cols] = (Z[:, bin_cols] >= thresh[None, :]).astype(np.float32)
        is_binary[bin_cols] = True

    if len(ord_cols):
        # Vectorized 4-level clinical scores (ECOG / stage / grade-like)
        c1 = rng.uniform(-1.3, -0.3, size=len(ord_cols)).astype(np.float32)
        c2 = c1 + rng.uniform(0.4, 1.0, size=len(ord_cols)).astype(np.float32)
        c3 = c2 + rng.uniform(0.4, 1.1, size=len(ord_cols)).astype(np.float32)
        zo = Z[:, ord_cols]
        X[:, ord_cols] = (
            (zo > c1[None, :]).astype(np.float32)
            + (zo > c2[None, :]).astype(np.float32)
            + (zo > c3[None, :]).astype(np.float32)
        )

    if len(log_cols):
        scales = rng.uniform(0.3, 0.8, size=len(log_cols)).astype(np.float32)
        X[:, log_cols] = np.exp(np.clip(Z[:, log_cols] * scales[None, :], -4.0, 4.0))

    return X, is_binary


def _make_categorical_dummy_block(
    z_col: np.ndarray,
    n_levels: int,
    rng: np.random.Generator,
    *,
    concentration: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one latent N(0, 1) column into (K-1) mutually exclusive Dirichlet one-hot dummies.

    Returns ``(dummies (N, K-1), level_idx (N,))``.
    """
    probs = rng.dirichlet(np.full(n_levels, concentration, dtype=np.float64))
    probs = np.clip(probs, 0.04, 0.90)
    probs /= probs.sum()
    cuts = ndtri(np.cumsum(probs[:-1])).astype(np.float32)
    level = np.searchsorted(cuts, z_col).astype(np.int64)  # 0 .. K-1
    dummies = np.stack([(level == k).astype(np.float32) for k in range(1, n_levels)], axis=1)
    return dummies, level


def _inject_extraction_noise(
    X_full: np.ndarray,
    is_binary: np.ndarray,
    active_cols: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate realistic Stage-1 LLM extraction noise on active/observed columns."""
    X_noisy = X_full.copy()
    if len(active_cols) == 0:
        return X_noisy
    n = len(X_full)
    for col in active_cols:
        if rng.random() > 0.65:
            continue
        if is_binary[col]:
            # Asymmetric false-negative (higher) & false-positive (lower) clinical extraction error
            fn_rate = float(rng.uniform(0.03, 0.18))
            fp_rate = float(rng.uniform(0.01, 0.07))
            pos = X_noisy[:, col] > 0.5
            flip_fn = pos & (rng.random(n) < fn_rate)
            flip_fp = (~pos) & (rng.random(n) < fp_rate)
            X_noisy[flip_fn, col] = 0.0
            X_noisy[flip_fp, col] = 1.0
        else:
            sd = float(X_noisy[:, col].std())
            if sd > 1e-5:
                sigma = float(rng.uniform(0.12, 0.38)) * sd
                X_noisy[:, col] += rng.normal(0.0, sigma, size=n).astype(np.float32)
    return X_noisy


def _inject_missingness_and_3state(
    X_in: np.ndarray,
    y_base: np.ndarray,
    roles: np.ndarray,
    prior: PriorConfig,
    rng: np.random.Generator,
    *,
    allow_3state_cols: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Inject realistic MCAR, MAR, or MNAR missingness (and optional 3-state EHR unknown flags).

    Returns ``(X_obs, M, roles_out, miss_mode)`` keeping total feature count exact.
    """
    n_total, p = X_in.shape
    X_obs = X_in.copy()
    M = np.zeros((n_total, p), dtype=np.float32)
    roles_out = roles.copy()
    if rng.random() >= prior.missing_study_prob:
        return X_obs, M, roles_out, "none"

    is_mnar = rng.random() < prior.mnar_prob
    miss_mode = "mnar" if is_mnar else "mar"
    miss_rates = rng.uniform(0.0, prior.max_missing_rate, size=p).astype(np.float32)
    miss_rates *= (rng.random(p) < 0.65).astype(np.float32)
    active_miss = np.where(miss_rates > 0.015)[0]
    if len(active_miss) == 0:
        return X_obs, M, roles_out, "none"

    b_m = np.log(miss_rates[active_miss] / np.maximum(1.0 - miss_rates[active_miss], 1e-4))
    if is_mnar:
        # True MNAR (matches synthetic_data/config.py documentation_mechanism="mnar"):
        # documentation probability depends on the column's OWN true standardized value
        X_sub = X_in[:, active_miss]
        mu = X_sub.mean(axis=0, keepdims=True)
        sd = np.maximum(X_sub.std(axis=0, keepdims=True), 1e-4)
        Z_sub = np.clip((X_sub - mu) / sd, -4.0, 4.0)
        slopes = rng.uniform(0.3, 1.1, size=len(active_miss)).astype(np.float32) * rng.choice(
            [-1.0, 1.0], size=len(active_miss)
        ).astype(np.float32)
        prob_m = _expit(b_m[None, :] + Z_sub * slopes[None, :])
    else:
        mar_driver = _standardize_vec(y_base)
        mar_strength = float(rng.uniform(0.0, 0.7))
        signs = rng.choice([-1.0, 1.0], size=len(active_miss)).astype(np.float32)
        prob_m = _expit(b_m[None, :] + (mar_strength * mar_driver[:, None]) * signs[None, :])

    mask = rng.random((n_total, len(active_miss))) < prob_m
    valid_mask_cols = mask.sum(axis=0) < (n_total - 2)
    mask[:, ~valid_mask_cols] = False
    M[:, active_miss] = mask.astype(np.float32)
    X_obs[:, active_miss] = np.where(mask, np.nan, X_obs[:, active_miss])

    # Optional 3-state EHR unknown/missingness indicator columns replacing spare noise slots
    if allow_3state_cols and rng.random() < 0.45:
        noise_slots = np.where(roles_out == ROLE_TO_ID["noise"])[0]
        high_miss_sig = [c for c in active_miss if roles_out[c] != 0 and M[:, c].mean() > 0.08]
        n_add = min(len(high_miss_sig), len(noise_slots), 4)
        if n_add > 0:
            for k in range(n_add):
                src = high_miss_sig[k]
                dst = noise_slots[k]
                # 3-state "documented_unknown" / missingness indicator (always observed)
                X_obs[:, dst] = M[:, src]
                M[:, dst] = 0.0
                # Under MNAR, missingness itself carries confounding/prognostic info from src
                roles_out[dst] = roles_out[src] if is_mnar else ROLE_TO_ID["noise"]

    return X_obs, M, roles_out, miss_mode


def _finalize_outcomes(
    t_signal: np.ndarray,
    y_base: np.ndarray,
    ht_part: np.ndarray,
    has_hte: bool,
    prior: PriorConfig,
    rng: np.random.Generator,
    *,
    force_scale: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Calibrate propensity, control outcome, and treatment effect (logOR or bounded RD)."""
    n_total = len(t_signal)
    # Occasional unconfounded / RCT-like studies (matching synthetic_data/longitudinal.py)
    is_rct = rng.random() < prior.rct_prob
    t_sd = float(t_signal.std())
    if is_rct or t_sd < 1e-6:
        t_scaled = np.zeros(n_total, dtype=np.float32)
    else:
        target_t_sd = float(rng.uniform(0.5, prior.max_treatment_logit_std))
        t_scaled = (t_signal * (target_t_sd / t_sd)).astype(np.float32)

    target_t_rate = float(rng.uniform(0.12, 0.88))
    b_t = _calibrate_intercept(t_scaled, target_t_rate)
    true_prop = _expit(t_scaled + b_t).astype(np.float32)

    y_sd = float(y_base.std())
    if y_sd > 1e-6:
        target_y_sd = float(rng.uniform(0.6, 2.3))
        y_scaled = (y_base * (target_y_sd / y_sd)).astype(np.float32)
    else:
        y_scaled = np.zeros(n_total, dtype=np.float32)

    scale_mode = force_scale if force_scale is not None else (
        "RD" if rng.random() < prior.rd_scale_prob else "logOR"
    )
    target_y_rate = float(rng.uniform(0.08, 0.88))

    if scale_mode == "RD":
        # Bounded additive Risk-Difference scale (matches sample_size_calculator/engine.R)
        target_ctrl = float(np.clip(target_y_rate, 0.15, 0.85))
        b_y = _calibrate_intercept(y_scaled, target_ctrl)
        true_y0 = _expit(y_scaled + b_y).astype(np.float32)
        ate_rd = float(rng.uniform(-0.22, 0.22))
        if has_hte and float(ht_part.std()) > 1e-6:
            rd_mod = _standardize_vec(ht_part) * float(rng.uniform(0.06, 0.18))
            rd = ate_rd + rd_mod
        else:
            rd = np.full(n_total, ate_rd, dtype=np.float32)
        true_y1 = np.clip(true_y0 + rd, 0.015, 0.985).astype(np.float32)
    else:
        ate_logit = float(rng.uniform(-1.3, 1.3))
        tau_logit = np.full(n_total, ate_logit, dtype=np.float32)
        if has_hte and float(ht_part.std()) > 1e-6:
            target_ht_sd = float(rng.uniform(0.35, 1.7))
            tau_logit = tau_logit + _standardize_vec(ht_part) * target_ht_sd
        b_y = _calibrate_intercept(y_scaled + true_prop * tau_logit, target_y_rate)
        true_y0 = _expit(y_scaled + b_y).astype(np.float32)
        true_y1 = _expit(y_scaled + b_y + tau_logit).astype(np.float32)

    true_ite = (true_y1 - true_y0).astype(np.float32)
    t_obs = (rng.random(n_total) < true_prop).astype(np.float32)
    p_obs = np.where(t_obs > 0.5, true_y1, true_y0)
    y_obs = (rng.random(n_total) < p_obs).astype(np.float32)
    return t_obs, y_obs, true_prop, true_y0, true_y1, true_ite, scale_mode


def _pad_with_noise_proxies_and_instruments(
    X_base: np.ndarray,
    is_bin_base: np.ndarray,
    roles_base: np.ndarray,
    t_signal: np.ndarray,
    p_target: int,
    prior: PriorConfig,
    rng: np.random.Generator,
    *,
    allow_extra_instruments: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad an OCI core feature table up to ``p_target`` with noise, collinear proxies, & instruments."""
    n_total, p_base = X_base.shape
    p = max(p_target, p_base)
    rem = p - p_base
    if rem <= 0:
        return X_base, is_bin_base, roles_base, t_signal

    # Optional extra instruments if none exist yet or budget allows
    n_inst = 0
    if allow_extra_instruments and rem >= 2 and rng.random() < prior.instrument_prob:
        n_inst = int(rng.integers(1, min(prior.max_instruments, max(2, rem // 2)) + 1))
    rem -= n_inst

    active_idx = np.where(roles_base != ROLE_TO_ID["noise"])[0]
    n_proxies = 0
    if rem >= 2 and len(active_idx) > 0:
        max_px = min(prior.max_proxies, rem // 2)
        if max_px >= 1:
            n_proxies = int(rng.integers(0, max_px + 1))
    n_noise = rem - n_proxies

    Z_pad = _generate_factor_latents(n_total, n_inst + n_proxies + n_noise, prior.factor_noise_rank, rng)
    # Build collinear proxies of active columns
    if n_proxies > 0:
        parents = rng.choice(active_idx, size=n_proxies, replace=True)
        X_par = X_base[:, parents]
        X_par_std = (X_par - X_par.mean(axis=0, keepdims=True)) / np.maximum(
            X_par.std(axis=0, keepdims=True), 1e-5
        )
        rhos = rng.uniform(0.55, 0.94, size=n_proxies).astype(np.float32)
        eps = rng.standard_normal((n_total, n_proxies)).astype(np.float32)
        Z_pad[:, n_inst : n_inst + n_proxies] = (
            rhos[None, :] * X_par_std + np.sqrt(1.0 - rhos * rhos)[None, :] * eps
        )

    X_pad, bin_pad = _realize_column_types(Z_pad, rng)
    roles_pad = np.zeros(n_inst + n_proxies + n_noise, dtype=np.int64)
    if n_inst > 0:
        roles_pad[:n_inst] = ROLE_TO_ID["instrument"]
        X_inst = X_pad[:, :n_inst]
        X_inst_std = (X_inst - X_inst.mean(axis=0, keepdims=True)) / np.maximum(
            X_inst.std(axis=0, keepdims=True), 1e-5
        )
        w_inst = rng.normal(0.0, float(rng.uniform(0.6, 1.8)), size=n_inst).astype(np.float32)
        t_signal = t_signal + X_inst_std @ w_inst

    X_all = np.concatenate([X_base, X_pad], axis=1)
    bin_all = np.concatenate([is_bin_base, bin_pad])
    roles_all = np.concatenate([roles_base, roles_pad])
    return X_all, bin_all, roles_all, t_signal


# ==============================================================================
# FAMILY 1: UPGRADED HIGH-P SPARSE ALL-EVIDENCE OCI PRIOR
# ==============================================================================
def _sample_sparse_oci(
    n_total: int, p: int, prior: PriorConfig, rng: np.random.Generator
) -> dict:
    is_null_hte = bool(rng.random() < prior.null_hte_prob)
    n_conf = int(rng.integers(1, min(8, max(2, p // 4)) + 1))
    n_prog = int(rng.integers(0, min(7, max(1, p // 5)) + 1))
    if is_null_hte:
        n_mod, n_dual = 0, 0
    else:
        n_mod = int(rng.integers(0, min(5, max(1, p // 6)) + 1))
        n_dual = int(rng.integers(0, min(3, max(1, p // 8)) + 1))
        if n_mod + n_dual == 0:
            if rng.random() < 0.6:
                n_mod = 1
            else:
                n_dual = 1

    # Cap total active features
    max_act = min(prior.max_active, p - 2)
    while n_conf + n_prog + n_mod + n_dual > max_act:
        if n_prog > 0:
            n_prog -= 1
        elif n_conf > 1:
            n_conf -= 1
        elif n_mod > 1:
            n_mod -= 1
        elif n_dual > 0:
            n_dual -= 1
        else:
            break
    n_active = n_conf + n_prog + n_mod + n_dual

    # Build active + optional multi-level Dirichlet one-hot categorical blocks
    Z_act = _generate_factor_latents(n_total, n_active, min(prior.factor_noise_rank, 6), rng)
    X_act_scalar, bin_act_scalar = _realize_column_types(Z_act, rng)

    role_scalar = np.concatenate(
        [
            np.full(n_conf, ROLE_TO_ID["confounder"], dtype=np.int64),
            np.full(n_prog, ROLE_TO_ID["prognostic"], dtype=np.int64),
            np.full(n_mod, ROLE_TO_ID["modifier"], dtype=np.int64),
            np.full(n_dual, ROLE_TO_ID["dual_role"], dtype=np.int64),
        ]
    )

    cols_list, bin_list, role_list = [], [], []
    use_cats = rng.random() < prior.categorical_block_prob
    for j in range(n_active):
        # Expand ~25% of active features into 3- or 4-level mutually exclusive one-hot dummies
        if use_cats and (len(cols_list) + 4 <= p - 2) and rng.random() < 0.25:
            n_lvl = int(rng.integers(3, 5))
            dummies, _ = _make_categorical_dummy_block(Z_act[:, j], n_lvl, rng)
            for d_i in range(dummies.shape[1]):
                cols_list.append(dummies[:, d_i])
                bin_list.append(True)
                role_list.append(int(role_scalar[j]))
        else:
            cols_list.append(X_act_scalar[:, j])
            bin_list.append(bool(bin_act_scalar[j]))
            role_list.append(int(role_scalar[j]))

    X_core = np.stack(cols_list, axis=1).astype(np.float32)
    bin_core = np.asarray(bin_list, dtype=bool)
    roles_core = np.asarray(role_list, dtype=np.int64)

    X_std = (X_core - X_core.mean(axis=0, keepdims=True)) / np.maximum(
        X_core.std(axis=0, keepdims=True), 1e-5
    )

    mech = str(rng.choice(["linear", "linear", "threshold", "quadratic", "hinge", "saturating"]))

    def _xf(indices: np.ndarray) -> np.ndarray:
        if len(indices) == 0:
            return np.empty((n_total, 0), dtype=np.float32)
        out = []
        for c in indices:
            m = "linear" if bin_core[c] else (mech if rng.random() < 0.7 else "linear")
            out.append(_apply_nonlinear(X_std[:, c], m, rng))
        return np.stack(out, axis=1)

    conf_idx = np.where(roles_core == ROLE_TO_ID["confounder"])[0]
    prog_idx = np.where(roles_core == ROLE_TO_ID["prognostic"])[0]
    mod_idx = np.where(roles_core == ROLE_TO_ID["modifier"])[0]
    dual_idx = np.where(roles_core == ROLE_TO_ID["dual_role"])[0]

    # Treatment equation (confounders + dual_role; instruments added during padding)
    t_drivers = np.concatenate([conf_idx, dual_idx])
    t_signal = np.zeros(n_total, dtype=np.float32)
    if len(t_drivers):
        phi_t = _xf(t_drivers)
        t_signal += phi_t @ rng.normal(0.0, 1.0, size=len(t_drivers)).astype(np.float32)

    # Outcome baseline (50% pure-crossover modifiers with NO outcome main effect!)
    y_drivers = np.concatenate([conf_idx, prog_idx, dual_idx])
    if len(mod_idx) and rng.random() < 0.50:
        y_drivers = np.concatenate([y_drivers, mod_idx])
    y_base = np.zeros(n_total, dtype=np.float32)
    if len(y_drivers):
        phi_y = _xf(y_drivers)
        y_base += phi_y @ rng.normal(0.0, 1.0, size=len(y_drivers)).astype(np.float32)
        if len(y_drivers) >= 2 and rng.random() < 0.40:
            a, b = rng.choice(len(y_drivers), size=2, replace=False)
            y_base += float(rng.normal(0.0, 0.50)) * _standardize_vec(phi_y[:, a] * phi_y[:, b])

    # Treatment effect heterogeneity
    ht_drivers = np.concatenate([mod_idx, dual_idx])
    ht_part = np.zeros(n_total, dtype=np.float32)
    if len(ht_drivers):
        phi_ht = _xf(ht_drivers)
        ht_part += phi_ht @ rng.normal(0.0, 1.0, size=len(ht_drivers)).astype(np.float32)
        if len(ht_drivers) >= 2 and rng.random() < 0.30:
            a, b = rng.choice(len(ht_drivers), size=2, replace=False)
            ht_part += float(rng.normal(0.0, 0.55)) * _standardize_vec(phi_ht[:, a] * phi_ht[:, b])

    # Optional unmeasured confounding (matches reequation.py -> creates real Z-bias on instruments!)
    n_unmeasured = 0
    if rng.random() < prior.unmeasured_conf_prob:
        n_unmeasured = int(rng.integers(1, 4))
        U = rng.standard_normal((n_total, n_unmeasured)).astype(np.float32)
        u_strength = float(rng.uniform(0.25, 0.75))
        t_signal += u_strength * _standardize_vec(
            U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32)
        )
        y_base += u_strength * _standardize_vec(
            U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32)
        )

    # Optional proxy-only latent confounder (~15%): replace 1 confounder with 2-3 noisy proxies
    X_core_obs = X_core.copy()
    if len(conf_idx) >= 2 and (p - X_core.shape[1]) >= 4 and rng.random() < 0.15:
        drop_c = int(rng.choice(conf_idx))
        parent_z = _standardize_vec(X_core[:, drop_c])
        # Overwrite drop_c with its first noisy proxy and append 1-2 more correlated proxies
        rho1 = float(rng.uniform(0.70, 0.90))
        z_px1 = rho1 * parent_z + math.sqrt(1.0 - rho1 * rho1) * rng.standard_normal(n_total).astype(np.float32)
        X_core_obs[:, drop_c] = (z_px1 > 0.0).astype(np.float32) if bin_core[drop_c] else z_px1
        extra_px, extra_bin, extra_roles = [], [], []
        for _ in range(int(rng.integers(1, 3))):
            rho_k = float(rng.uniform(0.68, 0.90))
            zk = rho_k * parent_z + math.sqrt(1.0 - rho_k * rho_k) * rng.standard_normal(n_total).astype(np.float32)
            extra_px.append((zk > 0.0).astype(np.float32) if bin_core[drop_c] else zk)
            extra_bin.append(bool(bin_core[drop_c]))
            extra_roles.append(ROLE_TO_ID["confounder"])
        X_core_obs = np.concatenate([X_core_obs, np.stack(extra_px, axis=1)], axis=1)
        bin_core = np.concatenate([bin_core, np.asarray(extra_bin, dtype=bool)])
        roles_core = np.concatenate([roles_core, np.asarray(extra_roles, dtype=np.int64)])

    # Optional Stage-1 LLM extraction noise on active observed columns
    has_ext_noise = rng.random() < prior.extraction_noise_prob
    if has_ext_noise:
        X_core_obs = _inject_extraction_noise(
            X_core_obs, bin_core, np.arange(X_core_obs.shape[1]), rng
        )

    X_full, _, roles_full, t_signal = _pad_with_noise_proxies_and_instruments(
        X_core_obs, bin_core, roles_core, t_signal, p, prior, rng
    )
    t_obs, y_obs, true_prop, true_y0, true_y1, true_ite, scale_mode = _finalize_outcomes(
        t_signal, y_base, ht_part, not is_null_hte, prior, rng
    )
    X_obs, M, roles_full, miss_mode = _inject_missingness_and_3state(
        X_full, y_base, roles_full, prior, rng
    )

    perm = rng.permutation(X_obs.shape[1])
    return {
        "X": X_obs[:, perm],
        "M": M[:, perm],
        "t": t_obs,
        "y": y_obs,
        "true_propensity": true_prop,
        "true_y0": true_y0,
        "true_y1": true_y1,
        "true_ite": true_ite,
        "roles": roles_full[perm],
        "meta": {
            "family": "sparse_oci",
            "p": int(X_obs.shape[1]),
            "n_active": int((roles_full >= 2).sum()),
            "n_mod": int((roles_full == ROLE_TO_ID["modifier"]).sum()),
            "n_dual": int((roles_full == ROLE_TO_ID["dual_role"]).sum()),
            "n_inst": int((roles_full == ROLE_TO_ID["instrument"]).sum()),
            "n_unmeasured": n_unmeasured,
            "has_extraction_noise": has_ext_noise,
            "is_null_hte": is_null_hte,
            "mechanism": mech,
            "scale": scale_mode,
            "missingness": miss_mode,
        },
    }


# ==============================================================================
# FAMILY 2: OCI REEQUATION / SAMPLE-SIZE-CALCULATOR TABULAR DGP FAMILY
# ==============================================================================
def _sample_reequation_calc(
    n_total: int, p: int, prior: PriorConfig, rng: np.random.Generator
) -> dict:
    """Exact vectorized OCI reequation.py + sample_size_calculator/engine.R tabular DGP."""
    is_null_hte = bool(rng.random() < prior.null_hte_prob)
    n_conf = int(rng.integers(2, 7))
    n_prog = int(rng.integers(0, 5))
    n_inst = int(rng.integers(0, 4)) if rng.random() < prior.instrument_prob else 0
    if is_null_hte:
        n_mod, n_dual = 0, 0
    else:
        n_mod = int(rng.integers(0, 4))
        n_dual = int(rng.integers(0, 3))
        if n_mod + n_dual == 0:
            n_mod = 1

    group_roles = (
        [ROLE_TO_ID["confounder"]] * n_conf
        + [ROLE_TO_ID["prognostic"]] * n_prog
        + [ROLE_TO_ID["instrument"]] * n_inst
        + [ROLE_TO_ID["modifier"]] * n_mod
        + [ROLE_TO_ID["dual_role"]] * n_dual
    )
    max_groups = max(3, min(16, p // 2))
    if len(group_roles) > max_groups:
        # Keep confounders & modifiers first, trim excess prognostics/instruments when P is small
        keep_priority = sorted(
            range(len(group_roles)),
            key=lambda i: 0 if group_roles[i] in (3, 4, 5) else 1,
        )[:max_groups]
        group_roles = [group_roles[i] for i in sorted(keep_priority)]
    n_groups = len(group_roles)

    # Exchangeable Gaussian copula (matches sample_size_calculator/engine.R & config.py)
    rho = float(rng.uniform(0.0, 0.6))
    common = rng.standard_normal((n_total, 1)).astype(np.float32)
    latent = math.sqrt(rho) * common + math.sqrt(1.0 - rho) * rng.standard_normal(
        (n_total, n_groups)
    ).astype(np.float32)

    mech = str(rng.choice(["none", "none", "threshold", "quadratic"]))
    use_uniform_cdf = rng.random() < 0.5  # engine.R uniform-CDF vs config.py positive_lognormal

    cols, bins, col_roles = [], [], []
    group_effects = []
    for g_i, role_id in enumerate(group_roles):
        z = latent[:, g_i]
        kind = str(rng.choice(["continuous", "binary", "categorical"], p=[0.40, 0.35, 0.25]))
        if kind == "categorical" and (len(cols) + 4 > p - 1):
            kind = "binary"
        if kind == "continuous":
            if use_uniform_cdf:
                # engine.R: sqrt(3) * (2 * Phi(z) - 1)
                x_val = (math.sqrt(3.0) * (2.0 * ndtr(z) - 1.0)).astype(np.float32)
            else:
                # config.py: positive_lognormal
                sig = float(rng.uniform(0.25, 0.7))
                x_val = np.exp(np.clip(z * sig, -4.0, 4.0)).astype(np.float32)
            cols.append(x_val)
            bins.append(False)
            col_roles.append(role_id)
            z_std = _standardize_vec(x_val)
            eff = (
                (z_std > 0.0).astype(np.float32)
                if mech == "threshold"
                else (z_std * z_std if mech == "quadratic" else z_std)
            )
            group_effects.append(_standardize_vec(eff))
        elif kind == "binary":
            # Beta-drawn prevalence (matches population.py binary_mode="beta")
            prev = float(np.clip(rng.beta(1.5, 4.0), 0.04, 0.65))
            x_val = (z >= float(ndtri(1.0 - prev))).astype(np.float32)
            cols.append(x_val)
            bins.append(True)
            col_roles.append(role_id)
            group_effects.append(_standardize_vec(x_val))
        else:
            n_lvl = int(rng.integers(3, 5))
            dummies, level = _make_categorical_dummy_block(z, n_lvl, rng, concentration=1.5)
            # Either ordinal contrast (engine.R) or per-dummy weights (dgp_terms.py)
            if rng.random() < 0.5:
                eff = level.astype(np.float32) / float(n_lvl - 1)
            else:
                w_lvl = rng.uniform(-1.0, 1.0, size=n_lvl - 1).astype(np.float32)
                eff = dummies @ w_lvl
            for d_i in range(dummies.shape[1]):
                cols.append(dummies[:, d_i])
                bins.append(True)
                col_roles.append(role_id)
            group_effects.append(_standardize_vec(eff))

    X_core = np.stack(cols, axis=1).astype(np.float32)
    bin_core = np.asarray(bins, dtype=bool)
    roles_core = np.asarray(col_roles, dtype=np.int64)
    E_mat = np.stack(group_effects, axis=1).astype(np.float32)  # (N, n_groups)

    t_signal = np.zeros(n_total, dtype=np.float32)
    y_base = np.zeros(n_total, dtype=np.float32)
    ht_part = np.zeros(n_total, dtype=np.float32)
    mod_main = rng.random() < 0.45

    y_active_g = []
    for g_i, role_id in enumerate(group_roles):
        if role_id in (ROLE_TO_ID["confounder"], ROLE_TO_ID["dual_role"]):
            t_signal += float(rng.uniform(-1.0, 1.0)) * E_mat[:, g_i]
            y_base += float(rng.uniform(-1.0, 1.0)) * E_mat[:, g_i]
            y_active_g.append(g_i)
        elif role_id == ROLE_TO_ID["instrument"]:
            t_signal += float(rng.uniform(-1.5, 1.5)) * E_mat[:, g_i]
        elif role_id == ROLE_TO_ID["prognostic"]:
            y_base += float(rng.uniform(-1.0, 1.0)) * E_mat[:, g_i]
            y_active_g.append(g_i)
        if role_id in (ROLE_TO_ID["modifier"], ROLE_TO_ID["dual_role"]):
            ht_part += float(rng.uniform(-1.0, 1.0)) * E_mat[:, g_i]
            if role_id == ROLE_TO_ID["modifier"] and mod_main:
                y_base += float(rng.uniform(-0.7, 0.7)) * E_mat[:, g_i]
                y_active_g.append(g_i)

    # Pairwise prognostic interactions (matches longitudinal.py / reequation.py)
    if len(y_active_g) >= 2 and rng.random() < 0.50:
        a, b = rng.choice(y_active_g, size=2, replace=False)
        y_base += float(rng.uniform(-0.5, 0.5)) * _standardize_vec(E_mat[:, a] * E_mat[:, b])

    n_unmeasured = 0
    if rng.random() < prior.unmeasured_conf_prob:
        n_unmeasured = int(rng.integers(1, 3))
        U = rng.standard_normal((n_total, n_unmeasured)).astype(np.float32)
        u_str = float(rng.uniform(0.25, 0.75))
        t_signal += u_str * _standardize_vec(U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32))
        y_base += u_str * _standardize_vec(U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32))

    has_ext_noise = rng.random() < prior.extraction_noise_prob
    if has_ext_noise:
        X_core = _inject_extraction_noise(X_core, bin_core, np.arange(X_core.shape[1]), rng)

    X_full, _, roles_full, t_signal = _pad_with_noise_proxies_and_instruments(
        X_core, bin_core, roles_core, t_signal, p, prior, rng, allow_extra_instruments=(n_inst == 0)
    )
    t_obs, y_obs, true_prop, true_y0, true_y1, true_ite, scale_mode = _finalize_outcomes(
        t_signal, y_base, ht_part, not is_null_hte, prior, rng
    )
    X_obs, M, roles_full, miss_mode = _inject_missingness_and_3state(
        X_full, y_base, roles_full, prior, rng
    )

    perm = rng.permutation(X_obs.shape[1])
    return {
        "X": X_obs[:, perm],
        "M": M[:, perm],
        "t": t_obs,
        "y": y_obs,
        "true_propensity": true_prop,
        "true_y0": true_y0,
        "true_y1": true_y1,
        "true_ite": true_ite,
        "roles": roles_full[perm],
        "meta": {
            "family": "reequation_calc",
            "p": int(X_obs.shape[1]),
            "n_active": int((roles_full >= 2).sum()),
            "n_mod": int((roles_full == ROLE_TO_ID["modifier"]).sum()),
            "n_dual": int((roles_full == ROLE_TO_ID["dual_role"]).sum()),
            "n_inst": int((roles_full == ROLE_TO_ID["instrument"]).sum()),
            "n_unmeasured": n_unmeasured,
            "has_extraction_noise": has_ext_noise,
            "is_null_hte": is_null_hte,
            "mechanism": mech,
            "scale": scale_mode,
            "missingness": miss_mode,
        },
    }


# ==============================================================================
# FAMILY 3: NEMOTRON 5,000 ONCOLOGY STUDY DESIGNS (STRICT TRAIN / HELD_OUT SPLIT)
# ==============================================================================
def _sample_nemotron_design(
    n_total: int, p: int, prior: PriorConfig, rng: np.random.Generator, designs: list[dict[str, Any]]
) -> dict:
    """Sample a tabular study directly from one Nemotron oncology design padded to P."""
    design = designs[int(rng.integers(0, len(designs)))]
    meas = design["measurements"]
    finds = design["findings"]
    n_m = len(meas)
    n_f = len(finds)
    n_cand = n_m + n_f

    # Correlated clinical severity latent across measurements & findings (matches longitudinal.py)
    common = rng.standard_normal((n_total, 1)).astype(np.float32)
    rho_m = float(rng.uniform(0.20, 0.50))
    Z_all = math.sqrt(rho_m) * common + math.sqrt(1.0 - rho_m) * rng.standard_normal(
        (n_total, n_cand)
    ).astype(np.float32)

    cols = np.empty((n_total, n_cand), dtype=np.float32)
    is_bin = np.zeros(n_cand, dtype=bool)
    role_menus: list[tuple[str, ...]] = []

    for i, (name, mu, sd, r_menu) in enumerate(meas):
        val = mu + sd * Z_all[:, i]
        if name == "latest_ecog":
            val = np.clip(np.round(val), 0.0, 4.0)
        elif name == "recent_ed_visits":
            val = np.clip(np.round(np.maximum(val, 0.0)), 0.0, 8.0)
        elif name.startswith("pro."):
            val = np.clip(val, 0.0, 4.0)
        elif name.startswith("lab.") and not name.endswith(".change"):
            val = np.maximum(val, max(0.1 * mu, 0.05))
        cols[:, i] = val.astype(np.float32)
        role_menus.append(r_menu)

    for j, (prev, r_menu) in enumerate(finds):
        idx = n_m + j
        base_l = math.log(prev / max(1.0 - prev, 1e-4))
        ps = _expit(base_l + 0.35 * common[:, 0] + 0.35 * Z_all[:, idx])
        cols[:, idx] = (rng.random(n_total) < ps).astype(np.float32)
        is_bin[idx] = True
        role_menus.append(r_menu)

    # Assign realized roles from each candidate's clinically plausible role menu
    is_null_hte = bool(rng.random() < prior.null_hte_prob)
    role_map = {
        "unused": ROLE_TO_ID["noise"],
        "instrument": ROLE_TO_ID["instrument"],
        "prognostic": ROLE_TO_ID["prognostic"],
        "confounder": ROLE_TO_ID["confounder"],
        "effect_modifier": ROLE_TO_ID["modifier"],
        "dual_role": ROLE_TO_ID["dual_role"],
    }
    roles_core = np.zeros(n_cand, dtype=np.int64)
    for idx, menu in enumerate(role_menus):
        valid_ids = [role_map.get(r, 0) for r in menu]
        if is_null_hte:
            valid_ids = [r for r in valid_ids if r not in (ROLE_TO_ID["modifier"], ROLE_TO_ID["dual_role"])]
            if not valid_ids:
                valid_ids = [0]
        # Prefer sparsity when designs are wide (30-100 candidates)
        non_zero = [r for r in valid_ids if r != 0]
        if non_zero and rng.random() < min(0.65, 10.0 / max(n_cand, 10)):
            roles_core[idx] = int(rng.choice(non_zero))
        else:
            roles_core[idx] = 0

    # Ensure realistic minimum active signal (at least 1 confounder/prognostic, and >=1 modifier if non-null)
    if (roles_core >= 2).sum() < 2 and n_cand >= 2:
        chosen = rng.choice(n_cand, size=min(3, n_cand), replace=False)
        roles_core[chosen[0]] = ROLE_TO_ID["confounder"]
        if len(chosen) > 1:
            roles_core[chosen[1]] = ROLE_TO_ID["prognostic"]
    if not is_null_hte and not np.isin(roles_core, [ROLE_TO_ID["modifier"], ROLE_TO_ID["dual_role"]]).any():
        mod_cands = [
            i for i, m in enumerate(role_menus)
            if "effect_modifier" in m or "dual_role" in m
        ]
        pick = int(rng.choice(mod_cands)) if mod_cands else int(rng.integers(0, n_cand))
        roles_core[pick] = ROLE_TO_ID["dual_role"] if roles_core[pick] == ROLE_TO_ID["confounder"] else ROLE_TO_ID["modifier"]

    # Cap overly dense active sets so high-P sparsity holds
    active_ids = np.where(roles_core >= 2)[0]
    if len(active_ids) > prior.max_active:
        drop = rng.choice(active_ids, size=len(active_ids) - prior.max_active, replace=False)
        roles_core[drop] = 0

    X_std = (cols - cols.mean(axis=0, keepdims=True)) / np.maximum(
        cols.std(axis=0, keepdims=True), 1e-5
    )
    mech = str(rng.choice(["none", "none", "threshold", "quadratic"]))

    t_signal = np.zeros(n_total, dtype=np.float32)
    y_base = np.zeros(n_total, dtype=np.float32)
    ht_part = np.zeros(n_total, dtype=np.float32)
    mod_main = rng.random() < 0.55

    y_active = []
    for idx in range(n_cand):
        r = int(roles_core[idx])
        if r == 0:
            continue
        z = X_std[:, idx] if is_bin[idx] else _apply_nonlinear(X_std[:, idx], mech, rng)
        if r in (ROLE_TO_ID["confounder"], ROLE_TO_ID["dual_role"]):
            t_signal += float(rng.uniform(-1.0, 1.0)) * z
            y_base += float(rng.uniform(-1.0, 1.0)) * z
            y_active.append(idx)
        elif r == ROLE_TO_ID["instrument"]:
            t_signal += float(rng.uniform(-1.4, 1.4)) * z
        elif r == ROLE_TO_ID["prognostic"]:
            y_base += float(rng.uniform(-1.0, 1.0)) * z
            y_active.append(idx)
        if r in (ROLE_TO_ID["modifier"], ROLE_TO_ID["dual_role"]):
            ht_part += float(rng.uniform(-1.0, 1.0)) * z
            if r == ROLE_TO_ID["modifier"] and mod_main:
                y_base += float(rng.uniform(-0.7, 0.7)) * z
                y_active.append(idx)

    # Pairwise outcome interactions (matches _calibrate_longitudinal_equations)
    if len(y_active) >= 2 and rng.random() < 0.65:
        n_pairs = int(rng.integers(1, min(4, len(y_active)) + 1))
        for _ in range(n_pairs):
            a, b = rng.choice(y_active, size=2, replace=False)
            y_base += float(rng.uniform(-0.35, 0.35)) * _standardize_vec(X_std[:, a] * X_std[:, b])

    n_unmeasured = 0
    if rng.random() < prior.unmeasured_conf_prob:
        n_unmeasured = int(rng.integers(1, 3))
        U = rng.standard_normal((n_total, n_unmeasured)).astype(np.float32)
        u_str = float(rng.uniform(0.25, 0.70))
        t_signal += u_str * _standardize_vec(U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32))
        y_base += u_str * _standardize_vec(U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32))

    # Realistic Stage-1 extraction error on binary clinical note findings
    has_ext_noise = rng.random() < max(prior.extraction_noise_prob, 0.45)
    cols_obs = _inject_extraction_noise(cols, is_bin, np.where(is_bin)[0], rng) if has_ext_noise else cols

    # If target P < n_cand, keep all active + subsample unused design candidates so P matches
    if p < n_cand:
        act = np.where(roles_core != 0)[0]
        unused = np.where(roles_core == 0)[0]
        keep_unused = max(0, p - len(act))
        if keep_unused < len(unused):
            sel_u = rng.choice(unused, size=keep_unused, replace=False)
            keep = np.sort(np.concatenate([act, sel_u]))
            cols_obs = cols_obs[:, keep]
            is_bin = is_bin[keep]
            roles_core = roles_core[keep]

    X_full, _, roles_full, t_signal = _pad_with_noise_proxies_and_instruments(
        cols_obs,
        is_bin,
        roles_core,
        t_signal,
        p,
        prior,
        rng,
        allow_extra_instruments=not bool((roles_core == ROLE_TO_ID["instrument"]).any()),
    )
    t_obs, y_obs, true_prop, true_y0, true_y1, true_ite, scale_mode = _finalize_outcomes(
        t_signal, y_base, ht_part, not is_null_hte, prior, rng
    )
    X_obs, M, roles_full, miss_mode = _inject_missingness_and_3state(
        X_full, y_base, roles_full, prior, rng
    )

    perm = rng.permutation(X_obs.shape[1])
    return {
        "X": X_obs[:, perm],
        "M": M[:, perm],
        "t": t_obs,
        "y": y_obs,
        "true_propensity": true_prop,
        "true_y0": true_y0,
        "true_y1": true_y1,
        "true_ite": true_ite,
        "roles": roles_full[perm],
        "meta": {
            "family": "nemotron_design",
            "design_id": design["id"],
            "cancer": design["cancer"],
            "design_split": prior.design_split,
            "p": int(X_obs.shape[1]),
            "n_active": int((roles_full >= 2).sum()),
            "n_mod": int((roles_full == ROLE_TO_ID["modifier"]).sum()),
            "n_dual": int((roles_full == ROLE_TO_ID["dual_role"]).sum()),
            "n_inst": int((roles_full == ROLE_TO_ID["instrument"]).sum()),
            "n_unmeasured": n_unmeasured,
            "has_extraction_noise": has_ext_noise,
            "is_null_hte": is_null_hte,
            "mechanism": mech,
            "scale": scale_mode,
            "missingness": miss_mode,
        },
    }


def sample_tabular_study(
    seed: int,
    n_total: int,
    *,
    num_features: int | None = None,
    prior: PriorConfig = PriorConfig(),
    family: str | None = None,
) -> dict:
    """Sample one synthetic high-dimensional sparse tabular causal study from the OCI mixture prior.

    Returns a dictionary containing:
      - ``X``: ``(N, P)`` float32 array with ``np.nan`` for missing values
      - ``M``: ``(N, P)`` float32 missingness indicator (1.0 where missing, else 0.0)
      - ``t``: ``(N,)`` float32 binary treatment indicator
      - ``y``: ``(N,)`` float32 binary outcome indicator
      - ``true_propensity``: ``(N,)`` float32 true treatment probability
      - ``true_y0``: ``(N,)`` float32 true control outcome probability
      - ``true_y1``: ``(N,)`` float32 true treated outcome probability
      - ``true_ite``: ``(N,)`` float32 true ``y1 - y0``
      - ``roles``: ``(P,)`` int64 ground-truth role IDs (``ROLE_NAMES``)
      - ``meta``: dict of study-level metadata (``family, p, n_active, is_null_hte, ...``)
    """
    rng = np.random.default_rng(seed)

    if num_features is None:
        log_p = rng.uniform(math.log(prior.min_features), math.log(prior.max_features))
        p = int(round(math.exp(log_p)))
    else:
        p = int(num_features)
    p = max(p, 6)

    designs = _load_design_split(prior.design_bank_path, prior.design_split)
    if family is not None:
        if family not in DGP_FAMILIES:
            raise ValueError(f"Unknown DGP family {family!r}; expected one of {DGP_FAMILIES}")
        chosen = family if (family != "nemotron_design" or designs) else "sparse_oci"
    else:
        w = np.asarray(prior.family_weights, dtype=np.float64)
        if not designs:
            w[2] = 0.0
        if w.sum() <= 0:
            w = np.array([1.0, 0.0, 0.0])
        w /= w.sum()
        chosen = str(rng.choice(DGP_FAMILIES, p=w))

    if chosen == "reequation_calc":
        return _sample_reequation_calc(n_total, p, prior, rng)
    if chosen == "nemotron_design":
        return _sample_nemotron_design(n_total, p, prior, rng, designs)
    return _sample_sparse_oci(n_total, p, prior, rng)
