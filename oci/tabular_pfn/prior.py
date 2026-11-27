"""Vectorized OCI-aligned 6-family high-dimensional sparse tabular causal prior.

Generates synthetic observational studies with N up to 10,000+ and P up to 4,000+
candidate features by mixing six fast vectorized DGP families:

  1. ``sparse_oci`` (default 20%):
     High-dimensional all-evidence clinical tables with 50–99.5% noise, explicit
     6-class causal roles (``noise``, ``instrument``, ``prognostic``, ``confounder``,
     ``modifier``, ``dual_role``), multi-level mutually exclusive Dirichlet one-hot
     categorical blocks, heavy-tailed labs & zero-inflated counts, optional dense
     weak-confounding regimes, collinear noisy proxies, block-correlated factor
     noise, rich 1D nonlinearities + confounder×confounder / confounder×modifier /
     modifier×modifier interactions (20% pure-epistatic), optional unmeasured
     confounding (Z-bias!), Stage-1 LLM extraction noise, fragmented synonymous
     extraction clusters, MCAR/MAR/MNAR + 3-state EHR documentation, occasional
     RCTs, strictly null-HTE studies, and both ``logOR`` and additive ``RD`` scales.
  2. ``reequation_calc`` (default 12%):
     Exact vectorized implementation of OCI's ``sample_size_calculator/engine.R``
     and ``synthetic_data/reequation.py`` tabular DGP family with optional
     confounder×confounder and confounder×modifier interactions embedded inside
     ``P = 20..4,000`` realistic Stage-2 noise/proxy/instrument/fragmented columns.
  3. ``nemotron_design`` (default 18% when bundled ``nemotron_5000.json.gz`` exists):
     Direct tabular episodes drawn from the 4,000 ``split == "train"`` oncology
     study designs in ``synthetic_data/designs/nemotron_5000.json.gz`` with rich
     clinical interactions embedded inside ``P = 20..4,000`` candidate columns.
     The 1,000 ``split == "held_out"`` designs (including 3 unseen cancers) are
     strictly excluded from training and reserved for zero-shot eval.
  4. ``tree_subgroup`` (default 18%):
     Clinical decision-tree / multi-way biomarker rule-list subgroup DGP (depth 2–4
     conjunctions including explicit confounder×modifier rules, discontinuous
     cutoffs, piecewise subgroups, near-positivity pockets).
  5. ``causal_scm`` (default 16%):
     General nonlinear Structural Causal Model (layered DAG + 2-layer MLP treatment,
     prognostic, and joint confounder×modifier HTE surfaces) with explicit
     observed-parent and latent-parent M-bias colliders, heavy tails, dense weak
     confounding, and harsh positivity.
  6. ``interacting_clinical`` (default 16%):
     Interaction-heavy clinical & multimodal-embedding hybrid prior: coupled
     confounder×confounder propensity interactions, prognostic×confounder outcome
     interactions, strong confounder×modifier / prognostic×modifier /
     modifier×modifier HTE (30% pure-epistatic crossover with zero marginal modifier
     slope!), and in 50% of wide cohorts a bounded correlated cosine-MaxSim
     embedding block mimicking frozen clinical-text representations (e.g.
     ``TinyOncBERT SentKM``) coupled with structured covariates.
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
DGP_FAMILIES = (
    "sparse_oci",
    "reequation_calc",
    "nemotron_design",
    "tree_subgroup",
    "causal_scm",
    "interacting_clinical",
    "universal_nonparametric",
)

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
    # OCI-aligned 7-family universal multi-DGP mixture settings
    family_weights: tuple[float, ...] = (0.16, 0.10, 0.16, 0.16, 0.14, 0.14, 0.14)
    design_bank_path: str | None = None
    design_split: str = "train"
    unmeasured_conf_prob: float = 0.25
    extraction_noise_prob: float = 0.35
    fragmented_dup_prob: float = 0.55
    collider_prob: float = 0.45
    dense_conf_prob: float = 0.15
    mnar_prob: float = 0.50
    rd_scale_prob: float = 0.25
    rct_prob: float = 0.08
    max_treatment_logit_std: float = 3.2
    categorical_block_prob: float = 0.70

    def __post_init__(self):
        fw = tuple(float(w) for w in self.family_weights)
        # Gracefully pad legacy 3-, 5-, or 6-family checkpoints with zeros for newer families
        if len(fw) < len(DGP_FAMILIES) and len(fw) in (3, 5, 6):
            fw = fw + (0.0,) * (len(DGP_FAMILIES) - len(fw))
        object.__setattr__(self, "family_weights", fw)
        if len(self.family_weights) != len(DGP_FAMILIES) or any(w < 0 for w in self.family_weights):
            raise ValueError(f"family_weights must be {len(DGP_FAMILIES)} non-negative floats")
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
    elif mode == "band":
        # Non-monotonic clinical reference range (e.g. abnormal low OR high lab)
        lo = float(rng.uniform(-1.3, -0.3))
        hi = float(rng.uniform(0.4, 1.4))
        out = ((z < lo) | (z > hi)).astype(np.float32)
    elif mode == "asym_sig":
        thresh = float(rng.uniform(-0.8, 0.9))
        slope = float(rng.uniform(1.8, 4.0))
        out = _expit(slope * (z - thresh)).astype(np.float32)
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

    # 40% binary, 14% ordinal (0..4), 22% Gaussian, 12% skewed log-normal,
    # 6% heavy-tailed Student-t (lab outliers), 6% zero-inflated counts (ED/hosp visits)
    u = rng.random(p)
    bin_cols = np.where(u < 0.40)[0]
    ord_cols = np.where((u >= 0.40) & (u < 0.54))[0]
    log_cols = np.where((u >= 0.76) & (u < 0.88))[0]
    ht_cols = np.where((u >= 0.88) & (u < 0.94))[0]
    cnt_cols = np.where(u >= 0.94)[0]

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
        scales = rng.uniform(0.3, 0.85, size=len(log_cols)).astype(np.float32)
        X[:, log_cols] = np.exp(np.clip(Z[:, log_cols] * scales[None, :], -4.0, 4.0))

    if len(ht_cols):
        # Heavy-tailed Student-t-like lab values (df ~ 3) with realistic extreme outliers
        df = rng.uniform(2.5, 4.5, size=len(ht_cols)).astype(np.float32)
        chi = rng.chisquare(df[None, :], size=(n, len(ht_cols))).astype(np.float32)
        scale_t = np.sqrt(df[None, :] / np.maximum(chi, 0.25))
        X[:, ht_cols] = np.clip(Z[:, ht_cols] * scale_t, -18.0, 18.0).astype(np.float32)

    if len(cnt_cols):
        # Zero-inflated clinical utilization counts (hospitalizations, ED visits, met sites)
        zi_cut = rng.uniform(-0.2, 1.1, size=len(cnt_cols)).astype(np.float32)
        rate = np.exp(np.clip(0.85 * Z[:, cnt_cols] - 0.3, -3.0, 2.6))
        cnt = rng.poisson(rate).astype(np.float32)
        X[:, cnt_cols] = np.where(Z[:, cnt_cols] > zi_cut[None, :], np.minimum(cnt, 25.0), 0.0)

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


def _inject_fragmented_extractions(
    X_core: np.ndarray,
    bin_core: np.ndarray,
    roles_core: np.ndarray,
    p_target: int,
    prior: PriorConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Simulate real Stage-1 LLM fragmented & synonymous extraction clusters.

    Mimics real OCI Stage-2 tables where one latent clinical variable (e.g. sex, age,
    ECOG) appears as 2-6 collinear, opposite-polarity, coarsened, and/or partially-missing
    synonymous columns (e.g. ``gender==male``, ``biological_sex==female``,
    ``ecog_components``, ``activity_tolerance``, ``performance_status``).
    """
    n_total, p_cur = X_core.shape
    spare = p_target - p_cur
    if spare < 3 or rng.random() >= prior.fragmented_dup_prob:
        return X_core, bin_core, roles_core, 0

    active_idx = np.where(roles_core >= 2)[0]
    if len(active_idx) == 0:
        return X_core, bin_core, roles_core, 0

    X_out = X_core.copy()
    n_parents = int(rng.integers(1, min(5, len(active_idx), max(2, spare // 2)) + 1))
    parents = rng.choice(active_idx, size=n_parents, replace=False)

    extra_cols: list[np.ndarray] = []
    extra_bins: list[bool] = []
    extra_roles: list[int] = []

    for c in parents:
        rem_budget = spare - len(extra_cols)
        if rem_budget < 1:
            break
        col = X_core[:, c]
        fin = np.isfinite(col)
        if fin.sum() < 10:
            continue
        mu = float(col[fin].mean())
        sd = float(col[fin].std())
        if sd < 1e-5:
            continue

        is_b = bool(bin_core[c])
        role_c = int(roles_core[c])
        k_dup = int(rng.integers(1, min(5, rem_budget) + 1))
        made_partial = False

        for d_i in range(k_dup):
            mode = float(rng.random())
            if is_b and d_i == 0 and mode < 0.55:
                # Opposite-polarity binary twin (e.g., sex==female alongside sex==male)
                dup = (1.0 - col).astype(np.float32)
                dup_bin = True
                if rng.random() < 0.35:
                    fl = fin & (rng.random(n_total) < float(rng.uniform(0.01, 0.06)))
                    dup[fl] = 1.0 - dup[fl]
                if rng.random() < 0.45:
                    miss = rng.random(n_total) < float(rng.uniform(0.15, 0.55))
                    if (~miss & fin).sum() >= 6:
                        dup[miss] = np.nan
                        made_partial = True
            elif (not is_b) and mode < 0.35:
                # Coarsened clinical cutoff indicator (e.g., age_group==elderly alongside age)
                cut = mu + float(rng.uniform(-0.7, 0.9)) * sd
                sign = 1.0 if rng.random() < 0.5 else -1.0
                dup = ((sign * (col - cut)) > 0.0).astype(np.float32)
                dup[~fin] = np.nan
                dup_bin = True
                if rng.random() < 0.45:
                    miss = rng.random(n_total) < float(rng.uniform(0.15, 0.55))
                    if (~miss & fin).sum() >= 6:
                        dup[miss] = np.nan
                        made_partial = True
            else:
                # Partial-coverage noisy synonymous extraction (e.g. ecog_components / performance_status)
                if is_b:
                    dup = col.copy()
                    fn_r = float(rng.uniform(0.02, 0.14))
                    fp_r = float(rng.uniform(0.01, 0.06))
                    pos = fin & (dup > 0.5)
                    neg = fin & (dup <= 0.5)
                    dup[pos & (rng.random(n_total) < fn_r)] = 0.0
                    dup[neg & (rng.random(n_total) < fp_r)] = 1.0
                    dup_bin = True
                else:
                    scale_s = float(rng.uniform(0.85, 1.15)) * (1.0 if rng.random() < 0.85 else -1.0)
                    noise_s = float(rng.uniform(0.10, 0.35)) * sd
                    dup = (scale_s * col + rng.normal(0.0, noise_s, size=n_total)).astype(np.float32)
                    dup[~fin] = np.nan
                    dup_bin = False
                # Observe on a random 30-80% patient subset (fragmented documentation)
                if rng.random() < 0.75:
                    miss_rate = float(rng.uniform(0.20, 0.68))
                    miss = rng.random(n_total) < miss_rate
                    if (~miss & fin).sum() >= 6:
                        dup[miss] = np.nan
                        made_partial = True

            extra_cols.append(dup)
            extra_bins.append(dup_bin)
            extra_roles.append(role_c)

        # In 40% of fragmented clusters, also make the original column partial-coverage
        # so the model MUST pool across the fragmented synonym cluster rather than relying on 1 column!
        if made_partial and k_dup >= 2 and rng.random() < 0.40:
            miss_orig = rng.random(n_total) < float(rng.uniform(0.30, 0.60))
            if (~miss_orig & fin).sum() >= 6:
                X_out[miss_orig, c] = np.nan

    if not extra_cols:
        return X_out, bin_core, roles_core, 0

    X_cat = np.concatenate([X_out, np.stack(extra_cols, axis=1)], axis=1).astype(np.float32)
    bin_cat = np.concatenate([bin_core, np.asarray(extra_bins, dtype=bool)])
    roles_cat = np.concatenate([roles_core, np.asarray(extra_roles, dtype=np.int64)])
    return X_cat, bin_cat, roles_cat, len(extra_cols)


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

    Preserves any pre-existing fragmented NaNs in ``X_in`` cleanly and guarantees that
    noise columns span the full EHR missingness distribution (including heavy Stage-1
    extraction missingness up to ~95%) so missingness rate can never act as a role shortcut.
    Returns ``(X_obs, M, roles_out, miss_mode)`` keeping total feature count exact.
    """
    n_total, p = X_in.shape
    X_obs = X_in.copy()
    pre_nan = ~np.isfinite(X_obs)
    has_pre_nan = bool(pre_nan.any())
    M = pre_nan.astype(np.float32)
    roles_out = roles.copy()
    # Never leave pre-existing fragmented NaNs solely on active columns: whenever any
    # fragmented NaNs exist, always apply study-wide missingness across ALL columns.
    if (not has_pre_nan) and rng.random() >= prior.missing_study_prob:
        return X_obs, M, roles_out, "none"

    is_mnar = rng.random() < prior.mnar_prob
    miss_mode = "mnar" if is_mnar else "mar"

    # Mix moderate EHR missingness (up to max_missing_rate) with heavy Stage-1 LLM extraction
    # missingness (heavy tail up to 0.96 across ALL columns, matching real Stage-2 tables!)
    heavy_ehr = has_pre_nan or (rng.random() < 0.50)
    if heavy_ehr:
        # Realistic bimodal/heavy-tailed missingness across noise & active columns alike
        u = rng.random(p)
        miss_rates = np.zeros(p, dtype=np.float32)
        mod_mask = (u >= 0.20) & (u < 0.55)
        hi_mask = u >= 0.55
        if mod_mask.any():
            miss_rates[mod_mask] = rng.uniform(0.02, prior.max_missing_rate, size=int(mod_mask.sum())).astype(np.float32)
        if hi_mask.any():
            miss_rates[hi_mask] = rng.uniform(prior.max_missing_rate, 0.95, size=int(hi_mask.sum())).astype(np.float32)
    else:
        miss_rates = rng.uniform(0.0, prior.max_missing_rate, size=p).astype(np.float32)
        miss_rates *= (rng.random(p) < 0.65).astype(np.float32)

    active_miss = np.where(miss_rates > 0.015)[0]
    if len(active_miss) == 0:
        return X_obs, M, roles_out, ("fragmented" if has_pre_nan else "none")

    b_m = np.log(miss_rates[active_miss] / np.maximum(1.0 - miss_rates[active_miss], 1e-4))
    if is_mnar:
        # True MNAR: documentation probability depends on the column's OWN true standardized value
        X_sub = X_in[:, active_miss]
        fin_sub = np.isfinite(X_sub)
        obs_cnt = np.maximum(fin_sub.sum(axis=0, keepdims=True), 1.0)
        mu = np.where(fin_sub, X_sub, 0.0).sum(axis=0, keepdims=True) / obs_cnt
        X_filled = np.where(fin_sub, X_sub, mu)
        sd = np.maximum(X_filled.std(axis=0, keepdims=True), 1e-4)
        Z_sub = np.clip((X_filled - mu) / sd, -4.0, 4.0)
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
    combined = pre_nan[:, active_miss] | mask
    min_obs = min(6, max(2, n_total // 20))
    valid_mask_cols = combined.sum(axis=0) <= (n_total - min_obs)
    mask[:, ~valid_mask_cols] = False
    M[:, active_miss] = np.where(mask, 1.0, M[:, active_miss])
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
    # Build collinear proxies of active columns (NaN-safe for fragmented extractions)
    if n_proxies > 0:
        parents = rng.choice(active_idx, size=n_proxies, replace=True)
        X_par = X_base[:, parents]
        fin_par = np.isfinite(X_par)
        cnt_par = np.maximum(fin_par.sum(axis=0, keepdims=True), 1.0)
        mu_par = np.where(fin_par, X_par, 0.0).sum(axis=0, keepdims=True) / cnt_par
        X_par_fill = np.where(fin_par, X_par, mu_par)
        X_par_std = (X_par_fill - X_par_fill.mean(axis=0, keepdims=True)) / np.maximum(
            X_par_fill.std(axis=0, keepdims=True), 1e-5
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

    # Also build fragmented synonymous clusters on pure NOISE columns (matching active fragmented
    # clusters: opposite-polarity pairs, coarsened cuts, & partial-coverage clones) so synonym
    # cluster geometry / missingness can never shortcut-predict role != 0!
    noise_start = n_inst + n_proxies
    if n_noise >= 6 and rng.random() < max(prior.fragmented_dup_prob, 0.50):
        n_noise_clusters = int(rng.integers(2, min(14, n_noise // 3) + 1))
        cur = noise_start
        for _ in range(n_noise_clusters):
            if cur + 2 >= X_pad.shape[1]:
                break
            parent_c = cur
            cur += 1
            k_dup = int(rng.integers(1, min(5, X_pad.shape[1] - cur) + 1))
            col = X_pad[:, parent_c]
            mu_c = float(col.mean())
            sd_c = float(col.std())
            is_b = bool(bin_pad[parent_c])
            made_part = False
            for d_i in range(k_dup):
                dst = cur
                cur += 1
                if is_b and d_i == 0 and rng.random() < 0.55:
                    dup = (1.0 - col).astype(np.float32)
                    bin_pad[dst] = True
                elif (not is_b) and sd_c > 1e-5 and rng.random() < 0.35:
                    cut = mu_c + float(rng.uniform(-0.7, 0.9)) * sd_c
                    dup = ((col - cut) > 0.0).astype(np.float32)
                    bin_pad[dst] = True
                else:
                    if is_b:
                        dup = col.copy()
                        pos = dup > 0.5
                        dup[pos & (rng.random(n_total) < float(rng.uniform(0.02, 0.14)))] = 0.0
                        dup[(~pos) & (rng.random(n_total) < float(rng.uniform(0.01, 0.06)))] = 1.0
                        bin_pad[dst] = True
                    else:
                        dup = (
                            col + rng.normal(0.0, float(rng.uniform(0.10, 0.35)) * max(sd_c, 1e-4), size=n_total)
                        ).astype(np.float32)
                        bin_pad[dst] = False
                if rng.random() < 0.70:
                    miss_r = float(rng.uniform(0.20, 0.85))
                    miss = rng.random(n_total) < miss_r
                    if (~miss).sum() >= 6:
                        dup[miss] = np.nan
                        made_part = True
                X_pad[:, dst] = dup
            if made_part and k_dup >= 2 and rng.random() < 0.45:
                miss_p = rng.random(n_total) < float(rng.uniform(0.30, 0.80))
                if (~miss_p).sum() >= 6:
                    X_pad[miss_p, parent_c] = np.nan

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
    is_dense_conf = (p >= 80) and (rng.random() < prior.dense_conf_prob)
    if is_dense_conf:
        # Dense polygenic/multi-biomarker regime (12-42 confounders, 4-26 modifiers)
        n_conf = int(rng.integers(12, min(42, max(13, p // 3)) + 1))
        n_prog = int(rng.integers(4, min(14, max(5, p // 6)) + 1))
    else:
        n_conf = int(rng.integers(1, min(8, max(2, p // 4)) + 1))
        n_prog = int(rng.integers(0, min(7, max(1, p // 5)) + 1))
    if is_null_hte:
        n_mod, n_dual = 0, 0
    elif is_dense_conf:
        n_mod = int(rng.integers(4, min(26, max(5, p // 4)) + 1))
        n_dual = int(rng.integers(1, min(8, max(2, p // 8)) + 1))
    else:
        n_mod = int(rng.integers(0, min(5, max(1, p // 6)) + 1))
        n_dual = int(rng.integers(0, min(3, max(1, p // 8)) + 1))
        if n_mod + n_dual == 0:
            if rng.random() < 0.6:
                n_mod = 1
            else:
                n_dual = 1

    # Cap total active features (keeps >96% noise at P = 2,000 while supporting dense 50+ active at P=120)
    max_act = min(72 if is_dense_conf else prior.max_active, max(4, p - 4))
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

    mech = str(
        rng.choice(
            ["linear", "linear", "threshold", "quadratic", "hinge", "saturating", "band", "asym_sig"]
        )
    )

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

    # Treatment equation (confounders + dual_role + optional confounder x confounder interactions)
    t_drivers = np.concatenate([conf_idx, dual_idx])
    t_signal = np.zeros(n_total, dtype=np.float32)
    if len(t_drivers):
        phi_t = _xf(t_drivers)
        t_signal += phi_t @ rng.normal(0.0, 1.0, size=len(t_drivers)).astype(np.float32)
        if len(t_drivers) >= 2 and rng.random() < 0.45:
            n_tp = int(rng.integers(1, min(3, len(t_drivers)) + 1))
            for _ in range(n_tp):
                a, b = rng.choice(len(t_drivers), size=2, replace=False)
                t_signal += float(rng.normal(0.0, 0.55)) * _standardize_vec(phi_t[:, a] * phi_t[:, b])

    # Outcome baseline (50% pure-crossover modifiers with NO outcome main effect!)
    y_drivers = np.concatenate([conf_idx, prog_idx, dual_idx])
    if len(mod_idx) and rng.random() < 0.50:
        y_drivers = np.concatenate([y_drivers, mod_idx])
    y_base = np.zeros(n_total, dtype=np.float32)
    if len(y_drivers):
        phi_y = _xf(y_drivers)
        y_base += phi_y @ rng.normal(0.0, 1.0, size=len(y_drivers)).astype(np.float32)
        if len(y_drivers) >= 2 and rng.random() < 0.55:
            n_yp = int(rng.integers(1, min(4, len(y_drivers)) + 1))
            for _ in range(n_yp):
                a, b = rng.choice(len(y_drivers), size=2, replace=False)
                y_base += float(rng.normal(0.0, 0.50)) * _standardize_vec(phi_y[:, a] * phi_y[:, b])

    # Treatment effect heterogeneity (including modifier x modifier AND confounder x modifier!)
    ht_drivers = np.concatenate([mod_idx, dual_idx])
    ht_part = np.zeros(n_total, dtype=np.float32)
    if len(ht_drivers):
        phi_ht = _xf(ht_drivers)
        # 20% pure epistatic HTE (weak/zero main linear slope, driven by interactions!)
        pure_epi = (len(ht_drivers) >= 2 or len(conf_idx) >= 1) and (rng.random() < 0.20)
        if not pure_epi:
            ht_part += phi_ht @ rng.normal(0.0, 1.0, size=len(ht_drivers)).astype(np.float32)
        if len(ht_drivers) >= 2 and (pure_epi or rng.random() < 0.45):
            n_mp = int(rng.integers(1, min(3, len(ht_drivers)) + 1))
            for _ in range(n_mp):
                a, b = rng.choice(len(ht_drivers), size=2, replace=False)
                ht_part += float(rng.normal(0.0, 0.70)) * _standardize_vec(phi_ht[:, a] * phi_ht[:, b])
        # Confounder x Modifier HTE interaction (e.g. biomarker benefit gated by age/ECOG!)
        if len(conf_idx) >= 1 and (pure_epi or rng.random() < 0.45):
            n_cm = int(rng.integers(1, min(3, len(conf_idx) * len(ht_drivers)) + 1))
            phi_c_all = _xf(conf_idx)
            for _ in range(n_cm):
                ci = int(rng.integers(0, len(conf_idx)))
                mi = int(rng.integers(0, len(ht_drivers)))
                ht_part += float(rng.normal(0.0, 0.75)) * _standardize_vec(phi_c_all[:, ci] * phi_ht[:, mi])
                # Accurate ground-truth role: confounder participating in HTE becomes dual_role
                roles_core[conf_idx[ci]] = ROLE_TO_ID["dual_role"]

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

    # Optional Stage-1 LLM extraction noise + fragmented synonym clusters
    has_ext_noise = rng.random() < prior.extraction_noise_prob
    if has_ext_noise:
        X_core_obs = _inject_extraction_noise(
            X_core_obs, bin_core, np.arange(X_core_obs.shape[1]), rng
        )
    X_core_obs, bin_core, roles_core, n_frag_dups = _inject_fragmented_extractions(
        X_core_obs, bin_core, roles_core, p, prior, rng
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
            "n_frag_dups": n_frag_dups,
            "is_dense_conf": is_dense_conf,
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
    group_col_spans: list[tuple[int, int]] = []
    for g_i, role_id in enumerate(group_roles):
        c_start = len(cols)
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
        group_col_spans.append((c_start, len(cols)))

    X_core = np.stack(cols, axis=1).astype(np.float32)
    bin_core = np.asarray(bins, dtype=bool)
    roles_core = np.asarray(col_roles, dtype=np.int64)
    E_mat = np.stack(group_effects, axis=1).astype(np.float32)  # (N, n_groups)

    t_signal = np.zeros(n_total, dtype=np.float32)
    y_base = np.zeros(n_total, dtype=np.float32)
    ht_part = np.zeros(n_total, dtype=np.float32)
    mod_main = rng.random() < 0.45

    conf_g, ht_g, y_active_g = [], [], []
    for g_i, role_id in enumerate(group_roles):
        if role_id in (ROLE_TO_ID["confounder"], ROLE_TO_ID["dual_role"]):
            t_signal += float(rng.uniform(-1.0, 1.0)) * E_mat[:, g_i]
            y_base += float(rng.uniform(-1.0, 1.0)) * E_mat[:, g_i]
            y_active_g.append(g_i)
            if role_id == ROLE_TO_ID["confounder"]:
                conf_g.append(g_i)
        elif role_id == ROLE_TO_ID["instrument"]:
            t_signal += float(rng.uniform(-1.5, 1.5)) * E_mat[:, g_i]
        elif role_id == ROLE_TO_ID["prognostic"]:
            y_base += float(rng.uniform(-1.0, 1.0)) * E_mat[:, g_i]
            y_active_g.append(g_i)
        if role_id in (ROLE_TO_ID["modifier"], ROLE_TO_ID["dual_role"]):
            ht_part += float(rng.uniform(-1.0, 1.0)) * E_mat[:, g_i]
            ht_g.append(g_i)
            if role_id == ROLE_TO_ID["modifier"] and mod_main:
                y_base += float(rng.uniform(-0.7, 0.7)) * E_mat[:, g_i]
                y_active_g.append(g_i)

    # Pairwise confounder x confounder propensity interactions
    t_conf_g = [g for g, r in enumerate(group_roles) if r in (ROLE_TO_ID["confounder"], ROLE_TO_ID["dual_role"])]
    if len(t_conf_g) >= 2 and rng.random() < 0.40:
        n_tp = int(rng.integers(1, min(3, len(t_conf_g)) + 1))
        for _ in range(n_tp):
            a, b = rng.choice(t_conf_g, size=2, replace=False)
            t_signal += float(rng.uniform(-0.55, 0.55)) * _standardize_vec(E_mat[:, a] * E_mat[:, b])

    # Pairwise prognostic / confounder outcome interactions
    if len(y_active_g) >= 2 and rng.random() < 0.55:
        n_yp = int(rng.integers(1, min(4, len(y_active_g)) + 1))
        for _ in range(n_yp):
            a, b = rng.choice(y_active_g, size=2, replace=False)
            y_base += float(rng.uniform(-0.5, 0.5)) * _standardize_vec(E_mat[:, a] * E_mat[:, b])

    # Pairwise modifier x modifier and confounder x modifier HTE interactions
    if not is_null_hte and len(ht_g) >= 2 and rng.random() < 0.40:
        n_mp = int(rng.integers(1, min(3, len(ht_g)) + 1))
        for _ in range(n_mp):
            a, b = rng.choice(ht_g, size=2, replace=False)
            ht_part += float(rng.uniform(-0.65, 0.65)) * _standardize_vec(E_mat[:, a] * E_mat[:, b])
    if not is_null_hte and len(ht_g) >= 1 and len(conf_g) >= 1 and rng.random() < 0.45:
        n_cm = int(rng.integers(1, min(3, len(conf_g) * len(ht_g)) + 1))
        for _ in range(n_cm):
            cg = int(rng.choice(conf_g))
            mg = int(rng.choice(ht_g))
            ht_part += float(rng.uniform(-0.75, 0.75)) * _standardize_vec(E_mat[:, cg] * E_mat[:, mg])
            s0, s1 = group_col_spans[cg]
            roles_core[s0:s1] = ROLE_TO_ID["dual_role"]

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
    X_core, bin_core, roles_core, n_frag_dups = _inject_fragmented_extractions(
        X_core, bin_core, roles_core, p, prior, rng
    )

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
            "n_frag_dups": n_frag_dups,
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

    # Pairwise confounder x confounder propensity interactions
    t_active = [i for i in range(n_cand) if roles_core[i] in (ROLE_TO_ID["confounder"], ROLE_TO_ID["dual_role"])]
    if len(t_active) >= 2 and rng.random() < 0.40:
        n_tp = int(rng.integers(1, min(3, len(t_active)) + 1))
        for _ in range(n_tp):
            a, b = rng.choice(t_active, size=2, replace=False)
            t_signal += float(rng.uniform(-0.45, 0.45)) * _standardize_vec(X_std[:, a] * X_std[:, b])

    # Pairwise outcome interactions (matches _calibrate_longitudinal_equations)
    if len(y_active) >= 2 and rng.random() < 0.65:
        n_pairs = int(rng.integers(1, min(4, len(y_active)) + 1))
        for _ in range(n_pairs):
            a, b = rng.choice(y_active, size=2, replace=False)
            y_base += float(rng.uniform(-0.35, 0.35)) * _standardize_vec(X_std[:, a] * X_std[:, b])

    # Pairwise modifier x modifier and confounder x modifier HTE interactions
    if not is_null_hte:
        ht_active = [i for i in range(n_cand) if roles_core[i] in (ROLE_TO_ID["modifier"], ROLE_TO_ID["dual_role"])]
        conf_only = [i for i in range(n_cand) if roles_core[i] == ROLE_TO_ID["confounder"]]
        if len(ht_active) >= 2 and rng.random() < 0.45:
            n_mp = int(rng.integers(1, min(3, len(ht_active)) + 1))
            for _ in range(n_mp):
                a, b = rng.choice(ht_active, size=2, replace=False)
                ht_part += float(rng.uniform(-0.65, 0.65)) * _standardize_vec(X_std[:, a] * X_std[:, b])
        if len(ht_active) >= 1 and len(conf_only) >= 1 and rng.random() < 0.45:
            n_cm = int(rng.integers(1, min(3, len(conf_only) * len(ht_active)) + 1))
            for _ in range(n_cm):
                ci = int(rng.choice(conf_only))
                mi = int(rng.choice(ht_active))
                ht_part += float(rng.uniform(-0.70, 0.70)) * _standardize_vec(X_std[:, ci] * X_std[:, mi])
                roles_core[ci] = ROLE_TO_ID["dual_role"]

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

    cols_obs, is_bin, roles_core, n_frag_dups = _inject_fragmented_extractions(
        cols_obs, is_bin, roles_core, p, prior, rng
    )

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
            "n_frag_dups": n_frag_dups,
            "has_extraction_noise": has_ext_noise,
            "is_null_hte": is_null_hte,
            "mechanism": mech,
            "scale": scale_mode,
            "missingness": miss_mode,
        },
    }


# ==============================================================================
# FAMILY 4: CLINICAL DECISION-TREE / MULTI-WAY BIOMARKER SUBGROUP HTE DGP
# ==============================================================================
def _eval_rule_ensemble(
    Z_std: np.ndarray,
    pool_idx: np.ndarray,
    n_rules: int,
    rng: np.random.Generator,
    *,
    min_depth: int = 2,
    max_depth: int = 4,
) -> tuple[np.ndarray, set[int]]:
    """Fast vectorized ensemble of random multi-way clinical threshold/decision rules."""
    n = Z_std.shape[0]
    if len(pool_idx) == 0 or n_rules <= 0:
        return np.zeros(n, dtype=np.float32), set()
    out = np.zeros(n, dtype=np.float32)
    used: set[int] = set()
    hi_d = min(max_depth, len(pool_idx))
    lo_d = min(min_depth, hi_d)
    for _ in range(n_rules):
        depth = int(rng.integers(lo_d, hi_d + 1))
        chosen = rng.choice(pool_idx, size=depth, replace=False)
        sharp = rng.random() < 0.65
        gate = np.ones(n, dtype=np.float32)
        for c in chosen:
            used.add(int(c))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            # Choose realistic thresholds so conjunctions activate on ~10-45% of patients
            cut = float(rng.uniform(-0.85, 0.55)) if depth >= 3 else float(rng.uniform(-0.9, 0.9))
            diff = sign * (Z_std[:, c] - cut)
            if sharp:
                gate *= (diff > 0.0).astype(np.float32)
            else:
                slope = float(rng.uniform(2.5, 7.0))
                gate *= _expit(slope * diff).astype(np.float32)
        if float(gate.std()) > 1e-4:
            weight = float(rng.normal(0.0, 1.0))
            out += weight * _standardize_vec(gate)
    return out, used


def _sample_tree_subgroup(
    n_total: int, p: int, prior: PriorConfig, rng: np.random.Generator
) -> dict:
    """Clinical decision-tree & multi-way biomarker subgroup HTE prior."""
    is_null_hte = bool(rng.random() < prior.null_hte_prob)
    n_core = int(rng.integers(min(6, max(4, p // 3)), min(20, max(7, p - 2)) + 1))
    n_core = max(4, min(n_core, p - 2))

    Z_core = _generate_factor_latents(n_total, n_core, min(prior.factor_noise_rank, 6), rng)
    X_core, bin_core = _realize_column_types(Z_core, rng)
    X_std = (X_core - X_core.mean(axis=0, keepdims=True)) / np.maximum(
        X_core.std(axis=0, keepdims=True), 1e-5
    )

    perm_c = rng.permutation(n_core)
    n_conf = max(2, min(int(rng.integers(2, 8)), n_core - 2))
    rem_c = n_core - n_conf
    n_prog = min(int(rng.integers(1, 7)), max(0, rem_c - (1 if not is_null_hte else 0)))
    rem_c -= n_prog
    n_mod = 0 if is_null_hte else max(1, min(int(rng.integers(2, 6)), max(1, rem_c)))
    n_dual = 0 if is_null_hte else min(int(rng.integers(0, 3)), max(0, n_conf - 1))

    conf_pool = perm_c[:n_conf]
    dual_pool = conf_pool[:n_dual] if n_dual > 0 else np.empty(0, dtype=np.int64)
    prog_pool = perm_c[n_conf : n_conf + n_prog]
    mod_only_pool = perm_c[n_conf + n_prog : n_conf + n_prog + n_mod]

    # Treatment rules (confounders + dual_role): depth 1-3 clinical prescribing guidelines
    t_pool = conf_pool
    t_rules, used_t = _eval_rule_ensemble(
        X_std, t_pool, int(rng.integers(3, 8)), rng, min_depth=1, max_depth=3
    )
    t_lin = X_std[:, t_pool] @ rng.normal(0.0, 0.35, size=len(t_pool)).astype(np.float32)
    t_signal = _standardize_vec(t_rules) + 0.35 * _standardize_vec(t_lin)
    for c in t_pool:
        used_t.add(int(c))

    # Prognostic / baseline rules (confounders + prognostics + optional modifier main effects)
    y_pool_parts = [conf_pool, prog_pool]
    if len(mod_only_pool) and rng.random() < 0.45:
        y_pool_parts.append(mod_only_pool)
    y_pool = np.concatenate(y_pool_parts)
    y_rules, used_y = _eval_rule_ensemble(
        X_std, y_pool, int(rng.integers(4, 10)), rng, min_depth=1, max_depth=3
    )
    y_lin = X_std[:, y_pool] @ rng.normal(0.0, 0.30, size=len(y_pool)).astype(np.float32)
    y_base = _standardize_vec(y_rules) + 0.30 * _standardize_vec(y_lin)
    for c in y_pool:
        used_y.add(int(c))

    # Multi-way subgroup HTE rules (depth 2-4 biomarker/clinical conjunctions + confounder x modifier!)
    ht_part = np.zeros(n_total, dtype=np.float32)
    used_tau: set[int] = set()
    if not is_null_hte:
        tau_pool = np.unique(np.concatenate([mod_only_pool, dual_pool]))
        if len(tau_pool) == 0:
            tau_pool = perm_c[-min(3, n_core) :]
        ht_rules, used_tau = _eval_rule_ensemble(
            X_std, tau_pool, int(rng.integers(2, 6)), rng, min_depth=2, max_depth=4
        )
        # Explicit confounder x modifier cross-role subgroup conjunctions (~55% of studies)
        if len(conf_pool) > 0 and len(tau_pool) > 0 and rng.random() < 0.55:
            n_cm_rules = int(rng.integers(1, 4))
            for _ in range(n_cm_rules):
                c_idx = int(rng.choice(conf_pool))
                m_idx = int(rng.choice(tau_pool))
                if c_idx == m_idx:
                    continue
                s1 = 1.0 if rng.random() < 0.5 else -1.0
                s2 = 1.0 if rng.random() < 0.5 else -1.0
                cut1 = float(rng.uniform(-0.7, 0.7))
                cut2 = float(rng.uniform(-0.7, 0.7))
                if rng.random() < 0.65:
                    g_cm = ((s1 * (X_std[:, c_idx] - cut1) > 0.0) & (s2 * (X_std[:, m_idx] - cut2) > 0.0)).astype(
                        np.float32
                    )
                else:
                    g_cm = (_expit(4.0 * s1 * (X_std[:, c_idx] - cut1)) * X_std[:, m_idx]).astype(np.float32)
                if float(g_cm.std()) > 1e-4:
                    ht_rules = ht_rules + float(rng.normal(0.0, 0.9)) * _standardize_vec(g_cm)
                    used_tau.add(c_idx)
                    used_tau.add(m_idx)
        # Occasional pure subgroup rule (zero linear main term) vs mixed linear + subgroup
        if float(ht_rules.std()) < 1e-4 or rng.random() < 0.35:
            ht_lin = X_std[:, tau_pool] @ rng.normal(0.0, 0.5, size=len(tau_pool)).astype(np.float32)
            for c in tau_pool:
                used_tau.add(int(c))
            ht_part = _standardize_vec(ht_rules) + 0.45 * _standardize_vec(ht_lin)
        else:
            ht_part = ht_rules

    # Exact ground-truth causal role assignment from actual rule usage
    roles_core = np.zeros(n_core, dtype=np.int64)
    for j in range(n_core):
        in_t = j in used_t
        in_y = j in used_y
        in_tau = j in used_tau
        if in_tau and in_t:
            roles_core[j] = ROLE_TO_ID["dual_role"]
        elif in_tau:
            roles_core[j] = ROLE_TO_ID["modifier"]
        elif in_t and in_y:
            roles_core[j] = ROLE_TO_ID["confounder"]
        elif in_y:
            roles_core[j] = ROLE_TO_ID["prognostic"]
        elif in_t:
            roles_core[j] = ROLE_TO_ID["instrument"]

    n_unmeasured = 0
    if rng.random() < prior.unmeasured_conf_prob:
        n_unmeasured = int(rng.integers(1, 3))
        U = rng.standard_normal((n_total, n_unmeasured)).astype(np.float32)
        u_str = float(rng.uniform(0.25, 0.70))
        t_signal += u_str * _standardize_vec(U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32))
        y_base += u_str * _standardize_vec(U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32))

    has_ext_noise = rng.random() < prior.extraction_noise_prob
    if has_ext_noise:
        X_core = _inject_extraction_noise(X_core, bin_core, np.arange(n_core), rng)
    X_core, bin_core, roles_core, n_frag_dups = _inject_fragmented_extractions(
        X_core, bin_core, roles_core, p, prior, rng
    )

    X_full, _, roles_full, t_signal = _pad_with_noise_proxies_and_instruments(
        X_core, bin_core, roles_core, t_signal, p, prior, rng
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
            "family": "tree_subgroup",
            "p": int(X_obs.shape[1]),
            "n_active": int((roles_full >= 2).sum()),
            "n_mod": int((roles_full == ROLE_TO_ID["modifier"]).sum()),
            "n_dual": int((roles_full == ROLE_TO_ID["dual_role"]).sum()),
            "n_inst": int((roles_full == ROLE_TO_ID["instrument"]).sum()),
            "n_unmeasured": n_unmeasured,
            "n_frag_dups": n_frag_dups,
            "has_extraction_noise": has_ext_noise,
            "is_null_hte": is_null_hte,
            "mechanism": "tree_rules",
            "scale": scale_mode,
            "missingness": miss_mode,
        },
    }


# ==============================================================================
# FAMILY 5: GENERAL NON-LINEAR STRUCTURAL CAUSAL MODEL (DAG / MLP + M-BIAS)
# ==============================================================================
def _random_mlp_surface(
    Z_in: np.ndarray, rng: np.random.Generator, *, hidden: int = 16
) -> np.ndarray:
    """Fast 2-layer random nonlinear MLP surface with linear skip connection."""
    n, d = Z_in.shape
    if d == 0:
        return np.zeros(n, dtype=np.float32)
    W1 = rng.normal(0.0, 1.0 / math.sqrt(max(d, 1)), size=(d, hidden)).astype(np.float32)
    b1 = rng.uniform(-0.8, 0.8, size=(1, hidden)).astype(np.float32)
    pre = Z_in @ W1 + b1
    act = str(rng.choice(["tanh", "relu", "silu", "sin"]))
    if act == "tanh":
        h = np.tanh(pre)
    elif act == "relu":
        h = np.maximum(pre, 0.0)
    elif act == "silu":
        h = pre * _expit(pre)
    else:
        h = np.sin(np.clip(pre, -6.0, 6.0))
    w2 = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=hidden).astype(np.float32)
    nl = _standardize_vec(h @ w2)
    lin = _standardize_vec(Z_in @ rng.normal(0.0, 1.0, size=d).astype(np.float32))
    alpha = float(rng.uniform(0.35, 0.85))
    return _standardize_vec(alpha * nl + (1.0 - alpha) * lin)


def _sample_causal_scm(
    n_total: int, p: int, prior: PriorConfig, rng: np.random.Generator
) -> dict:
    """Layered nonlinear DAG / SCM prior with MLP mechanisms and M-bias colliders."""
    is_null_hte = bool(rng.random() < prior.null_hte_prob)
    is_dense = (p >= 90) and (rng.random() < 0.30)
    n_nodes = (
        int(rng.integers(18, min(36, p - 4) + 1))
        if is_dense
        else int(rng.integers(min(8, max(5, p // 3)), min(22, max(9, p - 4)) + 1))
    )
    n_nodes = max(5, min(n_nodes, p - 2))

    # Build a fast topologically ordered nonlinear DAG over n_nodes core variables
    Z_dag = np.empty((n_total, n_nodes), dtype=np.float32)
    n_roots = max(2, n_nodes // 3)
    for j in range(n_nodes):
        if j < n_roots:
            # Mix Gaussian, Student-t, and bimodal mixture roots
            root_type = float(rng.random())
            if root_type < 0.60:
                v = rng.standard_normal(n_total).astype(np.float32)
            elif root_type < 0.82:
                df = float(rng.uniform(3.0, 5.5))
                v = rng.standard_t(df, size=n_total).astype(np.float32)
                v = np.clip(v, -12.0, 12.0)
            else:
                comp = (rng.random(n_total) < float(rng.uniform(0.25, 0.75))).astype(np.float32)
                shift = float(rng.uniform(1.2, 2.4))
                v = (comp * shift - (1.0 - comp) * shift + rng.standard_normal(n_total)).astype(
                    np.float32
                )
            Z_dag[:, j] = _standardize_vec(v)
        else:
            k_par = int(rng.integers(1, min(4, j) + 1))
            pars = rng.choice(j, size=k_par, replace=False)
            w = rng.normal(0.0, 1.0 / math.sqrt(k_par), size=k_par).astype(np.float32)
            lin = Z_dag[:, pars] @ w
            edge_mode = str(rng.choice(["linear", "tanh", "relu", "quad", "prod"]))
            if edge_mode == "tanh":
                sig = np.tanh(1.3 * lin)
            elif edge_mode == "relu":
                sig = np.maximum(lin, 0.0)
            elif edge_mode == "quad":
                sig = 0.7 * lin + 0.35 * (lin * lin - 1.0)
            elif edge_mode == "prod" and k_par >= 2:
                sig = 0.65 * lin + 0.45 * (Z_dag[:, pars[0]] * Z_dag[:, pars[1]])
            else:
                sig = lin
            noise_scale = float(rng.uniform(0.35, 0.85))
            Z_dag[:, j] = _standardize_vec(
                _standardize_vec(sig) + noise_scale * rng.standard_normal(n_total).astype(np.float32)
            )

    X_core, bin_core = _realize_column_types(Z_dag, rng)
    X_std = (X_core - X_core.mean(axis=0, keepdims=True)) / np.maximum(
        X_core.std(axis=0, keepdims=True), 1e-5
    )

    # Assign causal roles across DAG nodes
    perm_n = rng.permutation(n_nodes)
    if is_dense:
        n_conf = max(6, min(int(rng.integers(8, 22)), n_nodes - 4))
    else:
        n_conf = max(2, min(int(rng.integers(2, 8)), n_nodes - 3))
    rem = n_nodes - n_conf
    n_inst_core = min(int(rng.integers(1, 4)), max(0, rem - 2)) if rng.random() < prior.instrument_prob else 0
    rem -= n_inst_core
    n_prog = max(1, min(int(rng.integers(1, 7)), max(1, rem - (1 if not is_null_hte else 0))))
    rem = max(0, rem - n_prog)
    n_mod = 0 if is_null_hte else min(max(1, int(rng.integers(1, 5))), max(1, rem))
    n_dual = 0 if is_null_hte else min(int(rng.integers(0, 3)), max(0, n_conf - 1))

    cursor = 0
    conf_idx = perm_n[cursor : cursor + n_conf]; cursor += n_conf
    inst_idx = perm_n[cursor : cursor + n_inst_core]; cursor += n_inst_core
    prog_idx = perm_n[cursor : cursor + n_prog]; cursor += n_prog
    mod_idx = perm_n[cursor : cursor + n_mod] if n_mod > 0 else np.empty(0, dtype=np.int64)
    dual_idx = conf_idx[:n_dual] if n_dual > 0 else np.empty(0, dtype=np.int64)

    roles_core = np.zeros(n_nodes, dtype=np.int64)
    roles_core[conf_idx] = ROLE_TO_ID["confounder"]
    roles_core[inst_idx] = ROLE_TO_ID["instrument"]
    roles_core[prog_idx] = ROLE_TO_ID["prognostic"]
    roles_core[mod_idx] = ROLE_TO_ID["modifier"]
    roles_core[dual_idx] = ROLE_TO_ID["dual_role"]

    t_parents = np.concatenate([conf_idx, inst_idx])
    y_parents = np.concatenate([conf_idx, prog_idx, mod_idx]) if (len(mod_idx) and rng.random() < 0.5) else np.concatenate([conf_idx, prog_idx])
    ht_parents = np.unique(np.concatenate([mod_idx, dual_idx])) if not is_null_hte else np.empty(0, dtype=np.int64)
    if not is_null_hte and len(ht_parents) == 0:
        dual_idx = conf_idx[:1]
        roles_core[dual_idx] = ROLE_TO_ID["dual_role"]
        ht_parents = dual_idx

    t_signal = _random_mlp_surface(X_std[:, t_parents], rng, hidden=18)
    y_base = _random_mlp_surface(X_std[:, y_parents], rng, hidden=20)
    if len(ht_parents) > 0:
        # In 45% of non-null SCM studies, confounders jointly modulate modifiers inside the HTE MLP
        if len(conf_idx) > 0 and rng.random() < 0.45:
            k_cm = int(rng.integers(1, min(3, len(conf_idx)) + 1))
            cm_sel = rng.choice(conf_idx, size=k_cm, replace=False)
            roles_core[cm_sel] = ROLE_TO_ID["dual_role"]
            ht_in = np.unique(np.concatenate([ht_parents, cm_sel]))
        else:
            ht_in = ht_parents
        ht_part = _random_mlp_surface(X_std[:, ht_in], rng, hidden=18)
    else:
        ht_part = np.zeros(n_total, dtype=np.float32)

    # Explicit M-bias colliders (both observed-parent Z -> C <- W and latent-parent U_T -> C <- U_Y!)
    n_colliders = 0
    if rng.random() < prior.collider_prob and (p - n_nodes) >= 3:
        n_coll = int(rng.integers(1, min(5, p - n_nodes) + 1))
        coll_cols, coll_bins, coll_roles = [], [], []
        # Latent independent U_T (causes T) and U_Y (causes Y), U_T _|_ U_Y
        u_t = rng.standard_normal(n_total).astype(np.float32)
        u_y = rng.standard_normal(n_total).astype(np.float32)
        used_latent_u = False
        for k_c in range(n_coll):
            if len(inst_idx) > 0 and len(prog_idx) > 0 and (k_c == 0 or rng.random() < 0.55):
                # 100% observationally identifiable collider: Z (instrument) -> C <- W (prognostic)
                z_p = X_std[:, int(rng.choice(inst_idx))]
                w_p = X_std[:, int(rng.choice(prog_idx))]
                c_val = (
                    float(rng.uniform(0.7, 1.3)) * z_p
                    + float(rng.uniform(0.7, 1.3)) * w_p
                    + float(rng.uniform(-0.3, 0.3)) * (z_p * w_p)
                    + float(rng.uniform(0.35, 0.75)) * rng.standard_normal(n_total).astype(np.float32)
                )
            else:
                # Classic Pearl M-bias collider: U_T -> C <- U_Y
                used_latent_u = True
                c_val = (
                    float(rng.uniform(0.8, 1.4)) * u_t
                    + float(rng.uniform(0.8, 1.4)) * u_y
                    + float(rng.uniform(0.30, 0.70)) * rng.standard_normal(n_total).astype(np.float32)
                )
            c_std = _standardize_vec(c_val)
            if rng.random() < 0.35:
                c_out = (c_std > float(rng.uniform(-0.6, 0.6))).astype(np.float32)
                coll_bins.append(True)
            else:
                c_out = c_std
                coll_bins.append(False)
            coll_cols.append(c_out)
            # Colliders must NOT be adjusted for (label noise=0 so nuisance screener suppresses them!)
            coll_roles.append(ROLE_TO_ID["noise"])
        if used_latent_u:
            t_signal = t_signal + float(rng.uniform(0.45, 0.90)) * u_t
            y_base = y_base + float(rng.uniform(0.45, 0.90)) * u_y
        X_core = np.concatenate([X_core, np.stack(coll_cols, axis=1)], axis=1).astype(np.float32)
        bin_core = np.concatenate([bin_core, np.asarray(coll_bins, dtype=bool)])
        roles_core = np.concatenate([roles_core, np.asarray(coll_roles, dtype=np.int64)])
        n_colliders = len(coll_cols)

    n_unmeasured = 0
    if rng.random() < prior.unmeasured_conf_prob:
        n_unmeasured = int(rng.integers(1, 3))
        U = rng.standard_normal((n_total, n_unmeasured)).astype(np.float32)
        u_str = float(rng.uniform(0.25, 0.70))
        t_signal += u_str * _standardize_vec(U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32))
        y_base += u_str * _standardize_vec(U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32))

    has_ext_noise = rng.random() < prior.extraction_noise_prob
    if has_ext_noise:
        X_core = _inject_extraction_noise(X_core, bin_core, np.arange(X_core.shape[1]), rng)
    X_core, bin_core, roles_core, n_frag_dups = _inject_fragmented_extractions(
        X_core, bin_core, roles_core, p, prior, rng
    )

    X_full, _, roles_full, t_signal = _pad_with_noise_proxies_and_instruments(
        X_core,
        bin_core,
        roles_core,
        t_signal,
        p,
        prior,
        rng,
        allow_extra_instruments=(n_inst_core == 0),
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
            "family": "causal_scm",
            "p": int(X_obs.shape[1]),
            "n_active": int((roles_full >= 2).sum()),
            "n_mod": int((roles_full == ROLE_TO_ID["modifier"]).sum()),
            "n_dual": int((roles_full == ROLE_TO_ID["dual_role"]).sum()),
            "n_inst": int((roles_full == ROLE_TO_ID["instrument"]).sum()),
            "n_colliders": n_colliders,
            "n_unmeasured": n_unmeasured,
            "n_frag_dups": n_frag_dups,
            "is_dense_conf": is_dense,
            "has_extraction_noise": has_ext_noise,
            "is_null_hte": is_null_hte,
            "mechanism": "scm_mlp",
            "scale": scale_mode,
            "missingness": miss_mode,
        },
    }


# ==============================================================================
# FAMILY 6: INTERACTING CLINICAL & COSINE-MAXSIM EMBEDDING HYBRID PRIOR
# ==============================================================================
def _orthogonalize_interaction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a*b projected orthogonal to [1, a, b] (pure crossover interaction)."""
    a_c = _standardize_vec(a)
    b_c = _standardize_vec(b)
    prod = a_c * b_c
    # Solve 3x3 OLS on [1, a_c, b_c]
    D = np.stack([np.ones_like(a_c), a_c, b_c], axis=1)
    gram = D.T @ D + 1e-4 * np.eye(3, dtype=np.float32)
    coef = np.linalg.solve(gram, D.T @ prod)
    resid = prod - D @ coef
    return _standardize_vec(resid)


def _sample_interacting_clinical(
    n_total: int, p: int, prior: PriorConfig, rng: np.random.Generator
) -> dict:
    """Interaction-heavy clinical DGP (confounder x confounder, confounder x modifier,
    pure-epistatic crossover HTE) + 50% correlated cosine-MaxSim embedding feature blocks."""
    is_null_hte = bool(rng.random() < prior.null_hte_prob)
    use_emb_block = (p >= 60) and (rng.random() < 0.50)

    max_struct = min(16 if use_emb_block else 22, max(4, p // 3))
    n_conf = int(rng.integers(2, min(8, max(3, max_struct // 2)) + 1))
    n_prog = int(rng.integers(1, min(6, max(2, max_struct // 3)) + 1))
    if is_null_hte:
        n_mod, n_dual = 0, 0
    else:
        n_mod = int(rng.integers(1, min(5, max(2, max_struct // 3)) + 1))
        n_dual = int(rng.integers(0, min(3, max(1, max_struct // 4)) + 1))
    while n_conf + n_prog + n_mod + n_dual > max_struct:
        if n_prog > 1:
            n_prog -= 1
        elif n_conf > 2:
            n_conf -= 1
        elif n_mod > 1:
            n_mod -= 1
        elif n_dual > 0:
            n_dual -= 1
        else:
            break
    n_struct_base = n_conf + n_prog + n_mod + n_dual

    Z_struct = _generate_factor_latents(n_total, n_struct_base, min(prior.factor_noise_rank, 6), rng)
    X_s_base, bin_s_base = _realize_column_types(Z_struct, rng)
    role_s_base = np.concatenate(
        [
            np.full(n_conf, ROLE_TO_ID["confounder"], dtype=np.int64),
            np.full(n_prog, ROLE_TO_ID["prognostic"], dtype=np.int64),
            np.full(n_mod, ROLE_TO_ID["modifier"], dtype=np.int64),
            np.full(n_dual, ROLE_TO_ID["dual_role"], dtype=np.int64),
        ]
    )

    cols_list, bin_list, role_list = [], [], []
    use_cats = rng.random() < max(prior.categorical_block_prob, 0.60)
    cat_budget = p - (36 if use_emb_block else 4)
    for j in range(n_struct_base):
        if use_cats and (len(cols_list) + 4 <= cat_budget) and rng.random() < 0.30:
            n_lvl = int(rng.integers(3, 5))
            dummies, _ = _make_categorical_dummy_block(Z_struct[:, j], n_lvl, rng)
            for d_i in range(dummies.shape[1]):
                cols_list.append(dummies[:, d_i])
                bin_list.append(True)
                role_list.append(int(role_s_base[j]))
        else:
            cols_list.append(X_s_base[:, j])
            bin_list.append(bool(bin_s_base[j]))
            role_list.append(int(role_s_base[j]))

    X_core = np.stack(cols_list, axis=1).astype(np.float32)
    bin_core = np.asarray(bin_list, dtype=bool)
    roles_core = np.asarray(role_list, dtype=np.int64)
    n_struct_cols = X_core.shape[1]

    # Optional correlated cosine-MaxSim embedding block (mimics TinyOncBERT SentKM geometry!)
    k_emb = 0
    if use_emb_block and (p - n_struct_cols) >= 32:
        max_k = min(320, p - n_struct_cols - 4)
        min_k = min(48, max(24, max_k // 2))
        k_emb = int(rng.integers(min_k, max(min_k + 1, max_k + 1)))
        d_lat = int(rng.integers(14, 28))
        m_sent = int(rng.integers(4, 8))

        # Shared clinical-note direction so cosine similarities lie realistically in ~[0.15, 0.95]
        u0 = rng.standard_normal(d_lat).astype(np.float32)
        u0 /= max(float(np.linalg.norm(u0)), 1e-6)

        # Patient latent clinical-note subspace partially correlated with structured severity
        H_pat = rng.standard_normal((n_total, d_lat)).astype(np.float32)
        n_link = min(6, Z_struct.shape[1])
        W_link = rng.normal(0.0, 0.45 / math.sqrt(max(n_link, 1)), size=(n_link, d_lat)).astype(np.float32)
        H_pat = H_pat + Z_struct[:, :n_link] @ W_link
        H_pat /= np.maximum(np.linalg.norm(H_pat, axis=1, keepdims=True), 1e-6)

        # Per-patient multi-sentence embeddings (N, m_sent, d_lat), unit normalized
        c_base = float(rng.uniform(0.55, 0.95))
        c_pat = float(rng.uniform(0.50, 0.90))
        c_sent = float(rng.uniform(0.45, 0.85))
        S_pat = (
            c_base * u0[None, None, :]
            + c_pat * H_pat[:, None, :]
            + c_sent * rng.standard_normal((n_total, m_sent, d_lat)).astype(np.float32)
        )
        S_pat /= np.maximum(np.linalg.norm(S_pat, axis=-1, keepdims=True), 1e-6)

        # K unit prototype directions with uniform synonymous clusters (zero role/geometry shortcut!)
        V_proto = (
            float(rng.uniform(0.35, 0.80)) * u0[None, :]
            + rng.standard_normal((k_emb, d_lat)).astype(np.float32)
        )
        V_proto /= np.maximum(np.linalg.norm(V_proto, axis=1, keepdims=True), 1e-6)
        n_syn = int(0.25 * k_emb)
        if n_syn >= 2 and k_emb >= 8:
            src_idx = rng.integers(0, k_emb - n_syn, size=n_syn)
            noise_v = rng.standard_normal((n_syn, d_lat)).astype(np.float32)
            V_syn = V_proto[src_idx] + float(rng.uniform(0.18, 0.45)) * noise_v
            V_syn /= np.maximum(np.linalg.norm(V_syn, axis=1, keepdims=True), 1e-6)
            V_proto[k_emb - n_syn :] = V_syn
            V_proto = V_proto[rng.permutation(k_emb)]

        # Fast BLAS SGEMM MaxSim: (N*m_sent, d_lat) @ (d_lat, k_emb) -> max over m_sent
        sims = (S_pat.reshape(n_total * m_sent, d_lat) @ V_proto.T).reshape(n_total, m_sent, k_emb)
        X_emb = sims.max(axis=1).astype(np.float32)

        # Pick active prototypes uniformly at random so active & noise prototypes are identically distributed
        n_emb_act = int(rng.integers(2, min(9, max(3, k_emb // 6)) + 1))
        act_emb = rng.choice(k_emb, size=n_emb_act, replace=False)
        roles_emb = np.zeros(k_emb, dtype=np.int64)
        role_choices = (
            [ROLE_TO_ID["confounder"], ROLE_TO_ID["prognostic"]]
            if is_null_hte
            else [
                ROLE_TO_ID["confounder"],
                ROLE_TO_ID["prognostic"],
                ROLE_TO_ID["modifier"],
                ROLE_TO_ID["dual_role"],
            ]
        )
        for ei in act_emb:
            roles_emb[ei] = int(rng.choice(role_choices))

        X_core = np.concatenate([X_core, X_emb], axis=1)
        bin_core = np.concatenate([bin_core, np.zeros(k_emb, dtype=bool)])
        roles_core = np.concatenate([roles_core, roles_emb])

    X_std = (X_core - X_core.mean(axis=0, keepdims=True)) / np.maximum(
        X_core.std(axis=0, keepdims=True), 1e-5
    )
    mech = str(rng.choice(["linear", "quadratic", "hinge", "saturating", "threshold", "band"]))

    def _phi_col(c: int) -> np.ndarray:
        if bin_core[c]:
            return X_std[:, c]
        m_c = mech if rng.random() < 0.65 else "linear"
        return _apply_nonlinear(X_std[:, c], m_c, rng)

    conf_idx = np.where(roles_core == ROLE_TO_ID["confounder"])[0]
    prog_idx = np.where(roles_core == ROLE_TO_ID["prognostic"])[0]
    mod_idx = np.where(roles_core == ROLE_TO_ID["modifier"])[0]
    dual_idx = np.where(roles_core == ROLE_TO_ID["dual_role"])[0]

    # 1. Propensity equation: confounders + dual_role + strong confounder x confounder interactions
    t_drivers = np.concatenate([conf_idx, dual_idx])
    t_signal = np.zeros(n_total, dtype=np.float32)
    if len(t_drivers):
        phi_t = np.stack([_phi_col(int(c)) for c in t_drivers], axis=1)
        t_signal += phi_t @ rng.normal(0.0, 0.85, size=len(t_drivers)).astype(np.float32)
        if len(t_drivers) >= 2:
            n_tp = int(rng.integers(2, min(6, len(t_drivers) * (len(t_drivers) - 1) // 2 + 1) + 1))
            for _ in range(n_tp):
                a, b = rng.choice(len(t_drivers), size=2, replace=False)
                if rng.random() < 0.35:
                    cut = float(rng.uniform(-0.6, 0.6))
                    term = ((phi_t[:, a] > cut).astype(np.float32)) * phi_t[:, b]
                else:
                    term = phi_t[:, a] * phi_t[:, b]
                t_signal += float(rng.normal(0.0, 0.70)) * _standardize_vec(term)

    # 2. Baseline outcome equation: confounders + prognostics + dual_role + pairwise interactions
    # Decide pure-epistatic M1 x M2 mode upfront so pure epistatic modifiers have zero main effect on Y
    ht_drivers_pre = np.concatenate([mod_idx, dual_idx])
    can_epi = (len(ht_drivers_pre) >= 2) or (len(conf_idx) >= 1) or (len(prog_idx) >= 1)
    is_pure_epistatic = (not is_null_hte) and len(ht_drivers_pre) > 0 and can_epi and (rng.random() < 0.38)
    pure_mm_only = is_pure_epistatic and (len(mod_idx) >= 2) and (rng.random() < 0.50)

    y_drivers = np.concatenate([conf_idx, prog_idx, dual_idx])
    if len(mod_idx) and (not pure_mm_only) and rng.random() < 0.45:
        y_drivers = np.concatenate([y_drivers, mod_idx])
    y_base = np.zeros(n_total, dtype=np.float32)
    if len(y_drivers):
        phi_y = np.stack([_phi_col(int(c)) for c in y_drivers], axis=1)
        y_base += phi_y @ rng.normal(0.0, 0.85, size=len(y_drivers)).astype(np.float32)
        if len(y_drivers) >= 2:
            n_yp = int(rng.integers(2, min(7, len(y_drivers) * (len(y_drivers) - 1) // 2 + 1) + 1))
            for _ in range(n_yp):
                a, b = rng.choice(len(y_drivers), size=2, replace=False)
                y_base += float(rng.normal(0.0, 0.65)) * _standardize_vec(phi_y[:, a] * phi_y[:, b])

    # 3. HTE equation: rich confounder x modifier, prognostic x modifier, & pure M1 x M2 epistasis
    ht_drivers = np.concatenate([mod_idx, dual_idx])
    ht_part = np.zeros(n_total, dtype=np.float32)
    n_hte_inter = 0
    if not is_null_hte and len(ht_drivers) > 0:
        phi_ht = {int(c): _phi_col(int(c)) for c in ht_drivers}
        phi_c = {int(c): _phi_col(int(c)) for c in conf_idx}
        phi_p = {int(c): _phi_col(int(c)) for c in prog_idx}

        if not is_pure_epistatic:
            for c in ht_drivers:
                ht_part += float(rng.normal(0.0, 0.65)) * phi_ht[int(c)]

        # Build 2-5 interaction terms; in pure_mm_only mode build strictly orthogonal M1 x M2 pairs
        n_target_inter = int(rng.integers(2, 6))
        orth_pure = is_pure_epistatic and (pure_mm_only or rng.random() < 0.65)
        for k_i in range(n_target_inter):
            if pure_mm_only:
                ma, mb = rng.choice(mod_idx, size=2, replace=False)
                va, vb = phi_ht[int(ma)], phi_ht[int(mb)]
            else:
                choices = []
                if len(conf_idx) >= 1:
                    choices.append("cm")
                    if k_i == 0:
                        choices = ["cm"]
                if len(ht_drivers) >= 2 and k_i > 0:
                    choices.append("mm")
                elif len(ht_drivers) >= 2 and not choices:
                    choices.append("mm")
                if len(prog_idx) >= 1 and (k_i > 0 or not choices):
                    choices.append("pm")
                if not choices:
                    break
                kind = str(rng.choice(choices))
                if kind == "cm":
                    ca = int(rng.choice(conf_idx))
                    mb = int(rng.choice(ht_drivers))
                    va, vb = phi_c[ca], phi_ht[mb]
                    roles_core[ca] = ROLE_TO_ID["dual_role"]
                elif kind == "pm":
                    pa = int(rng.choice(prog_idx))
                    mb = int(rng.choice(ht_drivers))
                    va, vb = phi_p[pa], phi_ht[mb]
                    roles_core[pa] = ROLE_TO_ID["modifier"]
                else:
                    ma, mb = rng.choice(ht_drivers, size=2, replace=False)
                    va, vb = phi_ht[int(ma)], phi_ht[int(mb)]

            if orth_pure:
                term = _orthogonalize_interaction(va, vb)
            elif (not is_pure_epistatic) and rng.random() < 0.30:
                cut = float(rng.uniform(-0.6, 0.6))
                term = _standardize_vec(((va > cut).astype(np.float32)) * vb)
            else:
                term = _standardize_vec(va * vb)
            if float(term.std()) > 1e-4:
                ht_part += float(rng.normal(0.0, 0.90)) * term
                n_hte_inter += 1

        if float(ht_part.std()) < 1e-5:
            ht_part += phi_ht[int(ht_drivers[0])]

    n_unmeasured = 0
    if rng.random() < prior.unmeasured_conf_prob:
        n_unmeasured = int(rng.integers(1, 3))
        U = rng.standard_normal((n_total, n_unmeasured)).astype(np.float32)
        u_str = float(rng.uniform(0.25, 0.70))
        t_signal += u_str * _standardize_vec(U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32))
        y_base += u_str * _standardize_vec(U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32))

    # Optional extraction noise on structured columns only (embedding columns are already noisy cosines)
    has_ext_noise = rng.random() < prior.extraction_noise_prob
    if has_ext_noise and n_struct_cols > 0:
        X_core = _inject_extraction_noise(X_core, bin_core, np.arange(n_struct_cols), rng)

    # Fragmented synonym extraction only when ample noise padding remains so noise clusters stay balanced
    n_frag_dups = 0
    if (p - X_core.shape[1]) >= 24:
        X_core, bin_core, roles_core, n_frag_dups = _inject_fragmented_extractions(
            X_core, bin_core, roles_core, p, prior, rng
        )

    X_full, _, roles_full, t_signal = _pad_with_noise_proxies_and_instruments(
        X_core, bin_core, roles_core, t_signal, p, prior, rng
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
            "family": "interacting_clinical",
            "p": int(X_obs.shape[1]),
            "n_active": int((roles_full >= 2).sum()),
            "n_mod": int((roles_full == ROLE_TO_ID["modifier"]).sum()),
            "n_dual": int((roles_full == ROLE_TO_ID["dual_role"]).sum()),
            "n_inst": int((roles_full == ROLE_TO_ID["instrument"]).sum()),
            "n_unmeasured": n_unmeasured,
            "n_frag_dups": n_frag_dups,
            "has_embedding_block": bool(k_emb > 0),
            "k_emb": int(k_emb),
            "is_pure_epistatic": bool(is_pure_epistatic),
            "n_hte_interactions": int(n_hte_inter),
            "has_extraction_noise": has_ext_noise,
            "is_null_hte": is_null_hte,
            "mechanism": mech,
            "scale": scale_mode,
            "missingness": miss_mode,
        },
    }


# ==============================================================================
# FAMILY 7: UNIVERSAL NONPARAMETRIC, HIGHER-ORDER EPISTASIS & DENSE PRIOR (v5)
# ==============================================================================
def _rbf_kernel_surface(X_sub: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Localized Gaussian & Laplace radial-basis kernel mixture over active feature subsets."""
    n, d = X_sub.shape
    if d == 0:
        return np.zeros(n, dtype=np.float32)
    n_bumps = int(rng.integers(8, 17))
    out = np.zeros(n, dtype=np.float32)
    for _ in range(n_bumps):
        k_sub = int(rng.integers(1, min(7, d) + 1))
        idx = rng.choice(d, size=k_sub, replace=False)
        Xs = X_sub[:, idx]
        center = Xs[int(rng.integers(0, n))] + rng.normal(0.0, 0.25, size=k_sub).astype(np.float32)
        scale = rng.uniform(0.65, 2.2, size=k_sub).astype(np.float32) * math.sqrt(k_sub)
        diff = (Xs - center[None, :]) / scale[None, :]
        if rng.random() < 0.5:
            bump = np.exp(-0.5 * np.sum(diff * diff, axis=1))
        else:
            bump = np.exp(-np.sum(np.abs(diff), axis=1))
        out += float(rng.normal(0.0, 1.0)) * bump.astype(np.float32)
    return _standardize_vec(out)


def _mars_spline_surface(X_sub: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """MARS-style multivariate adaptive spline hinge products (1-, 2-, and 3-way)."""
    n, d = X_sub.shape
    if d == 0:
        return np.zeros(n, dtype=np.float32)
    n_terms = int(rng.integers(6, 15))
    out = np.zeros(n, dtype=np.float32)
    for _ in range(n_terms):
        order = int(rng.integers(1, min(3, d) + 1))
        idx = rng.choice(d, size=order, replace=False)
        prod = np.ones(n, dtype=np.float32)
        for j in idx:
            knot = float(rng.uniform(-1.1, 1.1))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            prod *= np.maximum(0.0, sign * (X_sub[:, int(j)] - knot))
        if float(prod.std()) > 1e-5:
            out += float(rng.normal(0.0, 1.0)) * _standardize_vec(prod)
    return _standardize_vec(out)


def _deep_res_mlp_surface(X_sub: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """3-layer LayerNorm-Swish residual neural field (general smooth high-order surface)."""
    n, d = X_sub.shape
    if d == 0:
        return np.zeros(n, dtype=np.float32)
    h_dim = int(rng.integers(20, 36))
    W0 = rng.normal(0.0, 1.0 / math.sqrt(max(d, 1)), size=(d, h_dim)).astype(np.float32)
    b0 = rng.normal(0.0, 0.35, size=(1, h_dim)).astype(np.float32)
    h = X_sub @ W0 + b0
    h = h * _expit(1.2 * h)  # smooth Swish
    for _ in range(2):
        mu = h.mean(axis=1, keepdims=True)
        sd = np.maximum(h.std(axis=1, keepdims=True), 1e-4)
        hn = (h - mu) / sd
        Wl = rng.normal(0.0, 1.0 / math.sqrt(h_dim), size=(h_dim, h_dim)).astype(np.float32)
        bl = rng.normal(0.0, 0.25, size=(1, h_dim)).astype(np.float32)
        u = hn @ Wl + bl
        h = h + 0.65 * (u * _expit(u))
    w_out = rng.normal(0.0, 1.0 / math.sqrt(h_dim), size=h_dim).astype(np.float32)
    return _standardize_vec((h @ w_out).astype(np.float32))


def _higher_order_epistasis_surface(
    X_sub: np.ndarray, rng: np.random.Generator, *, pure: bool = True
) -> np.ndarray:
    """Orthogonalized 2-way (M1 x M2) and 3-way (M1 x M2 x M3) epistatic crossover surface."""
    n, d = X_sub.shape
    if d == 0:
        return np.zeros(n, dtype=np.float32)
    if d == 1:
        return _standardize_vec(X_sub[:, 0])
    out = np.zeros(n, dtype=np.float32)
    if not pure:
        out += 0.35 * _standardize_vec(
            (X_sub @ rng.normal(0.0, 1.0 / math.sqrt(d), size=d).astype(np.float32))
        )
    n_pairs = int(rng.integers(2, min(7, d * (d - 1) // 2 + 1) + 1))
    for _ in range(n_pairs):
        a, b = rng.choice(d, size=2, replace=False)
        term = _orthogonalize_interaction(X_sub[:, int(a)], X_sub[:, int(b)])
        if float(term.std()) > 1e-4:
            out += float(rng.normal(0.0, 1.0)) * term
    if d >= 3 and rng.random() < 0.65:
        n_tri = int(rng.integers(1, min(4, d) + 1))
        for _ in range(n_tri):
            a, b, c = rng.choice(d, size=3, replace=False)
            ab = _orthogonalize_interaction(X_sub[:, int(a)], X_sub[:, int(b)])
            tri = _orthogonalize_interaction(ab, X_sub[:, int(c)])
            if float(tri.std()) > 1e-4:
                out += float(rng.normal(0.0, 0.85)) * tri
    return _standardize_vec(out)


def _eval_universal_surface(
    X_sub: np.ndarray, mode: str, rng: np.random.Generator, *, pure_epi: bool = False
) -> np.ndarray:
    if X_sub.shape[1] == 0:
        return np.zeros(len(X_sub), dtype=np.float32)
    if mode == "rbf_kernel":
        return _rbf_kernel_surface(X_sub, rng)
    if mode == "mars_spline":
        return _mars_spline_surface(X_sub, rng)
    if mode == "deep_res_mlp":
        return _deep_res_mlp_surface(X_sub, rng)
    if mode == "higher_order_epistasis":
        return _higher_order_epistasis_surface(X_sub, rng, pure=pure_epi)
    # Composite mixture of two distinct universal function families
    s1 = _rbf_kernel_surface(X_sub, rng) if rng.random() < 0.5 else _mars_spline_surface(X_sub, rng)
    s2 = _deep_res_mlp_surface(X_sub, rng) if rng.random() < 0.5 else _higher_order_epistasis_surface(X_sub, rng, pure=False)
    w = float(rng.uniform(0.3, 0.7))
    return _standardize_vec(w * s1 + (1.0 - w) * s2)


def _sample_universal_nonparametric(
    n_total: int, p: int, prior: PriorConfig, rng: np.random.Generator
) -> dict:
    """Universal nonparametric, higher-order epistatic, and dense multi-biomarker causal prior.

    Mixes localized RBF/Laplace kernel mixtures, 1-to-3-way MARS hinge splines, 3-layer residual
    neural fields, and pure 2-/3-way orthogonal epistasis across both sparse and dense (up to 45
    confounders + 35 modifiers) regimes, strictly distinct from benchmark Bochner-RFF equations.
    """
    is_null_hte = bool(rng.random() < prior.null_hte_prob)
    is_dense = (p >= 100) and (rng.random() < max(prior.dense_conf_prob, 0.35))
    surface_kind = str(
        rng.choice(
            ["rbf_kernel", "mars_spline", "deep_res_mlp", "higher_order_epistasis", "composite_mix"]
        )
    )
    is_pure_epi = (
        (not is_null_hte)
        and (surface_kind == "higher_order_epistasis" or (not is_dense and rng.random() < 0.25))
    )

    if is_dense:
        n_conf = int(rng.integers(15, min(42, max(16, p // 3)) + 1))
        n_prog = int(rng.integers(4, min(14, max(5, p // 6)) + 1))
        n_mod = 0 if is_null_hte else int(rng.integers(10, min(32, max(11, p // 4)) + 1))
        n_dual = 0 if is_null_hte else int(rng.integers(2, min(10, max(3, p // 10)) + 1))
    else:
        n_conf = int(rng.integers(2, min(9, max(3, p // 4)) + 1))
        n_prog = int(rng.integers(1, min(6, max(2, p // 5)) + 1))
        n_mod = 0 if is_null_hte else int(rng.integers(2 if is_pure_epi else 1, min(7, max(3, p // 5)) + 1))
        n_dual = 0 if is_null_hte else int(rng.integers(0, min(4, max(1, p // 8)) + 1))

    max_act = min(75 if is_dense else 24, max(4, p - 4))
    while n_conf + n_prog + n_mod + n_dual > max_act:
        if n_prog > 1:
            n_prog -= 1
        elif n_conf > 2:
            n_conf -= 1
        elif n_mod > (2 if is_pure_epi else 1):
            n_mod -= 1
        elif n_dual > 0:
            n_dual -= 1
        else:
            break
    n_base = n_conf + n_prog + n_mod + n_dual

    Z_base = _generate_factor_latents(n_total, n_base, min(prior.factor_noise_rank, 10), rng)
    X_base, bin_base = _realize_column_types(Z_base, rng)
    role_base = np.concatenate(
        [
            np.full(n_conf, ROLE_TO_ID["confounder"], dtype=np.int64),
            np.full(n_prog, ROLE_TO_ID["prognostic"], dtype=np.int64),
            np.full(n_mod, ROLE_TO_ID["modifier"], dtype=np.int64),
            np.full(n_dual, ROLE_TO_ID["dual_role"], dtype=np.int64),
        ]
    )

    cols_list, bin_list, role_list = [], [], []
    use_cats = rng.random() < prior.categorical_block_prob
    for j in range(n_base):
        if use_cats and (len(cols_list) + 4 <= p - 4) and rng.random() < (0.15 if is_dense else 0.25):
            n_lvl = int(rng.integers(3, 5))
            dummies, _ = _make_categorical_dummy_block(Z_base[:, j], n_lvl, rng)
            for d_i in range(dummies.shape[1]):
                cols_list.append(dummies[:, d_i])
                bin_list.append(True)
                role_list.append(int(role_base[j]))
        else:
            cols_list.append(X_base[:, j])
            bin_list.append(bool(bin_base[j]))
            role_list.append(int(role_base[j]))

    X_core = np.stack(cols_list, axis=1).astype(np.float32)
    bin_core = np.asarray(bin_list, dtype=bool)
    roles_core = np.asarray(role_list, dtype=np.int64)
    X_std = (X_core - X_core.mean(axis=0, keepdims=True)) / np.maximum(
        X_core.std(axis=0, keepdims=True), 1e-5
    )

    conf_idx = np.where(roles_core == ROLE_TO_ID["confounder"])[0]
    prog_idx = np.where(roles_core == ROLE_TO_ID["prognostic"])[0]
    mod_idx = np.where(roles_core == ROLE_TO_ID["modifier"])[0]
    dual_idx = np.where(roles_core == ROLE_TO_ID["dual_role"])[0]

    t_drivers = np.concatenate([conf_idx, dual_idx])
    y_drivers = np.concatenate([conf_idx, prog_idx, dual_idx])
    if len(mod_idx) and (not is_pure_epi) and rng.random() < 0.45:
        y_drivers = np.concatenate([y_drivers, mod_idx])
    ht_drivers = mod_idx if (is_pure_epi and len(mod_idx) >= 2 and rng.random() < 0.65) else np.concatenate([mod_idx, dual_idx])

    t_mode = "composite_mix" if surface_kind == "higher_order_epistasis" else surface_kind
    y_mode = "composite_mix" if surface_kind == "higher_order_epistasis" else surface_kind
    t_lin = (
        _standardize_vec(
            X_std[:, t_drivers] @ rng.normal(0.0, 1.0 / math.sqrt(max(len(t_drivers), 1)), size=len(t_drivers)).astype(np.float32)
        )
        if len(t_drivers)
        else np.zeros(n_total, dtype=np.float32)
    )
    t_nl = _eval_universal_surface(X_std[:, t_drivers], t_mode, rng, pure_epi=False)
    wt_nl = float(rng.uniform(0.35, 0.85))
    t_signal = _standardize_vec((1.0 - wt_nl) * t_lin + wt_nl * t_nl) if len(t_drivers) else np.zeros(n_total, dtype=np.float32)

    y_lin = (
        _standardize_vec(
            X_std[:, y_drivers] @ rng.normal(0.0, 1.0 / math.sqrt(max(len(y_drivers), 1)), size=len(y_drivers)).astype(np.float32)
        )
        if len(y_drivers)
        else np.zeros(n_total, dtype=np.float32)
    )
    y_nl = _eval_universal_surface(X_std[:, y_drivers], y_mode, rng, pure_epi=False)
    wy_nl = float(rng.uniform(0.35, 0.85))
    y_base = _standardize_vec((1.0 - wy_nl) * y_lin + wy_nl * y_nl) if len(y_drivers) else np.zeros(n_total, dtype=np.float32)

    if not is_null_hte and len(ht_drivers) > 0:
        if is_pure_epi:
            ht_part = _higher_order_epistasis_surface(X_std[:, ht_drivers], rng, pure=True)
        else:
            ht_lin = _standardize_vec(
                X_std[:, ht_drivers] @ rng.normal(0.0, 1.0 / math.sqrt(len(ht_drivers)), size=len(ht_drivers)).astype(np.float32)
            )
            ht_nl = _eval_universal_surface(X_std[:, ht_drivers], surface_kind, rng, pure_epi=False)
            wh_nl = float(rng.uniform(0.45, 0.90))
            ht_part = _standardize_vec((1.0 - wh_nl) * ht_lin + wh_nl * ht_nl)
    else:
        ht_part = np.zeros(n_total, dtype=np.float32)

    # Optional observed & latent M-bias colliders
    n_colliders = 0
    if rng.random() < prior.collider_prob and (p - X_core.shape[1]) >= 4:
        n_coll = int(rng.integers(1, min(4, p - X_core.shape[1] - 2) + 1))
        u_t = rng.standard_normal(n_total).astype(np.float32)
        u_y = rng.standard_normal(n_total).astype(np.float32)
        coll_cols = []
        for _ in range(n_coll):
            c_val = (
                float(rng.uniform(0.8, 1.3)) * u_t
                + float(rng.uniform(0.8, 1.3)) * u_y
                + float(rng.uniform(0.35, 0.75)) * rng.standard_normal(n_total).astype(np.float32)
            )
            coll_cols.append(_standardize_vec(c_val))
        t_signal = t_signal + float(rng.uniform(0.40, 0.80)) * u_t
        y_base = y_base + float(rng.uniform(0.40, 0.80)) * u_y
        X_core = np.concatenate([X_core, np.stack(coll_cols, axis=1)], axis=1).astype(np.float32)
        bin_core = np.concatenate([bin_core, np.zeros(n_coll, dtype=bool)])
        roles_core = np.concatenate([roles_core, np.zeros(n_coll, dtype=np.int64)])
        n_colliders = n_coll

    n_unmeasured = 0
    if rng.random() < prior.unmeasured_conf_prob:
        n_unmeasured = int(rng.integers(1, 3))
        U = rng.standard_normal((n_total, n_unmeasured)).astype(np.float32)
        u_str = float(rng.uniform(0.25, 0.70))
        t_signal += u_str * _standardize_vec(U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32))
        y_base += u_str * _standardize_vec(U @ rng.normal(0.0, 1.0, size=n_unmeasured).astype(np.float32))

    has_ext_noise = rng.random() < prior.extraction_noise_prob
    if has_ext_noise:
        X_core = _inject_extraction_noise(X_core, bin_core, np.arange(X_core.shape[1]), rng)

    n_frag_dups = 0
    if (p - X_core.shape[1]) >= 24:
        X_core, bin_core, roles_core, n_frag_dups = _inject_fragmented_extractions(
            X_core, bin_core, roles_core, p, prior, rng
        )

    X_full, _, roles_full, t_signal = _pad_with_noise_proxies_and_instruments(
        X_core, bin_core, roles_core, t_signal, p, prior, rng
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
            "family": "universal_nonparametric",
            "p": int(X_obs.shape[1]),
            "n_active": int((roles_full >= 2).sum()),
            "n_mod": int((roles_full == ROLE_TO_ID["modifier"]).sum()),
            "n_dual": int((roles_full == ROLE_TO_ID["dual_role"]).sum()),
            "n_inst": int((roles_full == ROLE_TO_ID["instrument"]).sum()),
            "n_colliders": n_colliders,
            "n_unmeasured": n_unmeasured,
            "n_frag_dups": n_frag_dups,
            "is_dense": is_dense,
            "is_pure_epistatic": is_pure_epi,
            "surface_kind": surface_kind,
            "has_extraction_noise": has_ext_noise,
            "is_null_hte": is_null_hte,
            "mechanism": surface_kind,
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
    """Sample one synthetic high-dimensional sparse tabular causal study from the 7-family prior.

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
            w = np.zeros(len(DGP_FAMILIES), dtype=np.float64)
            w[0] = 1.0
        w /= w.sum()
        chosen = str(rng.choice(DGP_FAMILIES, p=w))

    if chosen == "reequation_calc":
        return _sample_reequation_calc(n_total, p, prior, rng)
    if chosen == "nemotron_design":
        return _sample_nemotron_design(n_total, p, prior, rng, designs)
    if chosen == "tree_subgroup":
        return _sample_tree_subgroup(n_total, p, prior, rng)
    if chosen == "causal_scm":
        return _sample_causal_scm(n_total, p, prior, rng)
    if chosen == "interacting_clinical":
        return _sample_interacting_clinical(n_total, p, prior, rng)
    if chosen == "universal_nonparametric":
        return _sample_universal_nonparametric(n_total, p, prior, rng)
    return _sample_sparse_oci(n_total, p, prior, rng)
