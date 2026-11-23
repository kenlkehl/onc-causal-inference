"""Vectorized high-dimensional sparse tabular causal prior for all-evidence tables.

Generates synthetic observational studies with N up to 10,000 and P up to 2,000+
candidate features capturing the real pathologies of Stage-2 extracted clinical tables:
  - Extreme sparsity (50% to 99.5% noise columns; 2 to 20 active causal features).
  - Explicit variable roles: noise, instrument (strong T, zero Y), prognostic (Y only),
    confounder (T and Y), effect_modifier (Y x T interaction), and dual_role (both).
  - Collinear noisy proxies/duplicates of active variables (e.g., multiple extracted
    phrasings of the same clinical concept) plus factor-correlated background noise.
  - Mixed data types: rare & common binary findings, ordinal scores (ECOG-like),
    Gaussian & heavy-tailed log-normal labs, and informative 3-state EHR missingness.
  - Nonlinearities (threshold/step, quadratic, saturating, pairwise interactions) and a
    configurable fraction of strictly null-HTE (constant treatment effect) studies.
"""

from dataclasses import dataclass
import math
import numpy as np

ROLE_NAMES = (
    "noise",
    "instrument",
    "prognostic",
    "confounder",
    "modifier",
    "dual_role",
)
ROLE_TO_ID = {name: idx for idx, name in enumerate(ROLE_NAMES)}


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


def _expit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(linpred: np.ndarray, target_rate: float) -> float:
    """Solve mean(expit(linpred + b)) == target_rate via fast safeguarded Newton/bisection."""
    lo, hi = -12.0, 12.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if float(_expit(linpred + mid).mean()) < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _apply_nonlinear(z: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Apply a realistic clinical transform and re-standardize to unit scale."""
    if mode == "linear":
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
    sd = float(out.std())
    if sd > 1e-6:
        out = (out - float(out.mean())) / sd
    return out.astype(np.float32)


def _realize_column_types(
    Z: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Convert latent Gaussian columns into realistic mixed clinical types.

    Returns ``(X_raw, is_binary)`` where ``X_raw`` is float32 ``(N, P)``.
    """
    n, p = Z.shape
    X = Z.astype(np.float32, copy=True)
    is_binary = np.zeros(p, dtype=bool)

    # Assign column types vectorized: 45% binary, 15% ordinal (0..4), 25% Gaussian, 15% log-normal
    u = rng.random(p)
    bin_cols = np.where(u < 0.45)[0]
    ord_cols = np.where((u >= 0.45) & (u < 0.60))[0]
    log_cols = np.where(u >= 0.85)[0]

    if len(bin_cols):
        # Mix of rare (3-15%) and common (15-55%) clinical findings
        prev = np.exp(rng.uniform(math.log(0.03), math.log(0.55), size=len(bin_cols)))
        # Exact normal quantile via inverse error function approximation (fast vectorized)
        # Use empirical quantile on Z for exact prevalence matching!
        for j_pos, col in enumerate(bin_cols):
            q = float(np.quantile(Z[:, col], 1.0 - prev[j_pos]))
            X[:, col] = (Z[:, col] >= q).astype(np.float32)
        is_binary[bin_cols] = True

    if len(ord_cols):
        # 4- or 5-level clinical scores (e.g. ECOG 0-4, stage I-IV, grade 1-3)
        for col in ord_cols:
            levels = int(rng.integers(3, 6))
            cuts = np.sort(rng.uniform(-1.2, 1.5, size=levels - 1))
            X[:, col] = np.digitize(Z[:, col], cuts).astype(np.float32)

    if len(log_cols):
        # Skewed lab-like biomarkers (e.g., LDH, PSA, bilirubin, CRP)
        scales = rng.uniform(0.3, 0.8, size=len(log_cols)).astype(np.float32)
        X[:, log_cols] = np.exp(np.clip(Z[:, log_cols] * scales[None, :], -4.0, 4.0))

    return X, is_binary


def sample_tabular_study(
    seed: int,
    n_total: int,
    *,
    num_features: int | None = None,
    prior: PriorConfig = PriorConfig(),
) -> dict:
    """Sample one complete synthetic high-dimensional sparse tabular causal study.

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
      - ``meta``: dict of study-level metadata (``p, n_active, is_null_hte, n_instruments, ...``)
    """
    rng = np.random.default_rng(seed)

    if num_features is None:
        log_p = rng.uniform(math.log(prior.min_features), math.log(prior.max_features))
        p = int(round(math.exp(log_p)))
    else:
        p = int(num_features)
    p = max(p, 6)

    # 1. Decide sparse active causal roles (in latent canonical order before random column shuffle)
    is_null_hte = bool(rng.random() < prior.null_hte_prob)
    n_conf = int(rng.integers(1, min(8, max(2, p // 4)) + 1))
    n_prog = int(rng.integers(0, min(7, max(1, p // 5)) + 1))
    if is_null_hte:
        n_mod = 0
        n_dual = 0
    else:
        n_mod = int(rng.integers(0, min(5, max(1, p // 6)) + 1))
        n_dual = int(rng.integers(0, min(3, max(1, p // 8)) + 1))
        if n_mod + n_dual == 0:
            if rng.random() < 0.6:
                n_mod = 1
            else:
                n_dual = 1

    # Cap total active features to keep realistic sparsity
    n_active = n_conf + n_prog + n_mod + n_dual
    max_act = min(prior.max_active, p - 2)
    while n_active > max_act:
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

    remaining = p - n_active
    n_inst = 0
    if remaining >= 2 and rng.random() < prior.instrument_prob:
        n_inst = int(rng.integers(1, min(prior.max_instruments, remaining // 2) + 1))
    remaining -= n_inst

    # Collinear noisy proxies of active / instrument features (labeled as noise or proxy -> noise
    # role for strict sparsity, OR kept distinct from direct causal parents!)
    n_proxies = 0
    if remaining >= 2 and n_active + n_inst > 0:
        max_px = min(prior.max_proxies, remaining // 2)
        if max_px >= 1:
            n_proxies = int(rng.integers(0, max_px + 1))
    n_noise = p - (n_active + n_inst + n_proxies)

    # 2. Generate latent correlated features Z (N, P) via fast low-rank factor model + idiosyncratic noise
    rank = min(prior.factor_noise_rank, max(2, p // 4))
    factors = rng.standard_normal((n_total, rank)).astype(np.float32)
    loadings = rng.standard_normal((rank, p)).astype(np.float32) / math.sqrt(rank)
    factor_strength = float(rng.uniform(0.15, 0.65))
    # Sparse factor membership so columns form realistic correlated blocks (labs, comorbidities, etc.)
    loadings *= (rng.random((rank, p)) < float(rng.uniform(0.25, 0.7))).astype(np.float32)
    Z = factor_strength * (factors @ loadings) + math.sqrt(
        max(1.0 - factor_strength * factor_strength, 0.1)
    ) * rng.standard_normal((n_total, p)).astype(np.float32)
    # Standardize latent columns
    Z -= Z.mean(axis=0, keepdims=True)
    Z /= np.maximum(Z.std(axis=0, keepdims=True), 1e-5)

    # Index slices in canonical order before permutation
    idx = 0
    conf_idx = np.arange(idx, idx + n_conf); idx += n_conf
    prog_idx = np.arange(idx, idx + n_prog); idx += n_prog
    mod_idx = np.arange(idx, idx + n_mod); idx += n_mod
    dual_idx = np.arange(idx, idx + n_dual); idx += n_dual
    inst_idx = np.arange(idx, idx + n_inst); idx += n_inst
    proxy_idx = np.arange(idx, idx + n_proxies); idx += n_proxies
    noise_idx = np.arange(idx, idx + n_noise)

    # Make proxies strongly correlated noisy copies of active or instrument columns
    source_pool = np.concatenate([conf_idx, prog_idx, mod_idx, dual_idx, inst_idx])
    if len(proxy_idx) and len(source_pool):
        parents = rng.choice(source_pool, size=len(proxy_idx), replace=True)
        rhos = rng.uniform(0.55, 0.94, size=len(proxy_idx)).astype(np.float32)
        eps = rng.standard_normal((n_total, len(proxy_idx))).astype(np.float32)
        Z[:, proxy_idx] = (
            rhos[None, :] * Z[:, parents] + np.sqrt(1.0 - rhos * rhos)[None, :] * eps
        )

    # 3. Realize realistic mixed clinical column types (binary, ordinal, Gaussian, log-normal)
    X_full, is_binary = _realize_column_types(Z, rng)

    # Standardized realized values used as the true structural inputs to T and Y
    X_std = X_full - X_full.mean(axis=0, keepdims=True)
    X_std /= np.maximum(X_std.std(axis=0, keepdims=True), 1e-5)

    # 4. Choose nonlinear mechanism for this study
    mech = str(rng.choice(["linear", "linear", "threshold", "quadratic", "hinge", "saturating"]))

    def _transformed(col_indices: np.ndarray) -> np.ndarray:
        if len(col_indices) == 0:
            return np.empty((n_total, 0), dtype=np.float32)
        cols = []
        for c in col_indices:
            # Binary columns stay linear; continuous/ordinal columns get study nonlinearity
            m = "linear" if is_binary[c] else (mech if rng.random() < 0.7 else "linear")
            cols.append(_apply_nonlinear(X_std[:, c], m, rng))
        return np.stack(cols, axis=1)

    # 5. Build treatment propensity logit (confounders + dual_role + instruments!)
    t_drivers = np.concatenate([conf_idx, dual_idx])
    t_signal = np.zeros(n_total, dtype=np.float32)
    if len(t_drivers):
        phi_t = _transformed(t_drivers)
        w_t = rng.normal(0.0, 1.0, size=len(t_drivers)).astype(np.float32)
        t_signal += phi_t @ w_t
    if len(inst_idx):
        # Instruments can be quite strong drivers of T (realistic Stage-1 clinic/formulary cues!)
        phi_inst = _transformed(inst_idx)
        inst_scale = float(rng.uniform(0.6, 1.8))
        w_inst = rng.normal(0.0, inst_scale, size=len(inst_idx)).astype(np.float32)
        t_signal += phi_inst @ w_inst

    # Calibrate propensity signal strength (controls confounding + overlap hardness)
    t_sd = float(t_signal.std())
    if t_sd > 1e-6:
        target_t_sd = float(rng.uniform(0.6, 2.0))
        t_signal = t_signal * (target_t_sd / t_sd)
    target_t_rate = float(rng.uniform(0.20, 0.80))
    b_t = _calibrate_intercept(t_signal, target_t_rate)
    true_prop = _expit(t_signal + b_t).astype(np.float32)

    # 6. Build baseline prognostic outcome logit (confounders + prognostics + dual_role + optional modifier mains)
    y_drivers = np.concatenate([conf_idx, prog_idx, dual_idx])
    if len(mod_idx) and rng.random() < 0.75:
        y_drivers = np.concatenate([y_drivers, mod_idx])
    y_base = np.zeros(n_total, dtype=np.float32)
    if len(y_drivers):
        phi_y = _transformed(y_drivers)
        w_y = rng.normal(0.0, 1.0, size=len(y_drivers)).astype(np.float32)
        y_base += phi_y @ w_y
        # Optional pairwise prognostic interaction (35% of studies)
        if len(y_drivers) >= 2 and rng.random() < 0.35:
            a, b = rng.choice(len(y_drivers), size=2, replace=False)
            inter = phi_y[:, a] * phi_y[:, b]
            inter_sd = float(inter.std())
            if inter_sd > 1e-6:
                y_base += float(rng.normal(0.0, 0.45)) * ((inter - float(inter.mean())) / inter_sd)

    y_sd = float(y_base.std())
    if y_sd > 1e-6:
        target_y_sd = float(rng.uniform(0.7, 2.2))
        y_base = y_base * (target_y_sd / y_sd)

    # 7. Build treatment effect logit tau_logit(x) (constant ATE + sparse modifiers/dual_role)
    ate_logit = float(rng.uniform(-1.2, 1.2))
    tau_logit = np.full(n_total, ate_logit, dtype=np.float32)
    ht_drivers = np.concatenate([mod_idx, dual_idx])
    if len(ht_drivers):
        phi_ht = _transformed(ht_drivers)
        w_ht = rng.normal(0.0, 1.0, size=len(ht_drivers)).astype(np.float32)
        ht_part = phi_ht @ w_ht
        # Optional pairwise modifier interaction (25% of non-null studies)
        if len(ht_drivers) >= 2 and rng.random() < 0.25:
            a, b = rng.choice(len(ht_drivers), size=2, replace=False)
            inter = phi_ht[:, a] * phi_ht[:, b]
            inter_sd = float(inter.std())
            if inter_sd > 1e-6:
                ht_part += float(rng.normal(0.0, 0.5)) * ((inter - float(inter.mean())) / inter_sd)
        ht_sd = float(ht_part.std())
        if ht_sd > 1e-6:
            target_ht_sd = float(rng.uniform(0.35, 1.6))
            tau_logit = tau_logit + ht_part * (target_ht_sd / ht_sd)

    target_y_rate = float(rng.uniform(0.15, 0.85))
    b_y = _calibrate_intercept(y_base + true_prop * tau_logit, target_y_rate)
    true_y0 = _expit(y_base + b_y).astype(np.float32)
    true_y1 = _expit(y_base + b_y + tau_logit).astype(np.float32)
    true_ite = (true_y1 - true_y0).astype(np.float32)

    # Sample observed treatment & factual outcome
    t_obs = (rng.random(n_total) < true_prop).astype(np.float32)
    p_obs = np.where(t_obs > 0.5, true_y1, true_y0)
    y_obs = (rng.random(n_total) < p_obs).astype(np.float32)

    # 8. Inject realistic EHR missingness (MCAR + severity-coupled MAR)
    X_obs = X_full.copy()
    M = np.zeros((n_total, p), dtype=np.float32)
    if rng.random() < prior.missing_study_prob:
        miss_rates = rng.uniform(0.0, prior.max_missing_rate, size=p)
        # Keep ~35% of columns completely observed
        miss_rates *= (rng.random(p) < 0.65)
        mar_driver = (y_base - float(y_base.mean())) / max(float(y_base.std()), 1e-5)
        mar_strength = float(rng.uniform(0.0, 0.6))
        for col in np.where(miss_rates > 0.01)[0]:
            b_m = math.log(miss_rates[col] / max(1.0 - miss_rates[col], 1e-4))
            prob_m = _expit(b_m + mar_strength * float(rng.choice([-1.0, 1.0])) * mar_driver)
            mask_c = rng.random(n_total) < prob_m
            # Never make an entire column 100% missing
            if mask_c.sum() < n_total - 2:
                M[mask_c, col] = 1.0
                X_obs[mask_c, col] = np.nan

    # 9. Assign ground-truth roles and randomly permute column order (no positional shortcuts!)
    roles = np.zeros(p, dtype=np.int64)
    roles[inst_idx] = ROLE_TO_ID["instrument"]
    roles[prog_idx] = ROLE_TO_ID["prognostic"]
    roles[conf_idx] = ROLE_TO_ID["confounder"]
    roles[mod_idx] = ROLE_TO_ID["modifier"]
    roles[dual_idx] = ROLE_TO_ID["dual_role"]
    # proxies and background noise remain role "noise" (0)

    col_perm = rng.permutation(p)
    X_obs = X_obs[:, col_perm]
    M = M[:, col_perm]
    roles = roles[col_perm]

    return {
        "X": X_obs,
        "M": M,
        "t": t_obs,
        "y": y_obs,
        "true_propensity": true_prop,
        "true_y0": true_y0,
        "true_y1": true_y1,
        "true_ite": true_ite,
        "roles": roles,
        "meta": {
            "p": p,
            "n_conf": n_conf,
            "n_prog": n_prog,
            "n_mod": n_mod,
            "n_dual": n_dual,
            "n_inst": n_inst,
            "n_proxies": n_proxies,
            "n_noise": n_noise,
            "is_null_hte": is_null_hte,
            "mechanism": mech,
        },
    }
