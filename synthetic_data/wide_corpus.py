# synthetic_data/wide_corpus.py
"""Fast deterministic wide-feature synthetic clinical corpus generator (10 to 150+ features).

Produces a standard generator output directory (``dataset.parquet``,
``metadata.json``, ``generation_config.json``) with block-correlated Gaussian-copula
covariates and structured pretreatment clinical text, with no LLM calls required.

The resulting directory satisfies the full contract of
:func:`synthetic_data.reequation.reequation_cohort` and
``scripts/benchmark.py materialize --generated`` (``scenario: null``,
``post_treatment: null``, ``true_<variable>`` for every metadata feature), making
it straightforward to stress-test complex DGPs with 5 to 50 confounders and 5 to
50 effect modifiers immediately.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import ndtr

from . import generator as g
from .config import DEFAULT_CLINICAL_QUESTION, StructuredDataConfig, SyntheticDataConfig
from .dgp_terms import CONFOUNDER, EFFECT_MODIFIER, role_lists
from .reequation import (
    _features_with_roles,
    _linear_predictors,
    _sigmoid,
    draw_equations,
    resolve_scenario,
)

LOGGER = logging.getLogger(__name__)


def _cont(name: str, desc: str, mean: float, std: float, unit: str = "", positive: bool = True) -> Dict[str, Any]:
    return {
        "name": name,
        "type": "continuous",
        "description": desc,
        "unit": unit,
        "positive": positive,
        "stats": {"type": "continuous", "mean": float(mean), "std": float(std)},
    }


def _cat(name: str, desc: str, proportions: Dict[str, float]) -> Dict[str, Any]:
    cats = list(proportions.keys())
    total = sum(proportions.values())
    norm = {k: float(v) / total for k, v in proportions.items()}
    return {
        "name": name,
        "type": "categorical",
        "description": desc,
        "categories": cats,
        "stats": {"type": "categorical", "proportions": norm},
    }


# 120 distinct clinically grounded oncology covariates organized in 15 domains (8 per block).
CLINICAL_FEATURE_CATALOG: Tuple[Dict[str, Any], ...] = (
    # Block 1: Demographics & Functional Status
    _cont("age", "Age at index date", 66.0, 9.5, "years"),
    _cont("bmi", "Body mass index", 26.2, 4.8, "kg/m2"),
    _cont("bsa", "Body surface area", 1.85, 0.22, "m2"),
    _cont("weight_loss_pct", "Six-month pretreatment weight loss", 4.5, 4.0, "%"),
    _cat("ecog", "ECOG performance status", {"0": 0.28, "1": 0.48, "2": 0.19, "3": 0.05}),
    _cont("karnofsky_score", "Karnofsky performance score", 80.0, 12.0, "points"),
    _cont("charlson_index", "Charlson comorbidity index", 3.2, 2.1, "points"),
    _cont("frailty_index", "Clinical frailty index", 0.24, 0.12, "score"),
    # Block 2: Vitals & Cardiopulmonary
    _cont("systolic_bp", "Baseline systolic blood pressure", 128.0, 16.0, "mmHg"),
    _cont("diastolic_bp", "Baseline diastolic blood pressure", 76.0, 10.0, "mmHg"),
    _cont("heart_rate", "Resting heart rate", 82.0, 13.0, "bpm"),
    _cont("respiratory_rate", "Baseline respiratory rate", 17.5, 2.8, "breaths/min"),
    _cont("spo2_pct", "Resting oxygen saturation", 96.0, 2.2, "%"),
    _cont("fev1_pct_pred", "FEV1 percent predicted", 74.0, 18.0, "%"),
    _cont("dlco_pct_pred", "DLCO percent predicted", 68.0, 17.0, "%"),
    _cont("lvef_pct", "Left ventricular ejection fraction", 58.0, 7.5, "%"),
    # Block 3: Complete Blood Count
    _cont("hemoglobin", "Serum hemoglobin", 12.1, 1.7, "g/dL"),
    _cont("hematocrit", "Hematocrit", 36.5, 4.8, "%"),
    _cont("wbc", "White blood cell count", 8.4, 3.1, "K/uL"),
    _cont("anc", "Absolute neutrophil count", 5.6, 2.5, "K/uL"),
    _cont("alc", "Absolute lymphocyte count", 1.5, 0.7, "K/uL"),
    _cont("platelets", "Platelet count", 265.0, 88.0, "K/uL"),
    _cont("rdw_pct", "Red cell distribution width", 14.8, 2.1, "%"),
    _cont("baseline_nlr", "Neutrophil-to-lymphocyte ratio", 4.4, 2.6, "ratio"),
    # Block 4: Inflammatory & Coagulation
    _cont("baseline_plr", "Platelet-to-lymphocyte ratio", 195.0, 90.0, "ratio"),
    _cont("crp_mg_l", "C-reactive protein", 18.0, 16.0, "mg/L"),
    _cont("esr_mm_hr", "Erythrocyte sedimentation rate", 38.0, 22.0, "mm/hr"),
    _cont("ferritin_ng_ml", "Serum ferritin", 240.0, 160.0, "ng/mL"),
    _cont("ldh_u_l", "Lactate dehydrogenase", 255.0, 115.0, "U/L"),
    _cont("fibrinogen_mg_dl", "Plasma fibrinogen", 410.0, 95.0, "mg/dL"),
    _cont("d_dimer_ug_ml", "D-dimer level", 1.1, 0.9, "ug/mL"),
    _cont("monocyte_count", "Absolute monocyte count", 0.65, 0.28, "K/uL"),
    # Block 5: Renal & Electrolytes
    _cont("creatinine_mg_dl", "Serum creatinine", 1.02, 0.34, "mg/dL"),
    _cont("creatinine_clearance", "Estimated creatinine clearance", 76.0, 23.0, "mL/min"),
    _cont("bun_mg_dl", "Blood urea nitrogen", 18.5, 7.2, "mg/dL"),
    _cont("sodium_meq_l", "Serum sodium", 138.0, 3.6, "mEq/L"),
    _cont("potassium_meq_l", "Serum potassium", 4.2, 0.45, "mEq/L"),
    _cont("calcium_mg_dl", "Corrected serum calcium", 9.3, 0.65, "mg/dL"),
    _cont("magnesium_mg_dl", "Serum magnesium", 2.0, 0.25, "mg/dL"),
    _cont("uric_acid_mg_dl", "Serum uric acid", 5.8, 1.6, "mg/dL"),
    # Block 6: Hepatic & Nutritional
    _cont("albumin", "Serum albumin", 3.7, 0.55, "g/dL"),
    _cont("total_bilirubin", "Total bilirubin", 0.72, 0.42, "mg/dL"),
    _cont("ast_u_l", "Aspartate aminotransferase", 29.0, 16.0, "U/L"),
    _cont("alt_u_l", "Alanine aminotransferase", 27.0, 15.0, "U/L"),
    _cont("alkaline_phosphatase", "Alkaline phosphatase", 108.0, 58.0, "U/L"),
    _cont("prealbumin_mg_dl", "Serum prealbumin", 21.0, 6.2, "mg/dL"),
    _cont("glucose_mg_dl", "Baseline serum glucose", 112.0, 32.0, "mg/dL"),
    _cont("hba1c_pct", "Hemoglobin A1c", 6.0, 1.0, "%"),
    # Block 7: Cardiac & Endocrine
    _cont("troponin_ng_l", "High-sensitivity troponin", 14.0, 11.0, "ng/L"),
    _cont("nt_probnp_pg_ml", "NT-proBNP", 290.0, 260.0, "pg/mL"),
    _cont("tsh_miu_l", "Thyroid-stimulating hormone", 2.3, 1.4, "mIU/L"),
    _cont("cortisol_ug_dl", "Morning serum cortisol", 13.5, 4.5, "ug/dL"),
    _cont("vitamin_d_ng_ml", "25-hydroxy vitamin D", 28.0, 11.0, "ng/mL"),
    _cont("cholesterol_mg_dl", "Total cholesterol", 184.0, 38.0, "mg/dL"),
    _cont("triglycerides_mg_dl", "Serum triglycerides", 142.0, 62.0, "mg/dL"),
    _cont("serum_protein_g_dl", "Total serum protein", 6.8, 0.65, "g/dL"),
    # Block 8: Tumor Histology & Staging
    _cat("histology", "Tumor histology", {"Adenocarcinoma": 0.58, "Squamous": 0.28, "Large_cell_or_other": 0.14}),
    _cat("tumor_grade", "Histologic differentiation grade", {"Well": 0.18, "Moderate": 0.44, "Poor": 0.38}),
    _cat("t_stage", "Primary T stage", {"T1_T2": 0.32, "T3": 0.31, "T4": 0.37}),
    _cat("n_stage", "Regional nodal N stage", {"N0_N1": 0.30, "N2": 0.42, "N3": 0.28}),
    _cat("m_stage", "Metastatic M stage", {"M1a": 0.26, "M1b": 0.34, "M1c": 0.40}),
    _cont("sum_target_diameters_mm", "RECIST sum of target lesion diameters", 78.0, 42.0, "mm"),
    _cont("primary_tumor_cm", "Primary lesion diameter", 4.6, 2.3, "cm"),
    _cont("metastatic_sites_count", "Number of involved metastatic organ systems", 2.4, 1.3, "sites"),
    # Block 9: Organ Metastasis Burden
    _cat("liver_metastases", "Baseline liver metastases", {"Absent": 0.74, "Present": 0.26}),
    _cat("brain_metastases", "Baseline brain metastases", {"Absent": 0.80, "Treated": 0.12, "Untreated": 0.08}),
    _cat("bone_metastases", "Baseline osseous metastases", {"Absent": 0.62, "Present": 0.38}),
    _cat("adrenal_metastases", "Baseline adrenal metastases", {"Absent": 0.82, "Present": 0.18}),
    _cat("pleural_effusion", "Malignant pleural effusion", {"Absent": 0.72, "Present": 0.28}),
    _cat("pericardial_effusion", "Baseline pericardial effusion", {"Absent": 0.90, "Present": 0.10}),
    _cat("lymphangitic_spread", "Lymphangitic carcinomatosis", {"Absent": 0.86, "Present": 0.14}),
    _cat("visceral_crisis", "Symptomatic high visceral burden", {"Absent": 0.84, "Present": 0.16}),
    # Block 10: Immunotherapy & Molecular Biomarkers
    _cont("pdl1_tps_pct", "PD-L1 tumor proportion score", 28.0, 26.0, "%"),
    _cont("tmb_mut_mb", "Tumor mutational burden", 9.8, 7.2, "mut/Mb"),
    _cat("msi_status", "Microsatellite instability status", {"MSS": 0.93, "MSI_High": 0.07}),
    _cont("ctdna_vaf_pct", "Circulating tumor DNA max VAF", 4.8, 4.5, "%"),
    _cont("ki67_pct", "Tumor Ki-67 proliferation index", 42.0, 21.0, "%"),
    _cont("til_density", "Tumor-infiltrating lymphocyte density score", 35.0, 18.0, "score"),
    _cont("interferon_gamma_score", "IFN-gamma gene expression signature", 0.0, 1.0, "z", positive=False),
    _cat("hla_loh", "HLA class I loss of heterozygosity", {"Intact": 0.72, "LOH": 0.28}),
    # Block 11: Somatic Genomic Alterations
    _cat("kras_mutation", "KRAS mutation status", {"Wildtype": 0.68, "G12C": 0.15, "Non_G12C": 0.17}),
    _cat("egfr_mutation", "EGFR mutation status", {"Wildtype": 0.84, "Sensitizing": 0.11, "Exon20_or_other": 0.05}),
    _cat("tp53_mutation", "TP53 alteration status", {"Wildtype": 0.44, "Mutated": 0.56}),
    _cat("stk11_mutation", "STK11/LKB1 alteration status", {"Wildtype": 0.82, "Mutated": 0.18}),
    _cat("keap1_mutation", "KEAP1 alteration status", {"Wildtype": 0.84, "Mutated": 0.16}),
    _cat("pik3ca_mutation", "PIK3CA alteration status", {"Wildtype": 0.88, "Mutated": 0.12}),
    _cat("met_amplification", "MET amplification or exon 14 skip", {"Absent": 0.91, "Present": 0.09}),
    _cat("erbb2_alteration", "ERBB2/HER2 alteration status", {"Absent": 0.92, "Present": 0.08}),
    # Block 12: Prior Oncologic Therapy
    _cat("prior_lines_therapy", "Prior systemic therapy lines", {"0": 0.45, "1": 0.35, "2_plus": 0.20}),
    _cat("prior_platinum", "Prior platinum chemotherapy", {"None": 0.42, "First-line": 0.38, "Multiple": 0.20}),
    _cat("prior_immunotherapy", "Prior immune checkpoint inhibitor", {"None": 0.68, "Prior_PD1_PDL1": 0.32}),
    _cat("prior_targeted_therapy", "Prior tyrosine kinase inhibitor", {"None": 0.82, "Present": 0.18}),
    _cat("prior_thoracic_radiation", "Prior thoracic radiotherapy", {"None": 0.64, "Definitive": 0.20, "Palliative": 0.16}),
    _cat("prior_surgery", "Prior curative-intent resection", {"None": 0.72, "Resected": 0.28}),
    _cont("months_since_diagnosis", "Months since initial cancer diagnosis", 14.0, 12.0, "months"),
    _cont("treatment_free_interval_mo", "Months since last systemic therapy", 5.5, 5.0, "months"),
    # Block 13: Concomitant Medications & Comorbidities
    _cat("baseline_steroids", "Baseline systemic corticosteroids (>=10mg pred eq)", {"None": 0.78, "Present": 0.22}),
    _cat("recent_antibiotics", "Systemic antibiotics within 30 days", {"None": 0.76, "Present": 0.24}),
    _cat("ppi_use", "Baseline proton pump inhibitor use", {"None": 0.62, "Present": 0.38}),
    _cat("anticoagulation", "Therapeutic anticoagulation", {"None": 0.80, "Present": 0.20}),
    _cat("copd_severity", "Chronic obstructive pulmonary disease", {"None": 0.56, "Mild_Moderate": 0.30, "Severe": 0.14}),
    _cat("diabetes_status", "Diabetes mellitus", {"None": 0.72, "Controlled": 0.19, "Uncontrolled": 0.09}),
    _cat("coronary_artery_disease", "Coronary artery disease", {"Absent": 0.78, "Present": 0.22}),
    _cat("autoimmune_history", "History of autoimmune disease", {"Absent": 0.89, "Present": 0.11}),
    # Block 14: Healthcare Utilization & Lifestyle
    _cat("smoking_status", "Tobacco smoking history", {"Never": 0.22, "Former": 0.52, "Current": 0.26}),
    _cont("smoking_pack_years", "Cumulative smoking pack-years", 26.0, 20.0, "pack-years"),
    _cat("alcohol_use", "Baseline alcohol consumption", {"None": 0.44, "Moderate": 0.42, "Heavy": 0.14}),
    _cont("prior_ed_visits_180d", "Emergency visits in prior 180 days", 0.9, 1.0, "visits"),
    _cont("prior_hospitalizations_180d", "Inpatient admissions in prior 180 days", 0.6, 0.8, "admissions"),
    _cat("palliative_care_consult", "Baseline palliative care involvement", {"None": 0.74, "Engaged": 0.26}),
    _cat("social_support", "Caregiver and social support", {"Strong": 0.55, "Moderate": 0.30, "Limited": 0.15}),
    _cat("insurance_type", "Primary health insurance", {"Commercial": 0.36, "Medicare": 0.48, "Medicaid_Other": 0.16}),
    # Block 15: Patient-Reported Outcomes (PRO-CTCAE / EORTC)
    _cont("pro_fatigue", "PRO-CTCAE baseline fatigue severity (0-4)", 1.8, 1.0, "score"),
    _cont("pro_pain", "PRO-CTCAE baseline pain severity (0-4)", 1.4, 1.0, "score"),
    _cont("pro_dyspnea", "PRO-CTCAE baseline dyspnea severity (0-4)", 1.3, 0.95, "score"),
    _cont("pro_nausea", "PRO-CTCAE baseline nausea severity (0-4)", 0.7, 0.75, "score"),
    _cont("pro_appetite_loss", "PRO-CTCAE decreased appetite severity (0-4)", 1.2, 0.95, "score"),
    _cont("pro_neuropathy", "PRO-CTCAE sensory neuropathy severity (0-4)", 0.9, 0.85, "score"),
    _cont("pro_insomnia", "PRO-CTCAE insomnia severity (0-4)", 1.3, 0.95, "score"),
    _cont("pro_global_qol", "EORTC global quality of life score (0-100)", 62.0, 19.0, "points"),
)


def build_feature_catalog(num_features: int) -> List[Dict[str, Any]]:
    """Return ``num_features`` clinical feature definitions (extending cleanly beyond 120 if needed)."""
    if num_features < 1:
        raise ValueError("num_features must be at least 1")
    catalog = [dict(item) for item in CLINICAL_FEATURE_CATALOG[:num_features]]
    if len(catalog) < num_features:
        for idx in range(len(catalog), num_features):
            panel_idx = idx - len(CLINICAL_FEATURE_CATALOG) + 1
            if panel_idx % 3 == 0:
                catalog.append(
                    _cat(
                        f"molecular_pathway_{panel_idx:03d}",
                        f"Molecular pathway signature {panel_idx}",
                        {"Low": 0.45, "Intermediate": 0.35, "High": 0.20},
                    )
                )
            elif panel_idx % 3 == 1:
                catalog.append(
                    _cont(
                        f"proteomic_marker_{panel_idx:03d}",
                        f"Plasma proteomic biomarker {panel_idx}",
                        12.0 + (panel_idx % 7) * 3.0,
                        4.5,
                        "ng/mL",
                    )
                )
            else:
                catalog.append(
                    _cont(
                        f"radiomic_feature_{panel_idx:03d}",
                        f"Baseline CT radiomic heterogeneity feature {panel_idx}",
                        0.0,
                        1.0,
                        "z",
                        positive=False,
                    )
                )
    return catalog


def _block_correlation_cholesky(
    p: int,
    *,
    block_size: int = 8,
    within_block_corr: float = 0.30,
    cross_block_corr: float = 0.04,
) -> np.ndarray:
    """Positive-definite Cholesky factor for block-correlated clinical covariates."""
    if not (0.0 <= cross_block_corr <= within_block_corr < 0.95):
        raise ValueError("Require 0 <= cross_block_corr <= within_block_corr < 0.95")
    cov = np.full((p, p), float(cross_block_corr), dtype=float)
    for start in range(0, p, block_size):
        end = min(p, start + block_size)
        cov[start:end, start:end] = float(within_block_corr)
    np.fill_diagonal(cov, 1.0)
    return np.linalg.cholesky(cov)


def _sample_wide_covariates(
    catalog: Sequence[Mapping[str, Any]],
    n_patients: int,
    rng: np.random.Generator,
    *,
    continuous_distribution: str = "positive_lognormal",
    within_block_corr: float = 0.30,
    cross_block_corr: float = 0.04,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Sample block-correlated continuous and categorical covariates via Gaussian copula."""
    p = len(catalog)
    chol = _block_correlation_cholesky(
        p, within_block_corr=within_block_corr, cross_block_corr=cross_block_corr
    )
    z_mat = rng.standard_normal((n_patients, p)) @ chol.T
    columns: Dict[str, Any] = {}
    summary_stats: Dict[str, Any] = {}

    for col_idx, spec in enumerate(catalog):
        name = spec["name"]
        z = z_mat[:, col_idx]
        summary_stats[name] = dict(spec["stats"])
        if spec["type"] == "continuous":
            mean = float(spec["stats"]["mean"])
            std = float(spec["stats"]["std"])
            if continuous_distribution == "positive_lognormal" and spec.get("positive", True) and mean > 0:
                sigma2 = math.log1p((std / mean) ** 2)
                mu = math.log(mean) - 0.5 * sigma2
                vals = np.exp(mu + math.sqrt(sigma2) * z)
            else:
                vals = mean + std * z
            columns[name] = vals
        else:
            cats = list(spec["categories"])
            probs = np.array([spec["stats"]["proportions"][c] for c in cats], dtype=float)
            probs = probs / probs.sum()
            cum = np.cumsum(probs)
            u = ndtr(z)
            indices = np.minimum(np.searchsorted(cum, u, side="right"), len(cats) - 1)
            cat_arr = np.asarray(cats, dtype=object)
            columns[name] = cat_arr[indices]
    return columns, summary_stats


def _render_pretreatment_text(
    patient_idx: int,
    catalog: Sequence[Mapping[str, Any]],
    columns: Mapping[str, Any],
) -> Tuple[str, str]:
    """Render deterministic structured pretreatment notes and patient prompt (zero treatment/outcome leak)."""
    prompt_lines: List[str] = []
    sections: Dict[str, List[str]] = {
        "Oncology Consult Note": [],
        "Laboratory & Biomarker Summary": [],
        "Staging & Molecular Profile": [],
        "Medications, History & PRO Assessment": [],
    }
    section_names = list(sections.keys())
    for idx, spec in enumerate(catalog):
        name = spec["name"]
        desc = spec["description"]
        val = columns[name][patient_idx]
        if spec["type"] == "continuous":
            unit = f" {spec['unit']}" if spec.get("unit") else ""
            formatted = f"{float(val):.2f}{unit}"
        else:
            formatted = str(val)
        prompt_lines.append(f"- {desc} ({name}): {formatted}")
        sec = section_names[(idx // 8) % len(section_names)]
        sections[sec].append(f"{desc}: {formatted}.")

    notes = []
    for title, items in sections.items():
        if items:
            notes.append(f"{title} (Pretreatment, Patient {patient_idx}):\n" + " ".join(items))
    return "\n\n<new_note>\n\n".join(notes), "\n".join(prompt_lines)


def generate_wide_source_cohort(
    output_dir: Path,
    *,
    num_patients: int = 5000,
    num_features: int = 120,
    num_confounders: Optional[int] = None,
    num_effect_modifiers: Optional[int] = None,
    seed: int = 42,
    outcome_type: str = "binary",
    continuous_distribution: str = "positive_lognormal",
    within_block_corr: float = 0.30,
    cross_block_corr: float = 0.04,
    dgp_mode: str = "composite",
    clinical_question: str = DEFAULT_CLINICAL_QUESTION,
    render_text: bool = True,
    structured_data_enabled: bool = False,
    min_events_per_patient: int = 15,
    max_events_per_patient: int = 30,
) -> Path:
    """Write a wide-feature, re-equation-ready synthetic clinical cohort to ``output_dir``.

    Args:
        output_dir: Destination directory (writes ``dataset.parquet``,
            ``metadata.json``, and ``generation_config.json``).
        num_patients: Number of patients to simulate.
        num_features: Total candidate features (supports 10..150+; default 120 so
            scenarios can assign 5..50 confounders and 5..50 modifiers freely).
        num_confounders: Initial confounder count tagged in source metadata
            (defaults to ``num_features // 2``).
        num_effect_modifiers: Initial effect-modifier count tagged in source metadata
            (defaults to ``num_features - num_confounders``).
        seed: Random seed.
        outcome_type: ``"binary"`` or ``"continuous"``.
        continuous_distribution: ``"positive_lognormal"`` or ``"normal"``.
        within_block_corr: Within-domain Gaussian copula correlation.
        cross_block_corr: Cross-domain Gaussian copula correlation.
        dgp_mode: Initial DGP mode for the baseline source treatment/outcome draw.
        clinical_question: Clinical question recorded in config/metadata.
        render_text: If True (default), writes pretreatment ``clinical_text`` and
            ``patient_prompt`` columns; set False for ultra-fast tabular-only sources
            (still writes concise clinical_text placeholders).
    """
    if num_patients < 1:
        raise ValueError("num_patients must be at least 1")
    if num_features < 2:
        raise ValueError("num_features must be at least 2")
    if outcome_type not in {"binary", "continuous"}:
        raise ValueError(f"outcome_type must be 'binary' or 'continuous', got {outcome_type!r}")

    n_conf = (num_features // 2) if num_confounders is None else int(num_confounders)
    n_mod = (num_features - n_conf) if num_effect_modifiers is None else int(num_effect_modifiers)
    if n_conf < 0 or n_mod < 0 or n_conf > num_features or n_mod > num_features:
        raise ValueError("Invalid num_confounders / num_effect_modifiers for num_features")

    rng = np.random.default_rng(int(seed))
    catalog = build_feature_catalog(num_features)
    columns, summary_stats = _sample_wide_covariates(
        catalog,
        num_patients,
        rng,
        continuous_distribution=continuous_distribution,
        within_block_corr=within_block_corr,
        cross_block_corr=cross_block_corr,
    )

    # Tag default source roles (re-equation can freely reassign any subset/range of roles later).
    base_features: List[Dict[str, Any]] = []
    for idx, spec in enumerate(catalog):
        feat: Dict[str, Any] = {
            "name": spec["name"],
            "type": spec["type"],
            "description": spec["description"],
            "roles": [],
        }
        if spec["type"] == "categorical":
            feat["categories"] = list(spec["categories"])
        base_features.append(feat)

    # Assign source roles cleanly (disjoint when n_conf + n_mod <= num_features, dual_role overlap otherwise).
    overlap = max(0, n_conf + n_mod - num_features)
    init_scenario = {
        "name": "wide_source_initial",
        "roles": {
            "confounder": n_conf - overlap,
            "effect_modifier": n_mod - overlap,
            "dual_role": overlap,
        },
        "dgp_mode": dgp_mode,
        "nonlinearity": "mixed" if dgp_mode != "parametric" else "none",
        "treatment_nonlinearity": "mixed" if dgp_mode != "parametric" else "none",
        "modifier_main_effects": dgp_mode != "parametric",
    }
    resolved = resolve_scenario(init_scenario, base_features, seed=int(seed))
    features = _features_with_roles(base_features, resolved["roles"])
    defaults = SyntheticDataConfig()
    treatment_eq, outcome_eq, _, _ = draw_equations(
        features,
        resolved,
        int(seed),
        main_coefficient_scale=defaults.main_coefficient_scale,
        interaction_coefficient_scale=defaults.interaction_coefficient_scale,
        summary_stats=summary_stats,
    )

    names = [f["name"] for f in features]
    characteristics = [
        {name: (float(columns[name][i]) if features[j]["type"] == "continuous" else str(columns[name][i]))
         for j, name in enumerate(names)}
        for i in range(num_patients)
    ]

    mc_rows = rng.integers(0, num_patients, min(10000, max(2000, num_patients)))
    mc_chars = [characteristics[int(r)] for r in mc_rows]
    treatment_eq, outcome_eq = g._rescale_for_target_logit_std(
        confounders=features,
        summary_stats=summary_stats,
        treatment_eq=treatment_eq,
        outcome_eq=outcome_eq,
        treatment_logit_std=defaults.target_logit_std,
        outcome_logit_std=defaults.target_logit_std,
        sampled_chars=mc_chars,
    )
    treatment_eq, outcome_eq = g._calibrate_intercepts(
        confounders=features,
        summary_stats=summary_stats,
        treatment_eq=treatment_eq,
        outcome_eq=outcome_eq,
        target_treatment_rate=defaults.target_treatment_rate,
        target_control_outcome_rate=defaults.target_control_outcome_rate,
        sampled_chars=mc_chars,
    )

    term_matrix = g._build_term_matrix(characteristics, features, summary_stats)
    tx_lp = _linear_predictors(characteristics, features, summary_stats, treatment_eq, None, term_matrix=term_matrix)
    lp0 = _linear_predictors(characteristics, features, summary_stats, outcome_eq, 0, term_matrix=term_matrix)
    lp1 = _linear_predictors(characteristics, features, summary_stats, outcome_eq, 1, term_matrix=term_matrix)
    propensity = _sigmoid(tx_lp)
    treatment = (rng.random(num_patients) < propensity).astype("int64")

    data: Dict[str, Any] = {"patient_id": np.arange(num_patients, dtype=np.int64)}
    if render_text:
        texts, prompts = [], []
        for i in range(num_patients):
            txt, prm = _render_pretreatment_text(i, catalog, columns)
            texts.append(txt)
            prompts.append(prm)
        data["clinical_text"] = texts
        data["patient_prompt"] = prompts
        data["num_notes"] = np.full(num_patients, 4, dtype=np.int64)
    else:
        data["clinical_text"] = [f"Pretreatment record for patient {i}." for i in range(num_patients)]
        data["patient_prompt"] = [f"Patient {i}" for i in range(num_patients)]
        data["num_notes"] = np.ones(num_patients, dtype=np.int64)

    data["treatment_indicator"] = treatment
    data["true_treatment_prob"] = propensity
    if outcome_type == "continuous":
        lp_obs = np.where(treatment == 1, lp1, lp0)
        outcome = lp_obs + rng.normal(0.0, defaults.outcome_noise_std, num_patients)
        data["outcome_indicator"] = outcome
        data["true_y0"] = lp0
        data["true_y1"] = lp1
        data["true_ite"] = lp1 - lp0
    else:
        p0, p1 = g.potential_outcome_probabilities(lp0, lp1, outcome_eq)
        p_obs = np.where(treatment == 1, p1, p0)
        data["outcome_indicator"] = (rng.random(num_patients) < p_obs).astype("int64")
        data["true_outcome_prob"] = p_obs
        data["true_y0_prob"] = p0
        data["true_y1_prob"] = p1
        data["true_ite_prob"] = p1 - p0
        data["true_y0"] = p0
        data["true_y1"] = p1
        data["true_ite"] = p1 - p0

    for spec in catalog:
        name = spec["name"]
        if spec["type"] == "continuous":
            data[f"true_{name}"] = np.asarray(columns[name], dtype=float)
        else:
            data[f"true_{name}"] = [str(v) for v in columns[name]]

    frame = pd.DataFrame(data)
    config = SyntheticDataConfig(
        clinical_question=clinical_question,
        dataset_size=num_patients,
        num_features=num_features,
        num_confounders=n_conf,
        num_effect_modifiers=n_mod,
        outcome_type=outcome_type,
        continuous_covariate_distribution=continuous_distribution,
        covariate_correlation=within_block_corr,
        dgp_mode=dgp_mode,
        structured_data=StructuredDataConfig(enabled=structured_data_enabled),
        min_events_per_patient=min_events_per_patient,
        max_events_per_patient=max_events_per_patient,
        seed=int(seed),
        output_dir=str(output_dir),
    )
    public_config = config.to_public_dict()
    lists = role_lists(features)
    metadata = {
        "config": public_config,
        "features": features,
        "confounders": lists["confounders"],
        "effect_modifiers": lists["effect_modifiers"],
        "instruments": lists["instruments"],
        "prognostic_variables": lists["prognostic_variables"],
        "proxies": [],
        "treatment_equation": treatment_eq,
        "outcome_equation": outcome_eq,
        "summary_statistics": summary_stats,
        "unmeasured_confounders": [],
        "scenario": None,
        "documentation": None,
        "post_treatment": None,
        "wide_corpus": {
            "num_features": num_features,
            "within_block_corr": within_block_corr,
            "cross_block_corr": cross_block_corr,
            "render_text": render_text,
        },
        "dataset_statistics": g.oracle_dataset_statistics(frame),
    }

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "dataset.parquet", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (output_dir / "generation_config.json").write_text(
        json.dumps(public_config, indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info(
        "Generated wide synthetic corpus (%d patients, %d features) in %s",
        num_patients, num_features, output_dir,
    )
    return output_dir


def wide_corpus_main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m synthetic_data.cli wide-corpus``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m synthetic_data.cli wide-corpus",
        description=(
            "Generate a fast deterministic wide-feature synthetic clinical corpus "
            "(10-150+ correlated features) compatible with re-equation and S12 benchmarks."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-patients", type=int, default=5000, help="Number of patients (default: 5000)")
    parser.add_argument("--num-features", type=int, default=120, help="Number of features (default: 120)")
    parser.add_argument("--num-confounders", type=int, default=None, help="Initial confounder count")
    parser.add_argument("--num-effect-modifiers", type=int, default=None, help="Initial effect-modifier count")
    parser.add_argument("--outcome-type", choices=("binary", "continuous"), default="binary")
    parser.add_argument("--within-block-corr", type=float, default=0.30)
    parser.add_argument("--cross-block-corr", type=float, default=0.04)
    parser.add_argument("--dgp-mode", default="composite")
    parser.add_argument("--clinical-question", default=DEFAULT_CLINICAL_QUESTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--structured-data", action="store_true", help="Enable structured_data flag in saved config (for LLM expansion)")
    parser.add_argument("--min-events", type=int, default=15, help="min_events_per_patient recorded in generation_config.json")
    parser.add_argument("--max-events", type=int, default=30, help="max_events_per_patient recorded in generation_config.json")
    parser.add_argument("--no-text", action="store_true", help="Write minimal text placeholders for tabular-only runs")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    path = generate_wide_source_cohort(
        Path(args.output_dir),
        num_patients=args.num_patients,
        num_features=args.num_features,
        num_confounders=args.num_confounders,
        num_effect_modifiers=args.num_effect_modifiers,
        seed=args.seed,
        outcome_type=args.outcome_type,
        within_block_corr=args.within_block_corr,
        cross_block_corr=args.cross_block_corr,
        dgp_mode=args.dgp_mode,
        clinical_question=args.clinical_question,
        render_text=not args.no_text,
        structured_data_enabled=args.structured_data,
        min_events_per_patient=args.min_events,
        max_events_per_patient=args.max_events,
    )
    print(path)
    return 0
