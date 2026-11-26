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


# Default comparative effectiveness question per oncology context.
ONCOLOGY_CONTEXT_QUESTIONS: Dict[str, str] = {
    "nsclc": "Compare vinorelbine to gemcitabine for advanced/metastatic NSCLC.",
    "myeloma": (
        "Compare daratumumab+lenalidomide+dexamethasone with "
        "carfilzomib+lenalidomide+dexamethasone for relapsed/refractory multiple myeloma."
    ),
    "hcc": (
        "Compare atezolizumab+bevacizumab with durvalumab+tremelimumab "
        "for first-line unresectable hepatocellular carcinoma."
    ),
    "mcrc": (
        "Compare FOLFOX+bevacizumab with FOLFIRI+bevacizumab "
        "for first-line metastatic colorectal cancer."
    ),
    "breast_er_pos": (
        "Compare letrozole+palbociclib with letrozole+ribociclib "
        "for first-line metastatic ER-positive HER2-negative breast cancer."
    ),
    "mcrpc": (
        "Compare enzalutamide with abiraterone+prednisone "
        "for first-line metastatic castration-resistant prostate cancer."
    ),
}

# Disease-specific replacements for Blocks 2 (indices 8..16) and 8..13 (indices 56..104)
# so every oncology context has 120 clinically authentic, distinct features.
_CONTEXT_SPECIFIC_BLOCKS: Dict[str, Tuple[Tuple[Dict[str, Any], ...], Tuple[Dict[str, Any], ...]]] = {
    "myeloma": (
        # Block 2 (8): Vitals, Cardiac & Renal Functional Reserve
        (
            _cont("systolic_bp", "Baseline systolic blood pressure", 132.0, 17.0, "mmHg"),
            _cont("diastolic_bp", "Baseline diastolic blood pressure", 78.0, 10.0, "mmHg"),
            _cont("heart_rate", "Resting heart rate", 80.0, 12.0, "bpm"),
            _cont("qtc_interval_ms", "Baseline corrected QT interval (QTcF)", 432.0, 24.0, "ms"),
            _cont("lvef_pct", "Left ventricular ejection fraction", 57.0, 8.0, "%"),
            _cont("urine_protein_24h_g", "24-hour urine M-protein excretion", 0.85, 0.90, "g/24h"),
            _cont("serum_viscosity_cp", "Serum viscosity", 1.9, 0.55, "cP"),
            _cont("orthostatic_sbp_drop_mmhg", "Orthostatic systolic blood pressure drop", 11.0, 8.0, "mmHg"),
        ),
        # Blocks 8-13 (48): Myeloma Staging, Disease Burden, Biomarkers, Cytogenetics, Prior Tx, Comorbidities
        (
            # Block 8: Staging & Paraprotein Burden
            _cat("myeloma_isotype", "Monoclonal immunoglobulin isotype", {"IgG": 0.54, "IgA": 0.24, "Light_chain_only": 0.22}),
            _cat("r_iss_stage", "Revised International Staging System (R-ISS)", {"Stage_I": 0.22, "Stage_II": 0.56, "Stage_III": 0.22}),
            _cont("serum_m_protein_g_dl", "Serum monoclonal M-protein", 2.8, 1.9, "g/dL"),
            _cont("beta2_microglobulin_mg_l", "Serum beta-2 microglobulin", 4.6, 2.8, "mg/L"),
            _cont("involved_flc_mg_l", "Involved serum free light chain", 480.0, 420.0, "mg/L"),
            _cont("flc_ratio", "Involved-to-uninvolved free light chain ratio", 42.0, 38.0, "ratio"),
            _cont("marrow_plasma_cell_pct", "Bone marrow clonal plasma cell percentage", 44.0, 24.0, "%"),
            _cont("iga_igg_uninvolved_suppression", "Uninvolved immunoglobulin suppression index", 0.65, 0.25, "index"),
            # Block 9: CRAB & Skeletal / Extramedullary Involvement
            _cat("lytic_bone_lesions", "Skeletal survey/PET-CT lytic bone lesions", {"None": 0.22, "One_to_three": 0.36, "Extensive_four_plus": 0.42}),
            _cat("extramedullary_disease", "Extramedullary plasmacytoma", {"Absent": 0.84, "Paraskeletal": 0.10, "True_extramedullary": 0.06}),
            _cat("pathologic_fracture_history", "Prior vertebral or long-bone pathologic fracture", {"Absent": 0.72, "Present": 0.28}),
            _cat("hypercalcemia_at_relapse", "Symptomatic hypercalcemia at relapse", {"Absent": 0.85, "Present": 0.15}),
            _cat("renal_cast_nephropathy", "Myeloma-related renal impairment", {"Absent": 0.74, "Present": 0.26}),
            _cat("hyperviscosity_symptoms", "Clinical hyperviscosity symptoms", {"Absent": 0.92, "Present": 0.08}),
            _cat("spinal_cord_compression_risk", "Epidural or paraspinal cord impingement", {"Absent": 0.90, "Present": 0.10}),
            _cat("amyloidosis_overlap", "Concurrent AL amyloidosis features", {"Absent": 0.91, "Present": 0.09}),
            # Block 10: Immunophenotype, MRD & Prognostic Assays
            _cont("circulating_plasma_cells_pct", "Peripheral blood circulating plasma cells", 0.45, 0.55, "%"),
            _cat("mrd_prior_best_status", "Best prior measurable residual disease status", {"Never_achieved": 0.68, "MRD_negative_relapsed": 0.32}),
            _cont("cd38_expression_mfi", "Clonal plasma cell CD38 surface density", 18.0, 9.0, "kMFI"),
            _cont("bcma_soluble_ng_ml", "Serum soluble BCMA level", 145.0, 110.0, "ng/mL"),
            _cont("ki67_plasma_cell_pct", "Plasma cell Ki-67 proliferation index", 14.0, 10.0, "%"),
            _cat("pet_ct_fdg_avid_lesions", "FDG-avid focal marrow lesions on PET/CT", {"None": 0.28, "1_to_3": 0.38, "Over_3": 0.34}),
            _cont("sky92_gene_risk_score", "Myeloma high-risk gene expression score", 0.0, 1.0, "z", positive=False),
            _cat("immunoparesis_severity", "Polyclonal immunoparesis status", {"None": 0.22, "Partial": 0.43, "Severe": 0.35}),
            # Block 11: FISH Cytogenetics & Molecular Risk
            _cat("fish_del17p", "FISH del(17p) / TP53 deletion", {"Absent": 0.86, "Present": 0.14}),
            _cat("fish_t4_14", "FISH translocation t(4;14)", {"Absent": 0.85, "Present": 0.15}),
            _cat("fish_t14_16", "FISH translocation t(14;16)", {"Absent": 0.94, "Present": 0.06}),
            _cat("fish_gain_1q21", "FISH chromosome 1q21 gain/amplification", {"Absent": 0.62, "Gain_3_copies": 0.26, "Amp_4plus": 0.12}),
            _cat("fish_t11_14", "FISH translocation t(11;14)", {"Absent": 0.81, "Present": 0.19}),
            _cat("fish_del1p32", "FISH chromosome 1p32 deletion", {"Absent": 0.89, "Present": 0.11}),
            _cat("hyperdiploid_karyotype", "Hyperdiploid plasma cell karyotype", {"Non_hyperdiploid": 0.48, "Hyperdiploid": 0.52}),
            _cat("ras_braf_mutation", "KRAS/NRAS/BRAF somatic myeloma mutation", {"Wildtype": 0.56, "Mutated": 0.44}),
            # Block 12: Prior Myeloma Lines & Refractoriness
            _cat("prior_lines_therapy", "Prior systemic myeloma therapy lines", {"1": 0.38, "2": 0.34, "3_plus": 0.28}),
            _cat("prior_asct", "Prior autologous stem cell transplant", {"None": 0.46, "Single_ASCT": 0.44, "Tandem_ASCT": 0.10}),
            _cat("bortezomib_refractory", "Refractory to prior bortezomib", {"Sensitive_or_naive": 0.64, "Refractory": 0.36}),
            _cat("lenalidomide_exposure", "Prior lenalidomide exposure status", {"Naive": 0.24, "Exposed_sensitive": 0.44, "Refractory": 0.32}),
            _cat("anti_cd38_prior", "Prior anti-CD38 monoclonal antibody", {"Naive": 0.72, "Exposed": 0.28}),
            _cat("prior_palliative_bone_rt", "Prior palliative radiotherapy to bone lesions", {"None": 0.68, "Present": 0.32}),
            _cont("months_since_diagnosis", "Months since initial myeloma diagnosis", 34.0, 24.0, "months"),
            _cont("duration_prior_remission_mo", "Duration of immediately preceding remission", 16.0, 12.0, "months"),
            # Block 13: Myeloma-Specific Comorbidities & Supportive Meds
            _cat("baseline_peripheral_neuropathy", "Grade of baseline peripheral neuropathy", {"Grade_0": 0.46, "Grade_1": 0.36, "Grade_2_plus": 0.18}),
            _cat("vte_history", "Prior venous thromboembolism (DVT/PE)", {"Absent": 0.82, "Present": 0.18}),
            _cat("heart_failure_history", "Prior congestive heart failure or cardiomyopathy", {"Absent": 0.84, "Present": 0.16}),
            _cat("uncontrolled_hypertension", "Baseline refractory hypertension", {"Absent": 0.72, "Present": 0.28}),
            _cat("bone_modifying_agent", "Concurrent zoledronic acid or denosumab", {"None": 0.26, "Zoledronate": 0.46, "Denosumab": 0.28}),
            _cat("vte_prophylaxis_type", "Baseline antithrombotic prophylaxis", {"Aspirin": 0.56, "DOAC_or_LMWH": 0.30, "None": 0.14}),
            _cat("recurrent_infections_12m", "Serious bacterial/viral infections in past 12m", {"None": 0.66, "One": 0.22, "Two_plus": 0.12}),
            _cat("diabetes_status", "Diabetes mellitus", {"None": 0.70, "Controlled": 0.20, "Uncontrolled": 0.10}),
        ),
    ),
    "hcc": (
        # Block 2 (8): Vitals, Hemodynamics & Portal/Hepatic Physiology
        (
            _cont("systolic_bp", "Baseline systolic blood pressure", 126.0, 17.0, "mmHg"),
            _cont("diastolic_bp", "Baseline diastolic blood pressure", 74.0, 10.0, "mmHg"),
            _cont("heart_rate", "Resting heart rate", 78.0, 12.0, "bpm"),
            _cont("inr_ratio", "International normalized ratio (INR)", 1.24, 0.26, "ratio"),
            _cont("meld_na_score", "Model for End-Stage Liver Disease (MELD-Na)", 10.8, 3.8, "points"),
            _cont("albi_score", "Albumin-Bilirubin (ALBI) linear score", -2.35, 0.55, "score", positive=False),
            _cont("spleen_length_cm", "Craniocaudal spleen length on CT", 13.4, 2.8, "cm"),
            _cont("lvef_pct", "Left ventricular ejection fraction", 60.0, 6.5, "%"),
        ),
        # Blocks 8-13 (48): HCC Staging, Cirrhosis/Portal Burden, Biomarkers, Etiology/Genomics, Prior Tx, Comorbidities
        (
            # Block 8: Staging, Cirrhosis & Tumor Morphology
            _cat("bclc_stage", "Barcelona Clinic Liver Cancer (BCLC) stage", {"BCLC_B": 0.28, "BCLC_C": 0.72}),
            _cat("child_pugh_class", "Baseline Child-Pugh liver function class", {"A5": 0.46, "A6": 0.34, "B7": 0.20}),
            _cat("albi_grade", "Baseline ALBI grade", {"Grade_1": 0.42, "Grade_2": 0.50, "Grade_3": 0.08}),
            _cat("cirrhosis_etiology", "Underlying chronic liver disease etiology", {"HBV": 0.28, "HCV": 0.26, "Alcohol": 0.22, "MASLD_Other": 0.24}),
            _cont("largest_hepatic_lesion_cm", "Diameter of largest intrahepatic HCC mass", 6.4, 3.4, "cm"),
            _cont("hepatic_nodules_count", "Number of intrahepatic tumor nodules", 3.6, 2.4, "nodules"),
            _cont("liver_tumor_burden_pct", "Estimated percent hepatic parenchymal involvement", 24.0, 18.0, "%"),
            _cont("sum_target_diameters_mm", "mRECIST sum of viable target lesion diameters", 86.0, 48.0, "mm"),
            # Block 9: Vascular Invasion, Portal Hypertension & Extrahepatic Spread
            _cat("portal_vein_thrombus", "Macrovascular portal vein tumor thrombus (PVTT)", {"None": 0.60, "Branch_Vp1_Vp2": 0.22, "Main_Vp3_Vp4": 0.18}),
            _cat("esophageal_varices", "Esophageal/gastric varices on baseline EGD", {"None": 0.46, "Small_low_risk": 0.34, "Large_or_banded": 0.20}),
            _cat("prior_variceal_bleed", "History of upper GI variceal hemorrhage", {"Absent": 0.86, "Remote_over_6m": 0.10, "Recent_within_6m": 0.04}),
            _cat("baseline_ascites", "Baseline radiographic or clinical ascites", {"Absent": 0.74, "Mild_controlled": 0.20, "Moderate": 0.06}),
            _cat("hepatic_encephalopathy", "History of hepatic encephalopathy", {"Absent": 0.86, "Grade_1_controlled": 0.14}),
            _cat("extrahepatic_lung_mets", "Extrahepatic pulmonary metastases", {"Absent": 0.70, "Present": 0.30}),
            _cat("extrahepatic_lymph_nodes", "Extrahepatic malignant lymphadenopathy", {"Absent": 0.68, "Present": 0.32}),
            _cat("extrahepatic_bone_mets", "Extrahepatic skeletal metastases", {"Absent": 0.84, "Present": 0.16}),
            # Block 10: Hepatic Tumor Markers & Immune/Vascular Biomarkers
            _cont("afp_ng_ml", "Serum alpha-fetoprotein (AFP)", 420.0, 680.0, "ng/mL"),
            _cont("dcp_pivka_ii_mau_ml", "Des-gamma-carboxy prothrombin (PIVKA-II)", 650.0, 820.0, "mAU/mL"),
            _cont("afp_l3_pct", "AFP-L3 lectin-reactive fraction", 22.0, 18.0, "%"),
            _cont("ggt_u_l", "Gamma-glutamyl transferase (GGT)", 145.0, 110.0, "U/L"),
            _cont("urine_protein_creatinine_ratio", "Spot urine protein-to-creatinine ratio", 0.28, 0.30, "mg/mg"),
            _cont("vegf_plasma_pg_ml", "Circulating plasma VEGF-A", 110.0, 85.0, "pg/mL"),
            _cont("ctdna_vaf_pct", "Circulating tumor DNA max VAF", 3.8, 3.6, "%"),
            _cont("interferon_gamma_score", "Hepatic immune inflamed gene signature", 0.0, 1.0, "z", positive=False),
            # Block 11: HCC Molecular Pathways & Viral Control
            _cat("hbv_viral_suppression", "Hepatitis B antiviral suppression status", {"HBV_negative": 0.72, "Suppressed_on_NUC": 0.22, "Detectable_DNA": 0.06}),
            _cat("hcv_svr_status", "Hepatitis C sustained virologic response", {"HCV_negative": 0.74, "Prior_SVR": 0.18, "Active_viremia": 0.08}),
            _cat("ctnnb1_mutation", "CTNNB1 (beta-catenin) WNT pathway alteration", {"Wildtype": 0.68, "Mutated": 0.32}),
            _cat("tert_promoter_mutation", "TERT promoter alteration status", {"Wildtype": 0.44, "Mutated": 0.56}),
            _cat("tp53_mutation", "TP53 somatic alteration status", {"Wildtype": 0.66, "Mutated": 0.34}),
            _cat("arid1a_mutation", "ARID1A chromatin remodeling mutation", {"Wildtype": 0.86, "Mutated": 0.14}),
            _cat("fgf19_amplification", "FGF19/CCND1 locus amplification", {"Absent": 0.88, "Present": 0.12}),
            _cat("fib4_score", "FIB-4 hepatic fibrosis category", {"Low_Intermediate": 0.32, "High_cirrhotic": 0.68}),
            # Block 12: Prior Locoregional & Surgical Liver Therapy
            _cat("prior_tace", "Prior transarterial chemoembolization (TACE)", {"None": 0.52, "1_to_2_sessions": 0.32, "3_plus_sessions": 0.16}),
            _cat("prior_y90_tare", "Prior Y-90 transarterial radioembolization", {"None": 0.76, "Present": 0.24}),
            _cat("prior_ablation_rfa_mwa", "Prior percutaneous thermal ablation", {"None": 0.78, "Present": 0.22}),
            _cat("prior_hepatic_resection", "Prior partial hepatectomy", {"None": 0.76, "Resected": 0.24}),
            _cat("prior_sbrt_liver", "Prior stereotactic body liver radiotherapy", {"None": 0.84, "Present": 0.16}),
            _cat("prior_tki_adjuvant", "Prior multikinase inhibitor exposure", {"None": 0.88, "Present": 0.12}),
            _cont("months_since_diagnosis", "Months since initial HCC diagnosis", 12.0, 11.0, "months"),
            _cont("months_since_locoregional", "Months since last locoregional liver therapy", 4.5, 4.2, "months"),
            # Block 13: Bleeding, Autoimmune & Cirrhosis Comorbidities
            _cat("uncontrolled_hypertension", "Baseline refractory hypertension", {"Absent": 0.68, "Present": 0.32}),
            _cat("arterial_thromboembolism_hx", "Prior MI, ischemic stroke, or arterial thrombosis", {"Absent": 0.86, "Present": 0.14}),
            _cat("anticoagulation", "Therapeutic anticoagulation or antiplatelet therapy", {"None": 0.78, "Antiplatelet": 0.14, "Anticoagulant": 0.08}),
            _cat("nonselective_beta_blocker", "Baseline carvedilol/propranolol for portal HTN", {"None": 0.58, "Present": 0.42}),
            _cat("autoimmune_history", "History of autoimmune disease", {"Absent": 0.90, "Present": 0.10}),
            _cat("baseline_steroids", "Baseline systemic corticosteroids", {"None": 0.88, "Present": 0.12}),
            _cat("diabetes_status", "Diabetes mellitus", {"None": 0.56, "Controlled": 0.28, "Uncontrolled": 0.16}),
            _cat("ppi_use", "Baseline proton pump inhibitor use", {"None": 0.50, "Present": 0.50}),
        ),
    ),
    "mcrc": (
        # Block 2 (8): Vitals, GI/Vascular & Neurologic Baseline
        (
            _cont("systolic_bp", "Baseline systolic blood pressure", 130.0, 16.0, "mmHg"),
            _cont("diastolic_bp", "Baseline diastolic blood pressure", 77.0, 10.0, "mmHg"),
            _cont("heart_rate", "Resting heart rate", 79.0, 12.0, "bpm"),
            _cont("lvef_pct", "Left ventricular ejection fraction", 59.0, 6.8, "%"),
            _cont("urine_protein_creatinine_ratio", "Spot urine protein-to-creatinine ratio", 0.22, 0.24, "mg/mg"),
            _cont("stool_frequency_per_day", "Baseline daily bowel movement frequency", 2.4, 1.5, "per day"),
            _cont("spo2_pct", "Resting oxygen saturation", 97.0, 1.8, "%"),
            _cont("respiratory_rate", "Baseline respiratory rate", 16.5, 2.4, "breaths/min"),
        ),
        # Blocks 8-13 (48): mCRC Sidedness/Staging, Metastatic Sites, GI Biomarkers, RAS/RAF/MSI, Prior Tx, GI Comorbidities
        (
            # Block 8: Primary Location, Pathology & Disease Extent
            _cat("primary_sidedness", "Primary colorectal tumor sidedness", {"Left_colon": 0.44, "Right_colon": 0.30, "Rectum": 0.26}),
            _cat("mucinous_histology", "Mucinous or signet-ring histologic component", {"Conventional_adeno": 0.80, "Mucinous_or_signet": 0.20}),
            _cat("tumor_grade", "Histologic differentiation grade", {"Well": 0.16, "Moderate": 0.58, "Poor": 0.26}),
            _cat("presentation_timing", "Metastatic presentation timing", {"Synchronous": 0.62, "Metachronous": 0.38}),
            _cat("primary_in_situ", "Unresected primary colorectal tumor in situ", {"Resected": 0.58, "In_situ": 0.42}),
            _cont("sum_target_diameters_mm", "RECIST sum of target lesion diameters", 82.0, 44.0, "mm"),
            _cont("liver_met_count", "Number of discrete hepatic metastases", 4.2, 3.1, "lesions"),
            _cont("metastatic_sites_count", "Number of involved metastatic organ systems", 2.1, 1.1, "sites"),
            # Block 9: Organ Distribution & Surgical Resectability
            _cat("liver_limited_disease", "Liver-limited metastatic pattern", {"Non_liver_limited": 0.66, "Liver_limited": 0.34}),
            _cat("peritoneal_carcinomatosis", "Peritoneal carcinomatosis or omental caking", {"Absent": 0.78, "Present": 0.22}),
            _cat("lung_metastases", "Baseline pulmonary metastases", {"Absent": 0.64, "Present": 0.36}),
            _cat("retroperitoneal_nodes", "Distant retroperitoneal/para-aortic nodes", {"Absent": 0.74, "Present": 0.26}),
            _cat("bone_metastases", "Baseline osseous metastases", {"Absent": 0.90, "Present": 0.10}),
            _cat("bowel_obstruction_risk", "Partial colonic obstruction or intraluminal stent", {"Absent": 0.84, "Present": 0.16}),
            _cat("baseline_ascites", "Malignant ascites on staging CT", {"Absent": 0.86, "Present": 0.14}),
            _cat("conversion_resectability", "Multidisciplinary conversion-to-resection potential", {"Unresectable": 0.76, "Borderline_convertible": 0.24}),
            # Block 10: GI Tumor Markers & Pharmacogenomics
            _cont("cea_ng_ml", "Serum carcinoembryonic antigen (CEA)", 68.0, 95.0, "ng/mL"),
            _cont("ca19_9_u_ml", "Serum CA 19-9", 92.0, 120.0, "U/mL"),
            _cat("msi_mmr_status", "Microsatellite instability / MMR status", {"MSS_pMMR": 0.94, "MSI_H_dMMR": 0.06}),
            _cont("tmb_mut_mb", "Tumor mutational burden", 6.2, 4.8, "mut/Mb"),
            _cont("ctdna_vaf_pct", "Circulating tumor DNA max VAF", 6.5, 5.8, "%"),
            _cat("ugt1a1_genotype", "UGT1A1*28 irinotecan pharmacogenotype", {"Wildtype_6_6": 0.46, "Heterozygous_6_7": 0.42, "Homozygous_7_7": 0.12}),
            _cat("dpd_deficiency_variant", "DPYD fluoropyrimidine variant screen", {"Normal": 0.95, "Intermediate": 0.05}),
            _cont("cms_subtype_score", "Consensus molecular subtype mesenchymal score", 0.0, 1.0, "z", positive=False),
            # Block 11: Colorectal Somatic Genomics
            _cat("kras_mutation", "KRAS exon 2/3/4 mutation status", {"Wildtype": 0.56, "Exon2_G12_G13": 0.36, "Exon3_4_or_G12C": 0.08}),
            _cat("nras_mutation", "NRAS somatic mutation status", {"Wildtype": 0.94, "Mutated": 0.06}),
            _cat("braf_v600e", "BRAF V600E alteration status", {"Wildtype": 0.91, "V600E": 0.09}),
            _cat("her2_amplification", "ERBB2/HER2 amplification (IHC/FISH/NGS)", {"Absent": 0.95, "Present": 0.05}),
            _cat("apc_mutation", "APC tumor suppressor mutation", {"Wildtype": 0.24, "Mutated": 0.76}),
            _cat("tp53_mutation", "TP53 somatic alteration status", {"Wildtype": 0.34, "Mutated": 0.66}),
            _cat("pik3ca_mutation", "PIK3CA mutation status", {"Wildtype": 0.82, "Mutated": 0.18}),
            _cat("smad4_loss", "SMAD4 alteration or loss", {"Intact": 0.85, "Lost": 0.15}),
            # Block 12: Prior Surgery, Adjuvant Chemo & Pelvic RT
            _cat("prior_adjuvant_oxaliplatin", "Prior adjuvant oxaliplatin chemotherapy", {"None": 0.66, "Completed_over_12m": 0.22, "Completed_within_12m": 0.12}),
            _cat("prior_adjuvant_5fu", "Prior adjuvant fluoropyrimidine exposure", {"None": 0.62, "Present": 0.38}),
            _cat("prior_pelvic_radiation", "Prior neoadjuvant/pelvic radiotherapy", {"None": 0.78, "Present": 0.22}),
            _cat("prior_colectomy", "Prior primary colorectal surgical resection", {"None": 0.42, "Resected": 0.58}),
            _cat("prior_liver_metastasectomy", "Prior hepatic resection or ablation", {"None": 0.86, "Present": 0.14}),
            _cat("ostomy_present", "Baseline ileostomy or colostomy", {"Absent": 0.82, "Present": 0.18}),
            _cont("months_since_diagnosis", "Months since initial colorectal cancer diagnosis", 11.0, 12.0, "months"),
            _cont("disease_free_interval_mo", "Disease-free interval after curative therapy", 8.0, 10.0, "months"),
            # Block 13: Neuropathy, GI/Vascular Risks & Concomitant Meds
            _cat("baseline_peripheral_neuropathy", "Baseline sensory peripheral neuropathy", {"Grade_0": 0.64, "Grade_1": 0.26, "Grade_2_plus": 0.10}),
            _cat("uncontrolled_hypertension", "Baseline refractory hypertension", {"Absent": 0.72, "Present": 0.28}),
            _cat("recent_major_surgery", "Major abdominal surgery within 6 weeks", {"Absent": 0.88, "Present": 0.12}),
            _cat("arterial_thromboembolism_hx", "Prior MI, ischemic stroke, or arterial clot", {"Absent": 0.88, "Present": 0.12}),
            _cat("anticoagulation", "Therapeutic anticoagulation", {"None": 0.80, "Present": 0.20}),
            _cat("ibd_or_gi_bleed_hx", "Inflammatory bowel disease or recent GI bleed", {"Absent": 0.88, "Present": 0.12}),
            _cat("diabetes_status", "Diabetes mellitus", {"None": 0.70, "Controlled": 0.20, "Uncontrolled": 0.10}),
            _cat("gilbert_syndrome", "Baseline benign unconjugated hyperbilirubinemia", {"Absent": 0.92, "Present": 0.08}),
        ),
    ),
    "breast_er_pos": (
        # Block 2 (8): Vitals, Cardiac QTc & Endocrine Physiology
        (
            _cont("systolic_bp", "Baseline systolic blood pressure", 126.0, 15.0, "mmHg"),
            _cont("diastolic_bp", "Baseline diastolic blood pressure", 76.0, 9.5, "mmHg"),
            _cont("heart_rate", "Resting heart rate", 76.0, 11.0, "bpm"),
            _cont("qtc_interval_ms", "Baseline Fridericia QTc interval (QTcF)", 424.0, 22.0, "ms"),
            _cont("lvef_pct", "Left ventricular ejection fraction", 61.0, 6.0, "%"),
            _cont("estradiol_pg_ml", "Serum estradiol level", 16.0, 14.0, "pg/mL"),
            _cont("bone_mineral_density_tscore", "Lumbar spine DEXA T-score", -1.4, 1.1, "T-score", positive=False),
            _cont("spo2_pct", "Resting oxygen saturation", 97.5, 1.5, "%"),
        ),
        # Blocks 8-13 (48): Breast Histology/Receptors, Metastatic Pattern, Endocrine Biomarkers, Genomics, Prior Tx, Comorbidities
        (
            # Block 8: Histology, Hormone Receptors & Presentation
            _cat("breast_histology", "Breast carcinoma histologic subtype", {"Invasive_ductal": 0.76, "Invasive_lobular": 0.18, "Mixed_or_other": 0.06}),
            _cat("tumor_grade", "Nottingham histologic grade", {"Grade_1": 0.18, "Grade_2": 0.54, "Grade_3": 0.28}),
            _cont("er_expression_pct", "Estrogen receptor nuclear staining percentage", 82.0, 18.0, "%"),
            _cont("pr_expression_pct", "Progesterone receptor nuclear staining percentage", 48.0, 32.0, "%"),
            _cat("her2_ihc_status", "HER2 immunohistochemistry status", {"IHC_0": 0.42, "HER2_low_1plus": 0.38, "HER2_low_2plus_ISH_neg": 0.20}),
            _cat("menopausal_status", "Menopausal status at metastatic index", {"Postmenopausal": 0.78, "Premenopausal_on_OFS": 0.22}),
            _cat("metastatic_presentation", "De novo vs recurrent metastatic presentation", {"De_novo": 0.32, "Recurrent": 0.68}),
            _cont("sum_target_diameters_mm", "RECIST sum of measurable target lesions", 54.0, 34.0, "mm"),
            # Block 9: Organ Metastasis Distribution
            _cat("bone_only_disease", "Bone-only metastatic involvement", {"Non_bone_only": 0.66, "Bone_only": 0.34}),
            _cat("liver_metastases", "Baseline hepatic metastases", {"Absent": 0.70, "Present": 0.30}),
            _cat("lung_metastases", "Baseline pulmonary metastases", {"Absent": 0.68, "Present": 0.32}),
            _cat("brain_metastases", "Baseline CNS/brain metastases", {"Absent": 0.92, "Present": 0.08}),
            _cat("pleural_effusion", "Malignant pleural effusion", {"Absent": 0.84, "Present": 0.16}),
            _cat("soft_tissue_node_mets", "Distant lymph node or chest wall recurrence", {"Absent": 0.62, "Present": 0.38}),
            _cont("metastatic_sites_count", "Number of involved metastatic organ systems", 2.0, 1.1, "sites"),
            _cat("visceral_crisis", "Impending or clinical visceral crisis", {"Absent": 0.90, "Present": 0.10}),
            # Block 10: Breast Tumor Markers & Proliferation Assays
            _cont("ca15_3_u_ml", "Serum CA 15-3 tumor marker", 76.0, 82.0, "U/mL"),
            _cont("ca27_29_u_ml", "Serum CA 27.29 tumor marker", 88.0, 90.0, "U/mL"),
            _cont("cea_ng_ml", "Serum CEA level", 14.0, 20.0, "ng/mL"),
            _cont("ki67_pct", "Tumor Ki-67 proliferation index", 26.0, 16.0, "%"),
            _cont("ctdna_vaf_pct", "Circulating tumor DNA max VAF", 3.6, 3.8, "%"),
            _cat("endocrine_sensitivity", "Prior endocrine therapy sensitivity category", {"De_novo_or_sensitive": 0.62, "Secondary_resistant": 0.28, "Primary_resistant": 0.10}),
            _cont("pam50_luminal_b_score", "Luminal B proliferation signature score", 0.0, 1.0, "z", positive=False),
            _cont("tmb_mut_mb", "Tumor mutational burden", 4.4, 3.2, "mut/Mb"),
            # Block 11: Breast Cancer Somatic & Germline Genomics
            _cat("pik3ca_mutation", "PIK3CA activating mutation", {"Wildtype": 0.62, "Mutated": 0.38}),
            _cat("esr1_mutation", "ESR1 ligand-binding domain mutation", {"Wildtype": 0.86, "Mutated": 0.14}),
            _cat("germline_brca", "Germline BRCA1/BRCA2 pathogenic variant", {"Wildtype": 0.91, "BRCA2": 0.07, "BRCA1": 0.02}),
            _cat("tp53_mutation", "TP53 somatic alteration status", {"Wildtype": 0.72, "Mutated": 0.28}),
            _cat("akt1_pten_alteration", "AKT1 E17K or PTEN loss", {"Absent": 0.88, "Present": 0.12}),
            _cat("rb1_loss", "RB1 loss-of-function alteration", {"Intact": 0.94, "Lost": 0.06}),
            _cat("fgfr1_amplification", "FGFR1 amplification", {"Absent": 0.86, "Present": 0.14}),
            _cat("ccnd1_amplification", "CCND1 / 11q13 amplification", {"Absent": 0.80, "Present": 0.20}),
            # Block 12: Prior Adjuvant Endocrine, Chemo & Surgery
            _cat("prior_adjuvant_ai", "Prior adjuvant aromatase inhibitor", {"None": 0.54, "Completed_over_12m": 0.28, "Relapsed_on_or_within_12m": 0.18}),
            _cat("prior_adjuvant_tamoxifen", "Prior adjuvant tamoxifen exposure", {"None": 0.68, "Present": 0.32}),
            _cat("prior_adjuvant_chemo", "Prior neoadjuvant/adjuvant chemotherapy", {"None": 0.48, "Anthracycline_taxane": 0.36, "Non_anthracycline": 0.16}),
            _cat("prior_breast_surgery", "Prior primary breast surgery", {"None": 0.32, "Lumpectomy": 0.36, "Mastectomy": 0.32}),
            _cat("prior_radiation", "Prior breast/chest wall or bone radiotherapy", {"None": 0.42, "Present": 0.58}),
            _cat("bone_modifying_agent", "Concurrent bisphosphonate or denosumab", {"None": 0.48, "Present": 0.52}),
            _cont("months_since_diagnosis", "Months since initial breast cancer diagnosis", 42.0, 36.0, "months"),
            _cont("disease_free_interval_mo", "Disease-free interval before metastasis", 36.0, 32.0, "months"),
            # Block 13: Cardiac QTc, CYP3A4, Hepatic & Bone Comorbidities
            _cat("qt_prolonging_conmeds", "Concurrent QT-prolonging medications", {"None": 0.76, "Present": 0.24}),
            _cat("arrhythmia_or_cad_hx", "Prior arrhythmia, conduction block, or CAD", {"Absent": 0.84, "Present": 0.16}),
            _cat("cyp3a4_modulator_use", "Concurrent moderate CYP3A4 inhibitor/inducer", {"None": 0.88, "Present": 0.12}),
            _cat("interstitial_lung_disease_hx", "Baseline interstitial lung disease/pneumonitis", {"Absent": 0.94, "Present": 0.06}),
            _cat("baseline_fatty_liver", "Baseline hepatic steatosis / transaminitis risk", {"Absent": 0.76, "Present": 0.24}),
            _cat("vte_history", "Prior venous thromboembolism", {"Absent": 0.88, "Present": 0.12}),
            _cat("diabetes_status", "Diabetes mellitus", {"None": 0.76, "Controlled": 0.17, "Uncontrolled": 0.07}),
            _cat("ppi_use", "Baseline proton pump inhibitor use", {"None": 0.68, "Present": 0.32}),
        ),
    ),
    "mcrpc": (
        # Block 2 (8): Vitals, Cardiometabolic & Endocrine Reserve
        (
            _cont("systolic_bp", "Baseline systolic blood pressure", 134.0, 16.0, "mmHg"),
            _cont("diastolic_bp", "Baseline diastolic blood pressure", 78.0, 10.0, "mmHg"),
            _cont("heart_rate", "Resting heart rate", 76.0, 11.0, "bpm"),
            _cont("lvef_pct", "Left ventricular ejection fraction", 57.0, 8.0, "%"),
            _cont("serum_testosterone_ng_dl", "Castrate serum testosterone", 14.0, 8.0, "ng/dL"),
            _cont("qtc_interval_ms", "Baseline corrected QT interval (QTcF)", 428.0, 22.0, "ms"),
            _cont("bone_mineral_density_tscore", "Hip DEXA bone density T-score", -1.6, 1.1, "T-score", positive=False),
            _cont("spo2_pct", "Resting oxygen saturation", 96.5, 1.9, "%"),
        ),
        # Blocks 8-13 (48): Prostate Pathology/PSA, Skeletal/Visceral Burden, Biomarkers, HRR Genomics, Prior Tx, CNS/CV Comorbidities
        (
            # Block 8: Gleason Grade, PSA Kinetics & Disease Volume
            _cat("gleason_grade_group", "Primary biopsy Gleason Grade Group", {"GG1_GG2": 0.16, "GG3": 0.24, "GG4": 0.26, "GG5": 0.34}),
            _cont("psa_ng_ml", "Baseline serum prostate-specific antigen (PSA)", 42.0, 58.0, "ng/mL"),
            _cont("psa_doubling_time_mo", "PSA doubling time prior to index", 4.8, 3.6, "months"),
            _cat("chaarted_volume", "Metastatic disease volume classification", {"Low_volume": 0.38, "High_volume": 0.62}),
            _cat("de_novo_metastatic", "De novo vs recurrent metastatic history", {"Recurrent": 0.56, "De_novo": 0.44}),
            _cont("bone_scan_lesion_count", "Number of bone metastases on scintigraphy/PSMA", 7.5, 6.0, "lesions"),
            _cont("sum_target_diameters_mm", "RECIST soft-tissue target diameter sum", 44.0, 32.0, "mm"),
            _cont("metastatic_sites_count", "Number of involved metastatic organ systems", 1.8, 0.95, "sites"),
            # Block 9: Skeletal, Nodal & Visceral Involvement
            _cat("bone_only_disease", "Bone-only metastatic pattern", {"Non_bone_only": 0.44, "Bone_only": 0.56}),
            _cat("liver_metastases", "Baseline hepatic metastases", {"Absent": 0.88, "Present": 0.12}),
            _cat("lung_metastases", "Baseline pulmonary metastases", {"Absent": 0.84, "Present": 0.16}),
            _cat("distant_lymph_nodes", "Retroperitoneal or distant nodal metastases", {"Absent": 0.54, "Present": 0.46}),
            _cat("prior_skeletal_event", "Prior symptomatic skeletal-related event (SRE)", {"Absent": 0.74, "Present": 0.26}),
            _cat("urinary_obstruction", "Symptomatic bladder outlet obstruction/hydronephrosis", {"Absent": 0.84, "Present": 0.16}),
            _cat("neuroendocrine_features", "Treatment-emergent neuroendocrine features", {"Absent": 0.93, "Present": 0.07}),
            _cat("opioid_analgesia_use", "Baseline opioid requirement for cancer bone pain", {"None": 0.66, "Intermittent": 0.20, "Scheduled": 0.14}),
            # Block 10: Prostate Biomarkers & Imaging/Liquid Biopsy
            _cont("psma_pet_suv_mean", "PSMA PET/CT mean lesion SUV", 14.5, 7.5, "SUV"),
            _cont("ctc_count_per_7_5ml", "Circulating tumor cells per 7.5 mL blood", 8.0, 12.0, "cells"),
            _cont("ctdna_vaf_pct", "Circulating tumor DNA max VAF", 4.5, 4.8, "%"),
            _cont("chromogranin_a_ng_ml", "Serum chromogranin A", 110.0, 95.0, "ng/mL"),
            _cat("ar_v7_status", "Circulating AR-V7 splice variant status", {"Negative": 0.86, "Positive": 0.14}),
            _cont("tmb_mut_mb", "Tumor mutational burden", 3.6, 2.8, "mut/Mb"),
            _cat("msi_status", "Microsatellite instability status", {"MSS": 0.96, "MSI_High": 0.04}),
            _cont("decipher_genomic_score", "Genomic classifier aggressiveness score", 0.62, 0.22, "score"),
            # Block 11: Prostate HRR & Tumor Suppressor Genomics
            _cat("brca2_alteration", "Germline or somatic BRCA2 alteration", {"Wildtype": 0.88, "Mutated": 0.12}),
            _cat("brca1_atm_hrr", "BRCA1, ATM, CDK12, or PALB2 HRR alteration", {"Wildtype": 0.86, "Mutated": 0.14}),
            _cat("pten_loss", "PTEN deletion or loss", {"Intact": 0.64, "Lost": 0.36}),
            _cat("tp53_mutation", "TP53 somatic alteration status", {"Wildtype": 0.62, "Mutated": 0.38}),
            _cat("rb1_loss", "RB1 loss-of-function alteration", {"Intact": 0.86, "Lost": 0.14}),
            _cat("ar_amplification", "Androgen receptor gene amplification/mutation", {"Absent": 0.64, "Present": 0.36}),
            _cat("spop_mutation", "SPOP somatic mutation", {"Wildtype": 0.89, "Mutated": 0.11}),
            _cat("erg_fusion", "TMPRSS2-ERG gene fusion", {"Absent": 0.56, "Present": 0.44}),
            # Block 12: Prior ADT, Taxane & Local Prostate Therapy
            _cat("prior_docetaxel_mhspc", "Prior docetaxel for metastatic hormone-sensitive disease", {"None": 0.62, "Completed": 0.38}),
            _cat("prior_prostatectomy", "Prior radical prostatectomy", {"None": 0.64, "Present": 0.36}),
            _cat("prior_prostate_rt", "Prior definitive or salvage pelvic radiotherapy", {"None": 0.54, "Present": 0.46}),
            _cat("prior_bicalutamide", "Prior first-generation antiandrogen exposure", {"None": 0.48, "Present": 0.52}),
            _cat("prior_bone_radiation", "Prior palliative bone radiotherapy", {"None": 0.72, "Present": 0.28}),
            _cat("bone_modifying_agent", "Concurrent zoledronic acid or denosumab", {"None": 0.42, "Present": 0.58}),
            _cont("months_on_adt_to_crpc", "Months on ADT before castration resistance", 22.0, 16.0, "months"),
            _cont("months_since_diagnosis", "Months since initial prostate cancer diagnosis", 48.0, 34.0, "months"),
            # Block 13: CNS/Fall Risk vs Mineralocorticoid/CV Comorbidities
            _cat("seizure_or_cns_risk", "Prior seizure, stroke, or seizure-lowering medications", {"Absent": 0.88, "Present": 0.12}),
            _cat("cognitive_fall_risk", "Baseline cognitive impairment or recent falls", {"Absent": 0.80, "Present": 0.20}),
            _cat("heart_failure_history", "Prior congestive heart failure or fluid overload", {"Absent": 0.82, "Present": 0.18}),
            _cat("uncontrolled_hypertension", "Baseline refractory hypertension", {"Absent": 0.66, "Present": 0.34}),
            _cat("coronary_artery_disease", "Coronary artery disease", {"Absent": 0.70, "Present": 0.30}),
            _cat("diabetes_status", "Diabetes mellitus", {"None": 0.64, "Controlled": 0.24, "Uncontrolled": 0.12}),
            _cat("baseline_steroids", "Prior corticosteroid intolerance or baseline steroid use", {"None": 0.86, "Present": 0.14}),
            _cat("anticoagulation", "Therapeutic anticoagulation", {"None": 0.76, "Present": 0.24}),
        ),
    ),
}

ONCOLOGY_CONTEXTS = tuple(ONCOLOGY_CONTEXT_QUESTIONS.keys())


def build_feature_catalog(
    num_features: int, *, clinical_context: str = "nsclc"
) -> List[Dict[str, Any]]:
    """Return ``num_features`` clinical feature definitions for ``clinical_context``."""
    if num_features < 1:
        raise ValueError("num_features must be at least 1")
    if clinical_context not in ONCOLOGY_CONTEXT_QUESTIONS:
        raise ValueError(
            f"unknown clinical_context {clinical_context!r}; choose from {list(ONCOLOGY_CONTEXTS)}"
        )
    if clinical_context == "nsclc":
        base_catalog: Sequence[Dict[str, Any]] = CLINICAL_FEATURE_CATALOG
    else:
        block2, blocks_8_to_13 = _CONTEXT_SPECIFIC_BLOCKS[clinical_context]
        base_catalog = (
            CLINICAL_FEATURE_CATALOG[:8]
            + block2
            + CLINICAL_FEATURE_CATALOG[16:56]
            + blocks_8_to_13
            + CLINICAL_FEATURE_CATALOG[104:120]
        )
    catalog = [dict(item) for item in base_catalog[:num_features]]
    if len(catalog) < num_features:
        for idx in range(len(catalog), num_features):
            panel_idx = idx - len(base_catalog) + 1
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
    clinical_context: str = "nsclc",
    clinical_question: Optional[str] = None,
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
    catalog = build_feature_catalog(num_features, clinical_context=clinical_context)
    resolved_question = (
        clinical_question
        if clinical_question is not None
        else ONCOLOGY_CONTEXT_QUESTIONS[clinical_context]
    )
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
        clinical_question=resolved_question,
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
            "clinical_context": clinical_context,
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
    parser.add_argument(
        "--clinical-context",
        choices=list(ONCOLOGY_CONTEXTS),
        default="nsclc",
        help="Oncology clinical context profile (default: nsclc)",
    )
    parser.add_argument("--clinical-question", default=None)
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
        clinical_context=args.clinical_context,
        clinical_question=args.clinical_question,
        render_text=not args.no_text,
        structured_data_enabled=args.structured_data,
        min_events_per_patient=args.min_events,
        max_events_per_patient=args.max_events,
    )
    print(path)
    return 0
