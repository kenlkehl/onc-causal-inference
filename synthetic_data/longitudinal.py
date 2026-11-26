"""Generate a pretreatment, multimodal cohort for in-context effect estimation."""

import argparse
import fcntl
import json
import os
import re
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import brentq
from scipy.special import expit

from .config import LLMConfig
from .designed_history import (
    candidate_features,
    document_findings,
    measured_values,
    prepare_note_findings,
    sanitized_design_candidates,
)
from .dgp_terms import nonlinear_value, role_lists
from .generator import (
    _compute_logit,
    oracle_dataset_statistics,
    potential_outcome_probabilities,
    scale_equation_coefficients,
)
from .llm_client import LLMClient
from .longitudinal_history import (
    DAYS_PER_YEAR,
    LEGACY_STREAMS,
    STREAMS,
    history_features,
    simulate_history,
)
from .paraphrase_bank import load_paraphrase_bank
from .pro_ctcae import PRO_ATTRIBUTES, PRO_INSTRUMENT, PRO_SOURCE, RECALL_DAYS
from .prompts import CLINICAL_SYSTEM_PROMPT
from .reequation import _features_with_roles, draw_equations, resolve_scenario
from .saved_dgp import NOTE_OUTPUT_FORMAT, _clinical_document, _request
from .study_designs import load_bank, question_text

# Fixed reference means and scales let new patient samples share the exact same DGP.
# Age, weight, etc. appearing elsewhere in a record are not additional oracle inputs.
_FEATURE_SPEC = [
    (
        "latest_ecog",
        "Most recent documented ECOG performance score",
        1.1,
        0.85,
        ["confounder", "effect_modifier"],
    ),
    (
        "log_nlr",
        "Natural log of the latest absolute neutrophil to lymphocyte ratio",
        0.75,
        0.55,
        ["confounder"],
    ),
    (
        "creatinine_slope",
        "First-to-last creatinine change in mg/dL per year",
        0.03,
        0.19,
        ["effect_modifier"],
    ),
    (
        "weight_loss_per_year",
        "Percent of first recorded weight lost per year",
        3.2,
        5.5,
        ["confounder", "effect_modifier"],
    ),
    (
        "recent_ed_visits",
        "Emergency encounters during the 180 days before index",
        0.7,
        0.85,
        ["confounder"],
    ),
    (
        "mean_fatigue_interference",
        "Mean recorded PRO-CTCAE fatigue interference across pretreatment assessments",
        1.4,
        1.0,
        ["confounder", "effect_modifier"],
    ),
]
FEATURES = [
    dict(name=name, description=description, type="continuous", roles=roles)
    for name, description, _, _, roles in _FEATURE_SPEC
]
REFERENCE = {
    name: dict(type="continuous", mean=mean, std=scale) for name, _, mean, scale, _ in _FEATURE_SPEC
}
PROBABILITY_COLUMNS = (
    "true_treatment_prob",
    "true_outcome_prob",
    "true_y0_prob",
    "true_y1_prob",
    "true_ite_prob",
    "true_y0",
    "true_y1",
    "true_ite",
)
NOTE_SEPARATOR = "\n\n<new_note>\n\n"


def _schema(features, include_pros):
    clock = [("day_from_index", pa.int32()), ("delta_years", pa.float64())]

    def stream(*fields):
        return pa.list_(pa.struct([*clock, *fields]))

    result = pa.struct([("component", pa.string()), ("value", pa.float64()), ("unit", pa.string())])
    response = pa.struct(
        [("symptom", pa.string()), ("attribute", pa.string()), ("score", pa.int8())]
    )
    survey = stream(
        ("instrument", pa.string()), ("recall_days", pa.int8()), ("responses", pa.list_(response))
    )
    vital_fields = [
        (name, pa.int16())
        for name in (
            "systolic_mmhg",
            "diastolic_mmhg",
            "pulse_bpm",
            "respirations_per_min",
            "spo2_pct",
        )
    ]
    return pa.schema(
        [
            ("patient_id", pa.int64()),
            ("notes", stream(("text", pa.string()))),
            ("labs", stream(("panel", pa.string()), ("results", pa.list_(result)))),
            (
                "encounters",
                stream(
                    ("setting", pa.string()),
                    ("icd10cm", pa.list_(pa.string())),
                    ("cpt", pa.list_(pa.string())),
                ),
            ),
            (
                "vitals",
                stream(("weight_kg", pa.float64()), ("temperature_c", pa.float64()), *vital_fields),
            ),
            *([("pros", survey)] if include_pros else []),
            ("clinical_text", pa.string()),
            ("treatment_indicator", pa.int8()),
            ("outcome_indicator", pa.int8()),
            *(
                (f"true_{f['name']}", pa.string() if f["type"] == "categorical" else pa.float64())
                for f in features
            ),
            *((name, pa.float64()) for name in PROBABILITY_COLUMNS),
        ]
    )


_SEX_SPECIFIC_CANCERS = re.compile(
    r"\b(prostate|testicular|ovarian|cervical|endometrial|uterine|vulvar|vaginal|penile)\b",
    re.IGNORECASE,
)
_NEMOTRON_3_5_RE = re.compile(r"nemotron[-_ ]?3[._]5(?![0-9])", re.IGNORECASE)


def _is_nemotron_3_5(model_name: str | None) -> bool:
    return bool(model_name and _NEMOTRON_3_5_RE.search(str(model_name)))


def _nemotron_3_5_extra_body(*, enable_thinking: bool = False) -> dict:
    """NVIDIA recommended text sampling profile for Nemotron 3.5 (temp=1.0, top_p=0.95)."""
    return {
        "top_p": 0.95,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
_MAX_TX_INTERACTIONS = 15
_MAX_OUTCOME_INTERACTIONS = 6
_CALIBRATION_DRAWS = 512


def _fast_compute_logit(characteristics, features, summary_stats, equation, treatment=None):
    """O(1)-lookup equivalent of ``synthetic_data.generator._compute_logit``."""
    return _compute_logit(characteristics, features, summary_stats, equation, treatment=treatment)


def _calibration_samples(features, reference, rng, n_draws=_CALIBRATION_DRAWS):
    """Lightweight correlated draws from the feature reference moments for calibration."""
    common = rng.normal(size=n_draws)
    rho = 0.35
    sqrt_rho = float(np.sqrt(rho))
    sqrt_rem = float(np.sqrt(1.0 - rho))
    columns = {}
    for feature in features:
        name = feature["name"]
        stats = reference.get(name, {})
        if feature["type"] == "continuous":
            mean = float(stats.get("mean", 0.0))
            std = float(stats.get("std", 1.0) or 1.0)
            z = sqrt_rho * common + sqrt_rem * rng.normal(size=n_draws)
            columns[name] = mean + std * z
        else:
            probs = stats.get("proportions", {})
            p_present = float(probs.get("present", 0.25))
            p_present = float(np.clip(p_present, 1e-4, 1.0 - 1e-4))
            base_l = float(np.log(p_present / (1.0 - p_present)))
            ps = expit(base_l + 0.25 * common)
            draws = rng.random(n_draws) < ps
            columns[name] = ["present" if flag else "absent" for flag in draws]
    names = [f["name"] for f in features]
    return [{name: columns[name][i] for name in names} for i in range(n_draws)]


def _scale_equation(equation, scale):
    return scale_equation_coefficients(equation, scale)


def _calibrate_intercept(lps, target_rate):
    def error(intercept):
        return float(expit(intercept + lps).mean() - target_rate)

    try:
        return float(brentq(error, -12.0, 12.0))
    except ValueError:
        return -12.0 if error(-12.0) > 0 else 12.0


def _calibrate_longitudinal_equations(features, reference, treatment, outcome, rng):
    """Rescale linear predictors and calibrate intercepts across arbitrary design widths."""
    if len(treatment.get("interactions", ())) > _MAX_TX_INTERACTIONS:
        keep = rng.choice(len(treatment["interactions"]), size=_MAX_TX_INTERACTIONS, replace=False)
        treatment = {
            **treatment,
            "interactions": [treatment["interactions"][int(i)] for i in sorted(keep)],
        }

    # Add modest covariate-by-covariate interactions to outcome_eq for richer SCMs.
    owner_of = {}
    for feature in features:
        if feature["type"] == "continuous":
            owner_of[feature["name"]] = feature["name"]
        else:
            for cat in feature["categories"][1:]:
                owner_of[f"{feature['name']}_{cat}"] = feature["name"]
    out_terms = [t for t in outcome.get("coefficients", {}) if t in owner_of]
    candidate_pairs = [
        (out_terms[i], out_terms[j])
        for i in range(len(out_terms))
        for j in range(i + 1, len(out_terms))
        if owner_of[out_terms[i]] != owner_of[out_terms[j]]
    ]
    if not outcome.get("interactions") and candidate_pairs and rng.random() < 0.65:
        n_pairs = int(rng.integers(1, min(_MAX_OUTCOME_INTERACTIONS, len(candidate_pairs)) + 1))
        chosen = rng.choice(len(candidate_pairs), size=n_pairs, replace=False)
        outcome = {
            **outcome,
            "interactions": [
                {
                    "terms": list(candidate_pairs[int(idx)]),
                    "coefficient": float(rng.uniform(-0.35, 0.35)),
                }
                for idx in sorted(chosen)
            ],
        }

    samples = _calibration_samples(features, reference, rng)
    tx_no_int = {**treatment, "intercept": 0.0}
    out_no_int = {**outcome, "intercept": 0.0}
    tx_lps = np.array(
        [_fast_compute_logit(s, features, reference, tx_no_int, None) for s in samples]
    )
    lp0 = np.array([_fast_compute_logit(s, features, reference, out_no_int, 0) for s in samples])
    lp1 = np.array([_fast_compute_logit(s, features, reference, out_no_int, 1) for s in samples])

    # Occasional unconfounded / RCT-like studies alongside confounded observational ones.
    target_tx_std = 0.0 if rng.random() < 0.08 else float(rng.uniform(0.6, 1.8))
    target_out_std = float(rng.uniform(0.9, 2.0))
    tx_std = float(np.std(tx_lps))
    out_std = max(float(np.std(lp0)), float(np.std(lp1)))
    if tx_std > 1e-8:
        scale = target_tx_std / tx_std
        treatment = _scale_equation(treatment, scale)
        tx_lps = tx_lps * scale
    if out_std > 1e-8:
        scale = target_out_std / out_std
        outcome = _scale_equation(outcome, scale)
        lp0 = lp0 * scale

    target_tx_rate = float(rng.uniform(0.25, 0.75))
    target_ctrl_rate = float(rng.uniform(0.12, 0.82))
    treatment["intercept"] = _calibrate_intercept(tx_lps, target_tx_rate)
    outcome["intercept"] = _calibrate_intercept(lp0, target_ctrl_rate)
    return treatment, outcome


def _validate_note(text, *, allow_sex_terms=False):
    _clinical_document(text)
    # The simulated visit has no sex or gender field for a writer to infer unless
    # the study question itself defines a sex-specific malignancy.
    if not allow_sex_terms and re.search(
        r"\b(male|female|man|woman|he|she|his|her|him|hers)\b", text, re.IGNORECASE
    ):
        raise ValueError("Clinical note introduced an undocumented patient demographic")


def _render_llm_note(
    client, cache_dir, patient_id, note, *, allow_sex_terms=False, realistic_notes=False
):
    pronoun_rule = (
        ""
        if allow_sex_terms
        else "Use 'the patient' throughout because sex and gender are undocumented. "
    )
    if realistic_notes:
        prompt = (
            "Write a realistic, multi-section outpatient medical oncology clinic note "
            "(e.g. Interval History / HPI, Oncology & Past Medical History, Physical Exam, "
            "Assessment & Plan) from the visit facts below.\n"
            "Requirements:\n"
            "- Include EVERY supplied clinical finding and the exact ECOG performance status "
            "score naturally inside the clinical prose (do not drop findings or flip polarity).\n"
            "- Paraphrase naturally using realistic physician wording, synonyms, and standard "
            "oncology abbreviations rather than copying sentences verbatim.\n"
            f"- {pronoun_rule}"
            "Do not invent specific lab numbers, biomarkers, comorbidities, treatments, or "
            "future outcomes beyond the supplied facts.\n"
            "Return just the clinical note.\n\n" + note["text"]
        )
        path = cache_dir / f"patient_{patient_id:06d}" / f"day_{note['day_from_index']}.realistic.json"
        temp = 1.0 if _is_nemotron_3_5(client.config.model_name) else 0.7
        text = _request(
            client,
            path,
            client.config.model_name,
            prompt,
            temp,
            client.config.max_tokens,
            lambda t: _validate_note(t, allow_sex_terms=allow_sex_terms),
            system_prompt=CLINICAL_SYSTEM_PROMPT + NOTE_OUTPUT_FORMAT,
        )
        note["text"] = text.strip()
        return

    # Only the current visit goes to the writer. Keep the numeric ECOG field exact.
    draft, ecog = note["text"].rsplit(" ECOG performance status: ", 1)
    prompt = (
        "Write one short pretreatment clinical note from the visit facts below. "
        "Use only the supplied facts and preserve the findings and their polarity. "
        f"{pronoun_rule}"
        "Do not invent test results, treatments, outcomes, or a numeric performance score. "
        "Return just the clinical note.\n\n" + draft
    )
    path = cache_dir / f"patient_{patient_id:06d}" / f"day_{note['day_from_index']}.json"
    temp = 1.0 if _is_nemotron_3_5(client.config.model_name) else 0.5
    text = _request(
        client,
        path,
        client.config.model_name,
        prompt,
        temp,
        client.config.max_tokens,
        lambda t: _validate_note(t, allow_sex_terms=allow_sex_terms),
        system_prompt=CLINICAL_SYSTEM_PROMPT + NOTE_OUTPUT_FORMAT,
    )
    note["text"] = f"{text.strip()} ECOG performance status: {ecog}"


@dataclass
class LongitudinalDGP:
    features: list
    treatment: dict
    outcome: dict
    schema_version: int = 2
    reference: dict = field(default_factory=lambda: REFERENCE)
    design: dict | None = None
    paraphrase_bank: dict | None = field(default=None, repr=False, compare=False)
    prepared_findings: tuple | None = field(default=None, repr=False, compare=False)
    design_measurements: list | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.design is not None and self.prepared_findings is None:
            measurements, note_findings = sanitized_design_candidates(self.design)
            self.design_measurements = measurements
            self.prepared_findings = prepare_note_findings(
                note_findings, paraphrase_bank=self.paraphrase_bank
            )


def draw_longitudinal_dgp(
    dgp_seed,
    prior=None,
    *,
    schema_version=2,
    design=None,
    paraphrase_bank=None,
    dgp_mode="parametric",
    max_active=None,
):
    """One reproducible study, shared by its context and query patients."""
    if schema_version not in (1, 2):
        raise ValueError(f"Unknown longitudinal schema version: {schema_version}")
    if prior is None:
        prior = "design" if design is not None else "fixed_roles"
    if prior == "design" and design is None:
        raise ValueError("The design prior needs a study design")
    available = [
        f for f in FEATURES if schema_version == 2 or f["name"] != "mean_fatigue_interference"
    ]
    settings = {"name": "longitudinal", "modifier_strength": 3.0, "modifier_main_effects": True}
    reference = REFERENCE
    if design is not None:
        if prior != "design" or schema_version != 2:
            raise ValueError("A study design requires the design prior and five-stream schema")
        roles_rng = np.random.default_rng([dgp_seed, 981])
        if max_active is None:
            available, reference = candidate_features(design, roles_rng)
        else:
            available, reference = candidate_features(design, roles_rng, max_active=max_active)
    if prior in ("mixed", "design"):
        rng = np.random.default_rng([dgp_seed, 923])
        settings.update(
            nonlinearity=str(rng.choice(["none", "threshold", "quadratic"])),
            treatment_effect=float(rng.uniform(-1, 1)),
            modifier_strength=float(rng.uniform(0.5, 4)),
        )
        if prior == "mixed":
            settings["roles"] = {
                "confounder": 1,
                "dual_role": 1,
                "effect_modifier": 1,
                "prognostic": 1,
            }
    elif prior != "fixed_roles":
        raise ValueError(f"Unknown longitudinal prior: {prior}")
    if dgp_mode != "parametric":
        settings["dgp_mode"] = dgp_mode
        settings["nonlinearity"] = "mixed"
        settings["treatment_nonlinearity"] = "mixed"
    scenario = resolve_scenario(settings, available, seed=dgp_seed)
    features = _features_with_roles(available, scenario["roles"])
    treatment, outcome, _, _ = draw_equations(
        features,
        scenario,
        dgp_seed,
        main_coefficient_scale=0.85,
        interaction_coefficient_scale=0.30,
        summary_stats=reference,
    )
    if prior in ("mixed", "design"):
        treatment, outcome = _calibrate_longitudinal_equations(
            features, reference, treatment, outcome, rng
        )
    return LongitudinalDGP(
        features, treatment, outcome, schema_version, reference, design, paraphrase_bank
    )


def sample_longitudinal_patient(
    patient_id,
    *,
    seed,
    dgp,
    note_client=None,
    cache_dir=None,
    omit_absent_prob=0.0,
    style_variation=False,
    realistic_notes=False,
):
    design = dgp.design
    streams = simulate_history(
        seed,
        patient_id,
        include_pros=dgp.schema_version == 2,
        question=design["question"] if design else None,
    )
    if design:
        measurements = (
            dgp.design_measurements
            if dgp.design_measurements is not None
            else design["candidates"]["measurements"]
        )
        features = measured_values(streams, measurements)
        # Document design-specific findings before any LLM note rendering so
        # the LLM weaves the actual causal chart facts into realistic prose.
        features.update(
            document_findings(
                streams["notes"],
                design["candidates"]["note_findings"],
                np.random.default_rng([seed, patient_id, 4]),
                streams=streams,
                omit_absent_prob=omit_absent_prob,
                style_variation=style_variation,
                prepared=dgp.prepared_findings,
                paraphrase_bank=dgp.paraphrase_bank,
            )
        )
    else:
        features = history_features(streams)
    if note_client is not None:
        allow_sex = bool(
            design and _SEX_SPECIFIC_CANCERS.search(design["question"].get("cancer", ""))
        )
        for note in streams["notes"]:
            _render_llm_note(
                note_client,
                cache_dir,
                patient_id,
                note,
                allow_sex_terms=allow_sex,
                realistic_notes=realistic_notes,
            )
    # The label-draw stream is independent of every history and its renderer.
    rng = np.random.default_rng([seed, patient_id, 2])
    propensity = float(
        expit(_fast_compute_logit(features, dgp.features, dgp.reference, dgp.treatment))
    )
    lp0 = _fast_compute_logit(features, dgp.features, dgp.reference, dgp.outcome, 0)
    lp1 = _fast_compute_logit(features, dgp.features, dgp.reference, dgp.outcome, 1)
    p0, p1 = potential_outcome_probabilities(lp0, lp1, dgp.outcome)
    treatment = int(rng.random() < propensity)
    observed_p = p1 if treatment else p0
    return dict(
        patient_id=patient_id,
        **streams,
        clinical_text=NOTE_SEPARATOR.join(note["text"] for note in streams["notes"]),
        treatment_indicator=treatment,
        outcome_indicator=int(rng.random() < observed_p),
        **{f"true_{name}": value for name, value in features.items()},
        true_treatment_prob=propensity,
        true_outcome_prob=observed_p,
        true_y0_prob=p0,
        true_y1_prob=p1,
        true_ite_prob=p1 - p0,
        true_y0=p0,
        true_y1=p1,
        true_ite=p1 - p0,
    )


def generate_longitudinal_cohort(
    output_dir,
    *,
    dataset_size=500,
    seed=42,
    dgp_seed=101,
    llm_config=None,
    num_workers=4,
    prior=None,
    schema_version=2,
    design=None,
    paraphrase_bank=None,
    omit_absent_prob=0.0,
    realistic_notes=False,
):
    """Publish a Parquet cohort with longitudinal observations and the exact DGP."""
    if dataset_size < 1 or num_workers < 1 or seed < 0 or dgp_seed < 0:
        raise ValueError("dataset_size and num_workers must be positive and seeds nonnegative")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory already contains data: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if prior is None:
        prior = "design" if design is not None else "fixed_roles"
    dgp = draw_longitudinal_dgp(
        dgp_seed,
        prior,
        schema_version=schema_version,
        design=design,
        paraphrase_bank=paraphrase_bank,
    )
    streams = STREAMS if schema_version == 2 else LEGACY_STREAMS
    config = dict(
        generation_mode="longitudinal",
        version=schema_version,
        dataset_size=dataset_size,
        outcome_type="binary",
        seed=seed,
        dgp_seed=dgp_seed,
        prior=prior,
        note_style="llm" if llm_config else "offline",
        clinical_question=(
            question_text(design["question"])
            if design
            else "Risk of progression by six months under oncology regimens A and B."
        ),
        note_model=llm_config.model_name if llm_config else None,
    )
    if llm_config:
        config["note_max_tokens"] = llm_config.max_tokens
    if design:
        config["design_id"] = design["id"]
    counts = dict.fromkeys(streams, 0)
    schema = _schema(dgp.features, "pros" in streams)
    with TemporaryDirectory(dir=output_dir.parent, prefix=".longitudinal-") as temporary:
        staging = Path(temporary) / "cohort"
        staging.mkdir()
        with ExitStack() as stack:
            writer = cache_dir = None
            generate_rows = map
            if llm_config:
                cache_dir = output_dir.with_name(output_dir.name + ".note-cache")
                cache_dir.mkdir(exist_ok=True)
                lock = stack.enter_context((cache_dir / "writer.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                writer = stack.enter_context(closing(LLMClient(llm_config)))
                generate_rows = stack.enter_context(ThreadPoolExecutor(max_workers=num_workers)).map
            parquet = stack.enter_context(
                pq.ParquetWriter(staging / "dataset.parquet", schema, compression="zstd")
            )
            make_row = partial(
                sample_longitudinal_patient,
                seed=seed,
                dgp=dgp,
                note_client=writer,
                cache_dir=cache_dir,
                omit_absent_prob=omit_absent_prob,
                realistic_notes=realistic_notes,
            )
            for start in range(0, dataset_size, 64):
                rows = list(generate_rows(make_row, range(start, min(start + 64, dataset_size))))
                parquet.write_table(pa.Table.from_pylist(rows, schema=schema))
                for name in streams:
                    counts[name] += sum(len(row[name]) for row in rows)
        scores = pq.read_table(
            staging / "dataset.parquet",
            columns=["treatment_indicator", "outcome_indicator", *PROBABILITY_COLUMNS],
        )
        stats = oracle_dataset_statistics(scores.to_pandas())
        # A one-patient example has no sample standard deviation.
        stats = {k: (v if np.isfinite(v) else None) for k, v in stats.items()}
        metadata = dict(
            config=config,
            features=dgp.features,
            **role_lists(dgp.features),
            summary_statistics={f["name"]: dgp.reference[f["name"]] for f in dgp.features},
            treatment_equation=dgp.treatment,
            outcome_equation=dgp.outcome,
            dataset_statistics=stats,
            longitudinal=dict(
                streams=list(streams),
                event_counts=counts,
                days_per_year=DAYS_PER_YEAR,
                index_day=0,
                first_delta_years=None,
            ),
        )
        if design:
            metadata["design"] = design
        if "pros" in streams:
            metadata["longitudinal"]["pros"] = dict(
                instrument=PRO_INSTRUMENT,
                library="adult English v1.0",
                recall_days=RECALL_DAYS,
                attributes=PRO_ATTRIBUTES,
                source=PRO_SOURCE,
            )
        for name, value in [("generation_config.json", config), ("metadata.json", metadata)]:
            (staging / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        staging.rename(output_dir)
    return output_dir


def longitudinal_main(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, prog="python -m synthetic_data.cli longitudinal"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42, help="Patient histories and label draws.")
    parser.add_argument(
        "--dgp-seed", type=int, default=101, help="Treatment and outcome equation draw."
    )
    parser.add_argument("--note-style", choices=("offline", "llm"), default="offline")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prior", choices=("fixed_roles", "mixed"), default="fixed_roles")
    source.add_argument(
        "--design-bank", type=Path, help="Materialize one study from this design bank."
    )
    parser.add_argument("--design-id", help="Design to use from the supplied bank.")
    parser.add_argument(
        "--paraphrase-bank",
        type=Path,
        default=None,
        help="Optional LLM paraphrase + EHR-scaffold sidecar for richer offline notes.",
    )
    parser.add_argument(
        "--omit-absent-prob",
        type=float,
        default=0.0,
        help="Probability of omitting absent findings (positive-only charting).",
    )
    parser.add_argument(
        "--realistic-notes",
        action="store_true",
        help="Prompt the LLM for multi-section clinic notes with woven-in ECOG & paraphrases.",
    )
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking=False in extra_body for reasoning-capable vLLM models.",
    )
    parser.add_argument(
        "--schema-version",
        type=int,
        choices=(1, 2),
        default=2,
        help="Version 2 includes PRO-CTCAE. Use 1 to replay an older study.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel LLM note writers.")
    llm = LLMConfig()
    parser.add_argument("--api-url", default=llm.api_base_url)
    parser.add_argument("--model", default=llm.model_name)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=llm.max_tokens,
        help="LLM completion budget per note, including reasoning tokens.",
    )
    args = parser.parse_args(argv)
    if bool(args.design_bank) != bool(args.design_id):
        parser.error("--design-bank and --design-id must be supplied together")
    design = None
    if args.design_bank:
        designs = load_bank(args.design_bank)["designs"]
        design = next((d for d in designs if d["id"] == args.design_id), None)
        if design is None:
            parser.error(f"Unknown design ID: {args.design_id}")
    paraphrase_bank = load_paraphrase_bank(args.paraphrase_bank) if args.paraphrase_bank else None
    llm_config = None
    if args.note_style == "llm":
        if _is_nemotron_3_5(args.model):
            extra_body = _nemotron_3_5_extra_body(enable_thinking=not args.disable_thinking)
            llm_temp = 1.0
        else:
            extra_body = (
                {"chat_template_kwargs": {"enable_thinking": False}}
                if args.disable_thinking
                else None
            )
            llm_temp = llm.temperature
        llm_config = LLMConfig(
            api_base_url=args.api_url,
            model_name=args.model,
            temperature=llm_temp,
            max_tokens=args.max_tokens,
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or "",
            reject_incomplete=True,
            extra_body=extra_body,
        )
    path = generate_longitudinal_cohort(
        args.output_dir,
        dataset_size=args.dataset_size,
        seed=args.seed,
        dgp_seed=args.dgp_seed,
        prior="design" if design else args.prior,
        design=design,
        paraphrase_bank=paraphrase_bank,
        omit_absent_prob=args.omit_absent_prob,
        realistic_notes=args.realistic_notes,
        schema_version=args.schema_version,
        llm_config=llm_config,
        num_workers=args.num_workers,
    )
    print(f"Wrote {args.dataset_size} longitudinal histories to {path / 'dataset.parquet'}")
    return 0
