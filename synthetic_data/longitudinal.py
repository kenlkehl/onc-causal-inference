"""Generate a pretreatment, multimodal cohort for in-context effect estimation."""

import argparse
import fcntl
import json
import os
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit

from .config import LLMConfig
from .dgp_terms import role_lists
from .generator import _compute_logit, oracle_dataset_statistics
from .llm_client import LLMClient
from .longitudinal_history import (
    DAYS_PER_YEAR,
    LEGACY_STREAMS,
    STREAMS,
    history_features,
    simulate_history,
)
from .pro_ctcae import PRO_ATTRIBUTES, PRO_INSTRUMENT, PRO_SOURCE, RECALL_DAYS
from .prompts import CLINICAL_SYSTEM_PROMPT
from .reequation import _features_with_roles, draw_equations, resolve_scenario
from .saved_dgp import NOTE_OUTPUT_FORMAT, _clinical_document, _request

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
            *((f"true_{feature['name']}", pa.float64()) for feature in features),
            *((name, pa.float64()) for name in PROBABILITY_COLUMNS),
        ]
    )


def _validate_note(text):
    _clinical_document(text)
    # The simulated visit has no sex or gender field for a writer to infer.
    if re.search(r"\b(male|female|man|woman|he|she|his|her|him|hers)\b", text, re.IGNORECASE):
        raise ValueError("Clinical note introduced an undocumented patient demographic")


def _render_llm_note(client, cache_dir, patient_id, note):
    # Only the current visit goes to the writer. Keep the numeric ECOG field exact.
    draft, ecog = note["text"].rsplit(" ECOG performance status: ", 1)
    prompt = (
        "Write one short pretreatment clinical note from the visit facts below. "
        "Use only the supplied facts and preserve the findings and their polarity. "
        "Use 'the patient' throughout because sex and gender are undocumented. "
        "Do not invent test results, treatments, outcomes, or a numeric performance score. "
        "Return just the clinical note.\n\n" + draft
    )
    path = cache_dir / f"patient_{patient_id:06d}" / f"day_{note['day_from_index']}.json"
    text = _request(
        client,
        path,
        client.config.model_name,
        prompt,
        0.5,
        client.config.max_tokens,
        _validate_note,
        system_prompt=CLINICAL_SYSTEM_PROMPT + NOTE_OUTPUT_FORMAT,
    )
    note["text"] = f"{text.strip()} ECOG performance status: {ecog}"


@dataclass
class LongitudinalDGP:
    features: list
    treatment: dict
    outcome: dict
    schema_version: int = 2


def draw_longitudinal_dgp(dgp_seed, prior="fixed_roles", *, schema_version=2):
    """One reproducible study, shared by its context and query patients."""
    if schema_version not in (1, 2):
        raise ValueError(f"Unknown longitudinal schema version: {schema_version}")
    available = [
        f for f in FEATURES if schema_version == 2 or f["name"] != "mean_fatigue_interference"
    ]
    settings = {"name": "longitudinal", "modifier_strength": 3.0, "modifier_main_effects": True}
    if prior == "mixed":
        rng = np.random.default_rng([dgp_seed, 923])
        settings.update(
            roles={"confounder": 1, "dual_role": 1, "effect_modifier": 1, "prognostic": 1},
            nonlinearity=str(rng.choice(["none", "threshold", "quadratic"])),
            treatment_effect=float(rng.uniform(-1, 1)),
            modifier_strength=float(rng.uniform(0.5, 4)),
        )
    elif prior != "fixed_roles":
        raise ValueError(f"Unknown longitudinal prior: {prior}")
    scenario = resolve_scenario(settings, available, seed=dgp_seed)
    features = _features_with_roles(available, scenario["roles"])
    treatment, outcome, _, _ = draw_equations(
        features,
        scenario,
        dgp_seed,
        main_coefficient_scale=0.85,
        interaction_coefficient_scale=0.30,
    )
    if prior == "mixed":
        treatment["intercept"], outcome["intercept"] = map(float, rng.uniform(-1, 1, 2))
    return LongitudinalDGP(features, treatment, outcome, schema_version)


def sample_longitudinal_patient(patient_id, *, seed, dgp, note_client=None, cache_dir=None):
    streams = simulate_history(seed, patient_id, include_pros=dgp.schema_version == 2)
    features = history_features(streams)
    if note_client is not None:
        for note in streams["notes"]:
            _render_llm_note(note_client, cache_dir, patient_id, note)
    # The label-draw stream is independent of every history and its renderer.
    rng = np.random.default_rng([seed, patient_id, 2])
    propensity = float(expit(_compute_logit(features, dgp.features, REFERENCE, dgp.treatment)))
    p0, p1 = [
        float(expit(_compute_logit(features, dgp.features, REFERENCE, dgp.outcome, t)))
        for t in (0, 1)
    ]
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
    prior="fixed_roles",
    schema_version=2,
):
    """Publish a Parquet cohort with longitudinal observations and the exact DGP."""
    if dataset_size < 1 or num_workers < 1 or seed < 0 or dgp_seed < 0:
        raise ValueError("dataset_size and num_workers must be positive and seeds nonnegative")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory already contains data: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dgp = draw_longitudinal_dgp(dgp_seed, prior, schema_version=schema_version)
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
        clinical_question="Risk of progression by six months under oncology regimens A and B.",
        note_model=llm_config.model_name if llm_config else None,
    )
    if llm_config:
        config["note_max_tokens"] = llm_config.max_tokens
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
            summary_statistics={f["name"]: REFERENCE[f["name"]] for f in dgp.features},
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
    parser.add_argument("--prior", choices=("fixed_roles", "mixed"), default="fixed_roles")
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
    llm_config = None
    if args.note_style == "llm":
        llm_config = LLMConfig(
            api_base_url=args.api_url,
            model_name=args.model,
            max_tokens=args.max_tokens,
            api_key=os.getenv("OPENAI_API_KEY", ""),
            reject_incomplete=True,
        )
    path = generate_longitudinal_cohort(
        args.output_dir,
        dataset_size=args.dataset_size,
        seed=args.seed,
        dgp_seed=args.dgp_seed,
        prior=args.prior,
        schema_version=args.schema_version,
        llm_config=llm_config,
        num_workers=args.num_workers,
    )
    print(f"Wrote {args.dataset_size} longitudinal histories to {path / 'dataset.parquet'}")
    return 0
