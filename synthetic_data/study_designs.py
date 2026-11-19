"""Author, validate and freeze reusable clinical-question designs for longitudinal studies."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import fcntl
import json
import math
import os
from pathlib import Path
import re
import random
from functools import partial
from itertools import product

from .config import LLMConfig
from .dgp_terms import ROLE_GROUPS
from .generator import _RESERVED_ORACLE_COLUMNS
from .llm_client import LLMClient
from .pro_ctcae import PRO_ITEMS
from .saved_dgp import _checkpoint, _digest, _read_checkpoint, _write_json

DESIGN_DIR = Path(__file__).with_name("designs")
DEFAULT_BANK = DESIGN_DIR / "starter.json"
# Fixed simulation units for z-scoring across studies, independent of query patients.
LAB_REFERENCE = {
    "platelets": (250, 35),
    "hemoglobin": (13, 1),
    "wbc": (7.2, 1.5),
    "neutrophils_abs": (4.2, 1.5),
    "lymphocytes_abs": (2, 0.6),
    "creatinine": (1, 0.28),
    "sodium": (139, 2),
    "potassium": (4.2, 0.3),
    "bun": (17, 3),
    "glucose": (99, 13),
}
VITAL_REFERENCE = {
    "weight_kg": (80, 18),
    "systolic_mmhg": (124, 10),
    "pulse_bpm": (78, 8),
    "spo2_pct": (97, 1.8),
    "respirations_per_min": (16, 2),
}
MEASUREMENTS = {
    "latest_ecog": (1.1, 0.85),
    "recent_ed_visits": (0.7, 0.85),
    "log_nlr": (0.75, 0.55),
    "creatinine_slope": (0.03, 0.19),
    "weight_loss_per_year": (3.2, 5.5),
}
for source, fields in (("lab", LAB_REFERENCE), ("vital", VITAL_REFERENCE)):
    for field, (center, scale) in fields.items():
        for summary in ("last", "mean", "change"):
            MEASUREMENTS[f"{source}.{field}.{summary}"] = (
                0 if summary == "change" else center,
                scale,
            )
for symptom, attribute in PRO_ITEMS:
    MEASUREMENTS[f"pro.{symptom}.{attribute}.mean"] = (1.25, 1)

SPLITS = ("train", "held_out")
ROLES = (*ROLE_GROUPS, "unused")
ID = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
# Findings are sampled independently of the existing streams. Measured physiology
# must use an executable measurement, never a second independently sampled value.
MEASURED_IN_NOTE = re.compile(
    r"\b(ecog|performance status|blood pressure|pulse oximetry|oxygen saturation|"
    r"heart rate|body weight|blood counts?|liver (?:function tests?|enzymes)|"
    r"platelets?|hemoglobin|haemoglobin|neutrophils?|lymphocytes?|creatinine|"
    r"sodium|potassium|bilirubin|albumin|ast|alt|wbc|bun|glucose|pro[- ]ctcae)\b",
    re.I,
)
STUDY_LABEL_IN_NOTE = re.compile(
    r"\b(confounder|effect modifier|study arm|assigned to|randomized to|will receive|"
    r"counterfactual|treatment_indicator|outcome_indicator|true_y[01])\b",
    re.I,
)
QUESTION_KEYS = (
    "id",
    "split",
    "cancer",
    "population",
    "icd10cm",
    "control",
    "treatment",
    "outcome",
    "horizon_days",
)


def question_text(q):
    return (
        f"In {q['population']}, what is the risk of {q['outcome']} by day "
        f"{q['horizon_days']} under {q['treatment']} versus {q['control']}?"
    )


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_question(q):
    if not isinstance(q, dict) or set(q) != set(QUESTION_KEYS):
        raise ValueError(f"Expected question fields {QUESTION_KEYS}")
    if not all(_text(q[k]) for k in QUESTION_KEYS if k != "horizon_days"):
        raise ValueError("Question fields must contain text")
    if not ID.fullmatch(q["id"]) or q["split"] not in SPLITS:
        raise ValueError("Invalid question ID or split")
    if type(q["horizon_days"]) is not int or q["horizon_days"] <= 0:
        raise ValueError("Binary endpoint requires a positive integer horizon_days")
    if q["control"].strip().casefold() == q["treatment"].strip().casefold():
        raise ValueError("Study arms must differ")


def _check_split(questions):
    seen = {}
    for q in questions:
        arms = tuple(sorted(s.strip().casefold() for s in (q["control"], q["treatment"])))
        identity = (
            q["population"].strip().casefold(),
            arms,
            q["outcome"].strip().casefold(),
            q["horizon_days"],
        )
        if seen.setdefault(identity, q["split"]) != q["split"]:
            raise ValueError("Same clinical question occurs in different splits")


def validate_candidates(body):
    if not isinstance(body, dict) or set(body) != {"measurements", "note_findings"}:
        raise ValueError("Expected measurements and note_findings arrays")
    seen = set()
    for kind, candidates in body.items():
        if not isinstance(candidates, list):
            raise ValueError(f"{kind} must be an array")
        for item in candidates:
            keys = {"name", "roles"}
            if kind == "note_findings":
                keys.update(("prevalence", "present", "absent"))
            if not isinstance(item, dict) or set(item) != keys:
                raise ValueError(f"Expected {kind} item fields {sorted(keys)}")
            name, roles = item["name"], item["roles"]
            if (
                not _text(name)
                or name.replace(".", "_") in seen
                or f"true_{name}" in _RESERVED_ORACLE_COLUMNS
            ):
                raise ValueError(f"Invalid or duplicate candidate {name}")
            seen.add(name.replace(".", "_"))
            if (
                not isinstance(roles, list)
                or not roles
                or any(not isinstance(r, str) or r not in ROLES for r in roles)
                or len(roles) != len(set(roles))
            ):
                raise ValueError(f"{name} needs distinct roles from {ROLES}")
            if kind == "measurements":
                if name not in MEASUREMENTS:
                    raise ValueError(f"Unsupported measurement {name}")
            else:
                if not ID.fullmatch(name) or name in MEASUREMENTS:
                    raise ValueError(f"Invalid note finding {name}")
                p = item["prevalence"]
                if type(p) not in (int, float) or not math.isfinite(p) or not 0 < p < 1:
                    raise ValueError(f"{name} prevalence must lie between zero and one")
                if (
                    not all(_text(item[k]) for k in ("present", "absent"))
                    or item["present"].strip().casefold() == item["absent"].strip().casefold()
                ):
                    raise ValueError(
                        f"{name} needs distinct positive and negative chart statements"
                    )
                if any(
                    MEASURED_IN_NOTE.search(text)
                    for text in (name.replace("_", " "), item["present"], item["absent"])
                ):
                    raise ValueError(
                        f"{name} duplicates a separately simulated measurement; choose a distinct chart finding"
                    )
                if any(STUDY_LABEL_IN_NOTE.search(item[k]) for k in ("present", "absent")):
                    raise ValueError(
                        f"{name} chart sentences must exclude study labels and assignment"
                    )
    if not seen or not any(set(f["roles"]) - {"unused"} for items in body.values() for f in items):
        raise ValueError("Design needs at least one candidate with a causal role")


def validate_bank(bank):
    if (
        not isinstance(bank, dict)
        or bank.get("version") != 1
        or not isinstance(bank.get("designs"), list)
        or not bank["designs"]
    ):
        raise ValueError("Expected a version 1 study design bank with designs")
    ids, families = set(), {}
    for d in bank["designs"]:
        if (
            not isinstance(d, dict)
            or not isinstance(d.get("id"), str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,119}", d["id"])
            or d["id"] in ids
        ):
            raise ValueError("Invalid or duplicate design ID")
        ids.add(d["id"])
        q = d.get("question")
        validate_question(q)
        if families.setdefault(q["id"], q) != q:
            raise ValueError("Variants of a question must share the same question and split")
        validate_candidates(d.get("candidates"))
    _check_split(families.values())
    return bank


def load_bank(path):
    return validate_bank(json.loads(Path(path).read_text()))


def design_prompt(q, variant_seed, measurements=MEASUREMENTS):
    return f"""Design a synthetic comparative oncology study. Question: {question_text(q)}
Design variation seed: {variant_seed}. Choose a clinically distinct set of candidate pretreatment variables.
Return one JSON object with two arrays:
"measurements": 2-5 objects with "name" copied exactly from the list below and "roles".
"note_findings": 2-4 NEW cancer-specific binary characteristics (e.g. a pathology finding, prior exposure or comorbidity). Each object has "name" (unique lowercase snake_case), "prevalence" (probability), "present", "absent" (one natural pretreatment chart sentence each showing the corresponding state), and "roles".
Use pathology, imaging, molecular findings, anatomic involvement, surgery history or prior exposures. Both sentences should express actual presence/absence, without invented additional facts. Exclude study-arm assignment, causal roles, predicted outcomes, future events, and dates relative to enrollment. No lab results, vital signs, weight, ECOG, questionnaire items, or independently sampled measurements in either chart sentence.
"roles" is an array of plausible alternative roles this variable could play in different studies. Allowed: confounder, effect_modifier, dual_role, prognostic, instrument, unused. dual_role means both confounding and effect modification. Include some alternative roles and some candidates that could be unused; include candidates for both confounding and modification.
Available measurement names: {', '.join(measurements)}"""


def _author_design(task, *, seed, client, cache):
    q, i = task
    name = f"{q['id']}_v{i + 1}"
    variant_seed = f"{seed}/{q['id']}/{i}"
    measurements = list(MEASUREMENTS)
    if i:
        # Stratified menus counter the tendency to choose the same handful of
        # common labs for every question while preserving a clinical choice.
        rng = random.Random(variant_seed)
        measurements = []
        for prefix, n in (("lab.", 10), ("vital.", 6), ("pro.", 5)):
            pool = [s for s in MEASUREMENTS if s.startswith(prefix)]
            measurements.extend(rng.sample(pool, n))
        measurements.extend(rng.sample([s for s in MEASUREMENTS if "." not in s], 2))
        if not any(s.endswith(".change") for s in measurements):
            measurements.append(rng.choice([s for s in MEASUREMENTS if s.endswith(".change")]))
    prompt = design_prompt(q, variant_seed, measurements)
    if i:
        emphases = (
            "changes in measured physiology and patient-reported symptoms. Include a .change measurement and a pro. measurement from the offered menu",
            "prior exposures and delayed toxicity",
            "serial blood counts and evolving symptoms",
            "functional reserve and utilization",
            "radiologic burden and organ involvement",
            "surgical history and nutritional status",
        )
        prompt += (
            f"\nFor this variant, emphasize {emphases[(i - 1) % len(emphases)]} "
            "where clinically plausible; retain the specified population and comparison."
        )
    identity = _digest([client.config.model_name, prompt])
    path = cache / f"{name}.json"
    body = _read_checkpoint(path, identity)
    if body is None:
        request = prompt
        for attempt in range(3):
            body = client.generate_json(request, temperature=0.7)
            try:
                validate_candidates(body)
                chosen = {s["name"] for s in body["measurements"]}
                if chosen.difference(measurements):
                    raise ValueError("Select measurements only from this variant's offered names")
                if i % 6 == 1 and not (
                    any(k.startswith("pro.") for k in chosen)
                    and any(k.endswith(".change") for k in chosen)
                ):
                    raise ValueError(
                        "This longitudinal symptom variant needs a pro. item and a .change measurement"
                    )
                break
            except ValueError as exc:
                if attempt == 2:
                    raise
                request = (
                    f"{prompt}\nCorrect this validation error: {exc}. Return the complete JSON."
                )
        _checkpoint(path, identity, body)
    validate_candidates(body)
    return dict(id=name, question=q, candidates=body)


def build_bank(questions, variants, seed, client, output, *, workers=1):
    if not isinstance(questions, list) or not questions or min(variants, workers) < 1 or seed < 0:
        raise ValueError(
            "Provide seed questions, positive variants and workers, and a nonnegative seed"
        )
    for q in questions:
        validate_question(q)
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question ID")
    _check_split(questions)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.with_name(output.name + ".requests")
    cache.mkdir(exist_ok=True)
    with (cache / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generation = dict(
            model=client.config.model_name,
            seed=seed,
            variants=variants,
            prompt_version=2,
            questions_sha256=_digest(questions),
        )
        frozen = {}
        if output.exists():
            bank = load_bank(output)
            prior = bank.get("generation", {})
            if prior == generation:
                return bank
            if (
                prior.get("variants", variants) >= variants
                or {**prior, "variants": variants} != generation
            ):
                raise FileExistsError(f"Different design bank already exists: {output}")
            frozen = {d["id"]: d for d in bank["designs"]}
        author = partial(_author_design, seed=seed, client=client, cache=cache)
        tasks = [
            (q, i)
            for q, i in product(questions, range(variants))
            if f"{q['id']}_v{i + 1}" not in frozen
        ]
        if workers == 1:
            designs = list(map(author, tasks))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                designs = list(pool.map(author, tasks))
        frozen.update((d["id"], d) for d in designs)
        ordered = [frozen[f"{q['id']}_v{i+1}"] for q, i in product(questions, range(variants))]
        bank = validate_bank(dict(version=1, generation=generation, designs=ordered))
        _write_json(output, bank)
    return bank


def designs_main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DESIGN_DIR / "questions.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1, help="Parallel design requests.")
    parser.add_argument("--variants-per-question", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    defaults = LLMConfig()
    parser.add_argument("--api-url", default=defaults.api_base_url)
    parser.add_argument("--model", default=defaults.model_name)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args(argv)
    config = LLMConfig(
        api_base_url=args.api_url,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name=args.model,
        max_tokens=args.max_tokens,
        reject_incomplete=True,
    )
    with closing(LLMClient(config)) as client:
        bank = build_bank(
            json.loads(args.questions.read_text()),
            args.variants_per_question,
            args.seed,
            client,
            args.output,
            workers=args.workers,
        )
    print(f"Wrote {len(bank['designs'])} designs to {args.output}")
    return 0
