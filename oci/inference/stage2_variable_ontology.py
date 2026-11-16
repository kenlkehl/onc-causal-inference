"""Closed-format variable ontologies: define or decompose, then harmonized merging.

Stage 2 defines each discovered candidate before any semantic merging. One
``04_define_or_decompose`` response defines the candidate as one variable or
decomposes it into 2 to 12 component variables. Every variable has a closed
answer format: a number in one unit, or exactly one label from a fixed list of
mutually exclusive labels, each label defined by its clinical meaning and the
documented statements that establish it. Round-based ``03_merge_and_harmonize``
reviews then group variables that measure the same attribute and map every
member label onto one shared ontology through an explicit crosswalk.

The checks in this module are domain-agnostic shape rules applied to model
replies. They never read patient data, outcomes, or oracle columns. Each
validator returns a normalized reply that validates to itself, so checkpointed
replies are revalidated unchanged on resume.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping, Sequence

from . import stage2_clinical_prompts as clinical_prompts
from .stage2_documented_unknown import DOCUMENTED_UNKNOWN_RULE, is_unknown_like

ONTOLOGY_DEFINITION_SCHEMA_VERSION = "stage2_define_or_decompose_v3_compound_unknown_labels"
HARMONIZATION_SCHEMA_VERSION = "stage2_harmonized_merge_v2_compound_unknown_labels"
VALUE_TYPES = ("continuous", "binary", "categorical", "ordinal")
LABELED_VALUE_TYPES = frozenset({"binary", "categorical", "ordinal"})
STRATEGIES = ("latest", "earliest", "maximum", "minimum", "mode", "any_positive", "single_or_null")
MAX_COMPONENTS = 12
# A validation message lists at most this many problems; the model repairs the
# rest after the next reply.
_MAX_REPORTED_PROBLEMS = 12
_DEFINITION_FALLBACK_ACTION = "no_valid_definition"

_STRATEGY_ALIASES = {
    "most_recent": "latest", "last": "latest", "first": "earliest", "max": "maximum",
    "min": "minimum", "majority": "mode", "most_frequent": "mode",
}
# A label that only says the record lacks the value duplicates null.
_MISSING_LABEL = re.compile(
    r"^(not (stated|documented|mentioned|reported|recorded|specified|known|evaluated)"
    r"|none (documented|stated|reported|recorded)|no (documentation|data|information)"
    r"|unspecified|missing|n/?a)$", re.I)
# A label made only of unknown-type words (optionally with residual or filler
# words) duplicates the reserved documented_unknown answer.
_UNKNOWN_WORDS = re.compile(
    r"\b(unknown|undetermined|indeterminate|pending|not assessed|not evaluated|not determined|"
    r"not assessable|not evaluable|equivocal|inconclusive|uninformative|"
    r"cannot be (determined|assessed)|could not be determined|"
    r"un(stag|grad|assess|evaluat|typ|measur)(ed|eable|able))\b", re.I)
# Words that say a value is missing; they may accompany an unknown-type word
# ("Not assessed/N/A", "unknown_not_stated") without adding an answer.
_MISSING_WORDS = re.compile(
    r"\b(n a|na|not applicable|not (stated|documented|mentioned|reported|recorded|specified|available))\b", re.I)
_FILLER_WORDS = re.compile(
    r"\b(other|unspecified|not specified|not otherwise specified|nos|unclassified|or|and|"
    r"documented|status|value|result|type)\b", re.I)
# "Not tested" in a result variable is an unknown-type answer; a binary
# performed/not-performed variable may use it as its negative label.
_NOT_DONE = re.compile(
    r"\bnot\s+(tested|performed|done|measured|obtained|checked|sent|analy[sz]ed|available|staged|"
    r"graded|typed)\b|\buntested\b", re.I)
_TEST_WORDS = re.compile(r"\b(test|testing|tests|assessment|evaluation|analysis|sample|specimen|yet)\b", re.I)
# A continuous unit that names several scales.
_COMPOSITE_UNIT = re.compile(r"\bor\b|\beither\b|instrument|specific|varies|various|depend", re.I)
# A continuous variable whose number comes from any of several instruments or
# scales measures different quantities, unless a stated conversion maps them.
_SEVERAL_SCALES = re.compile(
    r"\b(any|various|multiple|several|different|other|validated|standardized|available)\b"
    r"(?:\W+\w+){0,4}?\W+(instruments|scales|scoring systems|questionnaires|tools)\b", re.I)
_CONVERSION = re.compile(r"\bconver(t|ts|ted|sion)\b", re.I)
_SCORE_WORDS = re.compile(r"\b(score|grade|scale|rating|severity|class)\b", re.I)
# A stated range with at most six levels (0-4, 1-5, 2/4) marks a graded score;
# units, decimals, and durations exclude measured quantities.
_SMALL_RANGE = re.compile(
    r"\b[01]\s*(?:-|–|—|to)\s*[2-5]\b(?!\s*(?:\.\d|%|mg|g/|cm|mm|kg|decimals?|digits?|days?|weeks?|"
    r"months?|years?|hours?))|(?<![\d.])[0-5]\s*/\s*[2-5]\b(?![\d.])")
# The widest stated range decides: a 0-100 score that mentions raw 0-4 responses stays continuous.
_WIDE_RANGE = re.compile(r"\b\d+\s*(?:-|–|—|to)\s*(\d{2,})\b|/\s*(\d{2,})\b")
_RESIDUAL = re.compile(r"\bother\b|unspecified|not otherwise specified|\bnos\b|unclassified", re.I)
# A meaning that only says how the record words the label is a lookup, not a clinical state.
_LOOKUP = re.compile(
    r"^(explicitly\s+)?(documented|recorded|reported|stated|noted|listed|charted)\s+(as|with)\b"
    r"|^(the\s+)?(record|documentation|text|note)s?\s+(explicitly\s+)?(states|documents|lists|reports)\b",
    re.I)
_STOP_WORDS = frozenset({
    "of", "the", "a", "an", "and", "or", "with", "without", "in", "on", "for", "to", "by", "type",
    "types", "subtype", "subtypes", "category", "class", "form", "status", "result", "value", "label",
})


def canonical_strategy(value: Any) -> str | None:
    """A named repeated-observation rule, from its name or a name it starts with."""

    key = snake_case(value)
    key = _STRATEGY_ALIASES.get(key, key)
    if key in STRATEGIES:
        return key
    for prefix, strategy in (("latest_", "latest"), ("most_recent", "latest"), ("earliest_", "earliest"),
                             ("first_", "earliest"), ("maximum_", "maximum"), ("minimum_", "minimum")):
        if key.startswith(prefix):
            return strategy
    return None


def _member_rule(raw: Mapping[str, Any], variables: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """The members' shared repeated-observation rule (positive label mapped), else latest.

    Used when a merge reply names no recognized rule: the prompt asks for the
    members' rule, and latest when they differ.
    """

    rules = {(variables[m]["conflict_resolution"]["strategy"]) for m in variables}
    if len(rules) != 1:
        return {"strategy": "latest", "positive_category": None}
    [strategy] = rules
    if strategy != "any_positive":
        return {"strategy": strategy, "positive_category": None}
    targets = set()
    for row in raw.get("category_crosswalk") or []:
        if not isinstance(row, Mapping) or row.get("member") not in variables:
            continue
        member = variables[row["member"]]
        if _resolve(row.get("member_label"), member["categories"]) == member["conflict_resolution"]["positive_category"]:
            targets.add(str(row.get("shared_label")))
    if len(targets) == 1:
        return {"strategy": "any_positive", "positive_category": targets.pop()}
    return {"strategy": "latest", "positive_category": None}


def snake_case(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _plain(label: Any) -> str:
    return re.sub(r"[_/,()\-]+", " ", str(label).lower())


def is_unknown_label(label: str) -> bool:
    """A label that names only an unknown or undetermined state."""

    if is_unknown_like(label):
        return True
    text = _plain(label)
    if not _UNKNOWN_WORDS.search(text):
        return False
    rest = _MISSING_WORDS.sub(" ", _UNKNOWN_WORDS.sub(" ", text))
    return not _FILLER_WORDS.sub(" ", rest).strip()


def is_not_done_label(label: str) -> bool:
    """A label that says only that a test or assessment was not done."""

    text = _plain(label)
    if not _NOT_DONE.search(text):
        return False
    rest = _TEST_WORDS.sub(" ", _UNKNOWN_WORDS.sub(" ", _NOT_DONE.sub(" ", text)))
    return not _FILLER_WORDS.sub(" ", rest).strip()


def is_residual_label(label: str) -> bool:
    return bool(_RESIDUAL.search(str(label).replace("_", " ")))


def _folded(label: Any) -> str:
    return re.sub(r"[\s_\-]+", "", str(label)).casefold()


def is_sublevel(parent: str, child: str) -> bool:
    """True when ``child`` is ``parent`` plus a sublevel suffix (M1 -> M1a, IV -> IVB, 3 -> 3A)."""

    a, b = _folded(parent), _folded(child)
    if not a or not b.startswith(a) or not re.fullmatch(r"[a-d]\d?", b[len(a):]):
        return False
    return bool(re.search(r"(\d|(?<![a-z])[ivx]+)$", str(parent).casefold().replace("_", " ").strip()))


def _stems(text: Any) -> set[str]:
    # A four-character stem lets mutant match mutation and tumours match tumor.
    return {word[:4] for word in re.findall(r"[a-z0-9]+", str(text).casefold().replace("_", " "))
            if word not in _STOP_WORDS}


def _residual_scope(label: str) -> set[str]:
    """The words of a residual label beyond its residual words."""

    return _stems(_RESIDUAL.sub(" ", str(label).replace("_", " ")))


def answer_format(variable: Mapping[str, Any]) -> str:
    if variable["value_type"] == "continuous":
        return f"continuous ({variable.get('unit') or 'unitless'})"
    return f"{variable['value_type']} ({'; '.join(variable['categories'])})"


def _text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip() if isinstance(value, (str, list, tuple)) else ""


def _resolve(label: Any, labels: Sequence[str]) -> str | None:
    """The one label equal to ``label``, ignoring case and surrounding space."""

    if not isinstance(label, str):
        return None
    if label in labels:
        return label
    matches = [item for item in labels if item.casefold() == label.strip().casefold()]
    return matches[0] if len(matches) == 1 else None


def normalize_label_definitions(
    raw: Any, categories: Sequence[str], where: str, *, ignored_labels: Sequence[str] = (),
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Return ({label: {meaning, established_by}}, problems) keyed exactly by ``categories``.

    ``ignored_labels`` may appear as keys and are dropped (labels that Python
    already removed, such as unknown-like labels that became documented_unknown).
    """

    if not isinstance(raw, Mapping):
        return {}, [f"{where} needs label_definitions (an object keyed by its labels)"]
    resolved: dict[str, Any] = {}
    for key, item in raw.items():
        label = _resolve(key, categories)
        if label is None and _resolve(key, ignored_labels) is not None:
            continue
        if label is None or label in resolved:
            return {}, [f"{where} label_definitions keys must be exactly its labels"]
        resolved[label] = item
    if set(resolved) != set(categories):
        return {}, [f"{where} label_definitions keys must be exactly its labels"]
    problems: list[str] = []
    definitions: dict[str, dict[str, str]] = {}
    for category in categories:
        item = resolved[category]
        meaning = _text(item.get("meaning")) if isinstance(item, Mapping) else ""
        established = _text(item.get("established_by")) if isinstance(item, Mapping) else ""
        if not meaning or not established:
            problems.append(f"{where} label {category!r} needs nonempty meaning and established_by")
        elif _LOOKUP.search(meaning):
            problems.append(
                f"{where} label {category!r}: meaning repeats how a record words the label; state the "
                "clinical state the label stands for and keep record wording in established_by")
        definitions[category] = {"meaning": meaning, "established_by": established}
    return definitions, problems


def normalize_variable(
    raw: Any, where: str, *, require_label_definitions: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Return (normalized variable, problems) for one closed-format definition."""

    if not isinstance(raw, Mapping):
        return {}, [f"{where} must be an object"]
    required = ("name", "description", "value_type", "unit", "categories", "measurement_definition",
                "missing_value_rule", "conflict_resolution")
    missing = [key for key in required if key not in raw]
    if require_label_definitions and "label_definitions" not in raw:
        missing.append("label_definitions")
    if missing:
        return {}, [f"{where} is missing {', '.join(missing)}"]
    problems: list[str] = []
    name = snake_case(raw["name"])
    if not name:
        problems.append(f"{where} needs a name")
    value_type = str(raw["value_type"] or "").strip().lower()
    if value_type not in VALUE_TYPES:
        problems.append(f"{where} value_type {raw['value_type']!r} is not one of continuous, binary, "
                        "categorical, ordinal; there is no free-text or ambiguous type")
    raw_categories = raw["categories"]
    if not isinstance(raw_categories, list) or not all(
            isinstance(item, (str, int, float)) and not isinstance(item, bool) and str(item).strip()
            for item in raw_categories):
        problems.append(f"{where} categories must be an array of labels")
        raw_categories = []
    categories = [str(item).strip() for item in raw_categories]
    unit = raw["unit"]
    if unit is not None and not isinstance(unit, str):
        problems.append(f"{where} unit must be a string or null")
        unit = None
    unit = (unit or "").strip() or None
    if value_type != "continuous":
        unit = None
    descriptions = {key: _text(raw[key]) for key in ("description", "measurement_definition",
                                                       "missing_value_rule")}
    # The reserved-answer rule applies to every variable, so an empty
    # missing-value rule takes it rather than costing a repair.
    descriptions["missing_value_rule"] = descriptions["missing_value_rule"] or DOCUMENTED_UNKNOWN_RULE
    for key, text in descriptions.items():
        if not text:
            problems.append(f"{where} needs a nonempty {key}")

    folded = [category.casefold() for category in categories]
    if len(set(folded)) != len(folded):
        problems.append(f"{where} has duplicate labels")
    elif value_type in LABELED_VALUE_TYPES and categories:
        # Extraction reads labels through the same normalizer; a label set it
        # would rewrite (for example a bare range 1-2 read as levels 1 and 2)
        # is rejected here rather than changed later.
        from .plain_handoff_stage2_analysis import _validated_closed_category_values

        try:
            checked = _validated_closed_category_values(
                value_type=value_type, values=categories, source=where, keep_unknown_like=True)
        except ValueError as exc:
            problems.append(str(exc))
        else:
            if checked != categories:
                problems.append(
                    f"{where} labels {categories!r} would be read as {checked!r}: write each label as one "
                    "answer (a bare integer range such as 1-2 is read as separate levels; name it, for "
                    "example Score 1-2)")
    for category in categories:
        if is_unknown_label(category):
            problems.append(
                f"{where} has the label {category!r}, which names an unknown or undetermined state: "
                "remove it (keep any residual label such as Other separately); documented_unknown is a "
                "reserved answer for every variable")
        elif value_type != "binary" and is_not_done_label(category):
            problems.append(
                f"{where} has the label {category!r}, which says the test or assessment was not done: "
                "remove it; a record without a result answers null, or documented_unknown when it states "
                "the value is not assessed (whether a test was done is a separate binary variable)")
        elif _MISSING_LABEL.match(category):
            problems.append(f"{where} has the label {category!r}, which means the value is not documented: "
                            "remove it and answer null in that case")
    numbered = [c for c in categories if re.match(r"^(grade|stage|class|level|score)\s*\d", c.strip(), re.I)]
    worded = [c for c in categories if c not in numbered and not re.search(r"\d", c)
              and not re.match(r"^(none|absent|no\b|not present|normal)", c.strip(), re.I)]
    if value_type == "ordinal" and len(numbered) >= 2 and len(worded) >= 2:
        problems.append(f"{where} mixes numbered levels ({'; '.join(numbered[:4])}) with worded levels "
                        f"({'; '.join(worded[:4])}): the labels of one variable come from one grading system")
    for parent in categories:
        for child in categories:
            if parent != child and is_sublevel(parent, child):
                problems.append(f"{where} lists {parent!r} together with its sublevel {child!r}: labels must "
                                "be mutually exclusive; use either the levels or their sublevels")
    if value_type == "binary" and len(categories) != 2:
        problems.append(f"{where} is binary and needs exactly two labels")
    if value_type in {"categorical", "ordinal"} and len(categories) < 2:
        problems.append(f"{where} needs at least two labels")
    if value_type == "continuous":
        if categories:
            problems.append(f"{where} is continuous and must have no categories")
        if unit and _COMPOSITE_UNIT.search(unit):
            problems.append(f"{where} unit {unit!r} names more than one scale: a continuous variable "
                            "measures one quantity on one scale in one unit; define one variable per scale")
        scale_text = f"{descriptions['description']} {descriptions['measurement_definition']}"
        several = _SEVERAL_SCALES.search(scale_text)
        if several and not _CONVERSION.search(scale_text):
            problems.append(f"{where} takes its number from several instruments or scales "
                            f"({several.group(0).strip()!r}): a continuous variable measures one quantity "
                            "on one scale; define one variable per scale, or state the conversion that "
                            "maps the others onto it")
        if re.search(r"\bgrade\b", f"{name} {unit or ''}".replace("_", " "), re.I):
            problems.append(f"{where} is a grade: give it ordinal labels rather than a number")
        else:
            text = " ".join([name, unit or "", descriptions["description"],
                             descriptions["measurement_definition"]])
            small = _SMALL_RANGE.search(text)
            if _SCORE_WORDS.search(text.replace("_", " ")) and small and not any(
                    int(number) > 10 for pair in _WIDE_RANGE.findall(text) for number in pair if number):
                problems.append(f"{where} records a score with a few defined levels "
                                f"({small.group(0).strip()}) as a number: give it ordinal labels for its levels")

    label_definitions: dict[str, dict[str, str]] = {}
    raw_definitions = raw.get("label_definitions")
    if value_type == "continuous":
        if raw_definitions not in (None, {}, []):
            problems.append(f"{where} is continuous and needs an empty label_definitions object")
    elif require_label_definitions:
        label_definitions, found = normalize_label_definitions(raw_definitions, categories, where)
        problems += found

    raw_rule = raw["conflict_resolution"]
    if isinstance(raw_rule, str):
        raw_rule = {"strategy": raw_rule}
    rule = raw_rule if isinstance(raw_rule, Mapping) else {}
    strategy = canonical_strategy(rule.get("strategy")) or str(rule.get("strategy") or "")
    positive = None
    if strategy not in STRATEGIES:
        problems.append(f"{where} conflict_resolution.strategy {rule.get('strategy')!r} is not one of "
                        + ", ".join(STRATEGIES))
    elif strategy == "any_positive":
        positive = _resolve(rule.get("positive_category"), categories)
        if value_type != "binary" or positive is None:
            problems.append(f"{where} any_positive needs a binary variable and one of its labels as "
                            "positive_category")
    elif strategy in {"maximum", "minimum"} and value_type not in {"continuous", "ordinal"}:
        problems.append(f"{where} {strategy} applies only to numbers or ordinal grades")
    variable = {
        "name": name,
        "description": descriptions["description"],
        "value_type": value_type,
        "unit": unit,
        "categories": categories,
        "label_definitions": label_definitions,
        "measurement_definition": descriptions["measurement_definition"],
        "missing_value_rule": descriptions["missing_value_rule"],
        "conflict_resolution": {"strategy": strategy, "positive_category": positive},
    }
    return variable, problems


def _raise(problems: Sequence[str]) -> None:
    if problems:
        shown = list(dict.fromkeys(problems))
        more = len(shown) - _MAX_REPORTED_PROBLEMS
        text = "; ".join(shown[:_MAX_REPORTED_PROBLEMS])
        raise ValueError(text + (f"; and {more} more problem(s)" if more > 0 else ""))


def definition_fallback(error: Any) -> dict[str, Any]:
    """The recorded outcome when a candidate has no valid closed-format definition."""

    return {"action": _DEFINITION_FALLBACK_ACTION, "reason": str(error), "variables": []}


def validate_definition_response(value: Any) -> dict[str, Any]:
    """Validate one define-or-decompose reply (or its recorded fallback)."""

    if isinstance(value, Mapping) and value.get("action") == _DEFINITION_FALLBACK_ACTION:
        if value.get("variables") != [] or not isinstance(value.get("reason"), str):
            raise ValueError("an invalid-definition record has no variables and a reason")
        return {"action": _DEFINITION_FALLBACK_ACTION, "reason": value["reason"], "variables": []}
    if not isinstance(value, Mapping) or value.get("action") not in {"define", "decompose"}:
        raise ValueError("action must be define or decompose")
    raw_variables = value.get("variables")
    if not isinstance(raw_variables, list) or not all(isinstance(item, Mapping) for item in raw_variables):
        raise ValueError("variables must be an array of objects")
    if value["action"] == "define" and len(raw_variables) != 1:
        raise ValueError("define returns exactly one variable")
    if value["action"] == "decompose" and not 2 <= len(raw_variables) <= MAX_COMPONENTS:
        raise ValueError(f"decompose returns 2 to {MAX_COMPONENTS} variables")
    problems: list[str] = []
    variables = []
    for index, raw in enumerate(raw_variables, start=1):
        variable, found = normalize_variable(raw, f"variable {index} ({raw.get('name')})")
        problems += found
        variables.append(variable)
    names = [variable.get("name") for variable in variables]
    if len(set(names)) != len(names):
        problems.append("variable names must be distinct")
    _raise(problems)
    return {"action": str(value["action"]), "reason": _text(value.get("reason")), "variables": variables}


def definition_request_text(
    *, name: str, descriptions: Sequence[str], rationales: Sequence[str], evidence: Sequence[str],
) -> str:
    """The user message of one define-or-decompose request (readable text only)."""

    distinct = list(dict.fromkeys(text.strip() for text in descriptions if text and text.strip()))
    reasons = list(dict.fromkeys(text.strip() for text in rationales if text and text.strip()))
    lines = [f"Candidate variable: {clinical_prompts.label(name)}",
             f"Description: {distinct[0] if distinct else clinical_prompts.label(name)}"]
    if len(distinct) > 1:
        lines.append("Other descriptions of this candidate: " + "; ".join(distinct[1:6]))
    if reasons:
        lines.append("Suggested by: " + " ".join(reasons[:3]))
    return "\n".join(lines) + "\n\nExample clinical excerpts\n\n" + "\n\n".join(evidence)


def merge_listing(entries: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    """Render one merge batch; returns (user message, batch-local identifiers in order)."""

    identifiers = [f"V{index:02d}" for index in range(1, len(entries) + 1)]
    blocks = []
    for identifier, entry in zip(identifiers, entries):
        variable = entry["definition"]
        lines = [f"{identifier}: {clinical_prompts.label(variable['name'])}"]
        if entry.get("locked"):
            lines.append("- Locked: its definition cannot change")
        lines += [f"- Description: {variable['description'][:600]}", f"- Answer format: {answer_format(variable)}"]
        definitions = variable.get("label_definitions") or {}
        if definitions:
            rendered = "\n".join(
                f"  - {label}: meaning: {item['meaning'][:220]} | established by: {item['established_by'][:160]}"
                for label, item in definitions.items())
            lines.append("- Label definitions:\n" + rendered[:2400])
        lines.append(f"- Measurement rule: {variable['measurement_definition'][:400]}")
        blocks.append("\n".join(lines))
    return "Variables\n\n" + "\n\n".join(blocks), identifiers


def validate_merge_response(value: Any, *, members: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Validate one merge reply against its batch (batch-local identifier -> pool entry)."""

    groups = value.get("groups") if isinstance(value, Mapping) else None
    if not isinstance(groups, list):
        raise ValueError("groups must be an array")
    problems: list[str] = []
    normalized_groups = []
    seen: set[str] = set()
    for n, group in enumerate(groups, start=1):
        ids = group.get("members") if isinstance(group, Mapping) else None
        if not isinstance(ids, list) or len(ids) < 2 or any(not isinstance(m, str) or m not in members for m in ids):
            problems.append(f"group {n} members must be at least two listed identifiers")
            continue
        if len(set(ids)) != len(ids) or seen & set(ids):
            problems.append(f"group {n} reuses an identifier")
        seen |= set(ids)
        locked = [m for m in ids if members[m].get("locked")]
        if len(locked) > 1:
            problems.append(f"group {n} contains more than one locked variable: keep them apart")
            continue
        variables = {m: members[m]["definition"] for m in ids}
        continuous = {variables[m]["value_type"] == "continuous" for m in ids}
        if len(continuous) > 1:
            problems.append(f"group {n} mixes numbers and labels: keep those members apart")
            continue
        if locked:
            # The locked definition is the shared definition; the reply only has
            # to restate its identity, and its own labels are never re-checked.
            fixed, raw = variables[locked[0]], group.get("definition")
            raw = raw if isinstance(raw, Mapping) else {}
            raw_labels = raw.get("categories") if isinstance(raw.get("categories"), list) else []
            same = (snake_case(raw.get("name")) == fixed["name"]
                    and str(raw.get("value_type") or "").strip().lower() == fixed["value_type"]
                    and sorted(str(c).strip().casefold() for c in raw_labels)
                    == sorted(c.casefold() for c in fixed["categories"])
                    and str(raw.get("unit") or "").strip().casefold() == (fixed["unit"] or "").casefold())
            if not same:
                problems.append(f"group {n} contains the locked variable {locked[0]}: the shared definition "
                                "keeps its name, value type, unit, and labels unchanged")
                continue
            definition = dict(fixed)
        else:
            raw_definition = group.get("definition")
            if isinstance(raw_definition, Mapping):
                raw_rule = raw_definition.get("conflict_resolution")
                raw_strategy = raw_rule.get("strategy") if isinstance(raw_rule, Mapping) else raw_rule
                if canonical_strategy(raw_strategy) is None:
                    raw_definition = {**raw_definition, "conflict_resolution": _member_rule(group, variables)}
            definition, found = normalize_variable(raw_definition, f"group {n} definition")
            problems += found
            if found:
                continue
        if (definition["value_type"] == "continuous") not in continuous:
            problems.append(f"group {n} mixes numbers and labels: keep those members apart")
            continue
        shared = definition["categories"]
        crosswalk: dict[tuple[str, str], str] = {}
        rows = group.get("category_crosswalk") or []
        if not isinstance(rows, list):
            problems.append(f"group {n} category_crosswalk must be an array")
            continue
        for row in rows:
            if not isinstance(row, Mapping) or row.get("member") not in ids:
                problems.append(f"group {n} has a crosswalk row without a member identifier of the group")
                continue
            member = row["member"]
            label = _resolve(row.get("member_label"), variables[member]["categories"])
            target = _resolve(row.get("shared_label"), shared)
            if label is None:
                problems.append(f"group {n} maps {row.get('member_label')!r}, which is not a label of {member}")
                continue
            if target is None:
                problems.append(f"group {n} maps {label!r} to a label outside the shared definition")
                continue
            if (member, label) in crosswalk:
                problems.append(f"group {n} maps {member} label {label!r} more than once")
            crosswalk[(member, label)] = target
        for member in ids:
            variable = variables[member]
            targets = [crosswalk.get((member, label)) for label in variable["categories"]]
            if None in targets:
                problems.append(f"group {n} must map every label of {member}")
                continue
            if len(set(targets)) < len(targets):
                problems.append(f"group {n} collapses two labels of {member} into one shared label: keep it "
                                "apart or use labels that preserve the distinction")
                continue
            context = _stems(f"{variable['name']} {variable['description']}") | {
                stem for label in variable["categories"] for stem in _stems(label)}
            for label, target in zip(variable["categories"], targets):
                residual = is_residual_label(label)
                if residual and not is_residual_label(target):
                    problems.append(f"group {n} maps the residual label {label!r} of {member} to the specific "
                                    f"label {target!r}: map it to a residual shared label or keep {member} apart")
                elif residual and _residual_scope(target) - _residual_scope(label) - context:
                    problems.append(f"group {n} maps the residual label {label!r} of {member} to {target!r}, the "
                                    f"residual of a narrower class: a residual label keeps its scope; keep "
                                    f"{member} apart")
                elif not residual and is_residual_label(target):
                    problems.append(f"group {n} maps the specific label {label!r} of {member} to the residual "
                                    f"label {target!r}: a residual label holds only answers the specific labels "
                                    f"do not name; when {label!r} covers several shared labels, keep {member} apart")
                elif is_sublevel(label, target):
                    problems.append(f"group {n} maps {label!r} of {member} to its sublevel {target!r}: a member "
                                    f"label maps to a shared label with the same meaning; keep {member} apart")
        if definition["value_type"] != "continuous":
            unused = [label for label in shared if label not in set(crosswalk.values())]
            if unused:
                problems.append(f"group {n} has shared labels that no member label maps to "
                                f"({'; '.join(unused)}): do not add labels that no member has")
            targets_by_member = {m: {crosswalk.get((m, label)) for label in variables[m]["categories"]}
                                 for m in ids}
            for index, first in enumerate(ids):
                for second in ids[index + 1:]:
                    a, b = targets_by_member[first], targets_by_member[second]
                    if len(a) > 1 and len(b) > 1 and not a & b:
                        problems.append(f"group {n}: {first} and {second} share no answer (their labels map to "
                                        "different shared labels), so they classify the patient in different "
                                        "systems or scopes: keep them apart")
        normalized_groups.append({
            "members": list(ids),
            "reason": _text(group.get("reason")),
            "definition": definition,
            "category_crosswalk": [
                {"member": member, "member_label": label, "shared_label": target}
                for (member, label), target in sorted(
                    crosswalk.items(), key=lambda item: (ids.index(item[0][0]),
                                                         variables[item[0][0]]["categories"].index(item[0][1])))],
        })
    _raise(problems)
    return {"groups": normalized_groups}


def _combined(values: Sequence[Sequence[str]]) -> list[str]:
    return list(dict.fromkeys(item for value in values for item in value))


def merged_entry(entry_id: str, members: Sequence[Mapping[str, Any]], *, definition: Mapping[str, Any],
                 label_map: Mapping[str, str | None]) -> dict[str, Any]:
    # A locked member leads, so its candidate stays the retained origin.
    members = sorted(members, key=lambda member: not member.get("locked"))
    locked = [member for member in members if member.get("locked")]
    return {
        "entry_id": entry_id,
        "definition": dict(locked[0]["definition"] if locked else definition),
        "locked": bool(locked),
        "explicit_feature": locked[0].get("explicit_feature") if locked else None,
        "origin_variable_ids": _combined([member["origin_variable_ids"] for member in members]),
        "origin_candidate_ids": _combined([member["origin_candidate_ids"] for member in members]),
        "supporting_packet_ids": _combined([member["supporting_packet_ids"] for member in members]),
        "supporting_architectures": _combined([member["supporting_architectures"] for member in members]),
        "label_map": dict(label_map),
    }


def apply_merge_groups(
    entries: Sequence[Mapping[str, Any]], identifiers: Sequence[str], response: Mapping[str, Any],
    *, entry_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace each merged group of one batch by its shared definition."""

    by_identifier = dict(zip(identifiers, entries))
    merged_ids: set[str] = set()
    output: list[dict[str, Any]] = []
    summaries = []
    for number, group in enumerate(response["groups"], start=1):
        crosswalk = {(row["member"], row["member_label"]): row["shared_label"]
                     for row in group["category_crosswalk"]}
        label_map: dict[str, str | None] = {}
        for identifier in group["members"]:
            member = by_identifier[identifier]
            for key, current in member["label_map"].items():
                label_map[key] = crosswalk.get((identifier, current)) if current is not None else None
        members = [by_identifier[identifier] for identifier in group["members"]]
        entry = merged_entry(f"{entry_prefix}_{number:03d}", members, definition=group["definition"],
                             label_map=label_map)
        output.append(entry)
        merged_ids |= {member["entry_id"] for member in members}
        summaries.append({"entry_id": entry["entry_id"], "members": [m["entry_id"] for m in members],
                          "name": entry["definition"]["name"], "reason": group["reason"]})
    unchanged = [dict(entry) for entry in entries if entry["entry_id"] not in merged_ids]
    return unchanged + output, summaries


def _format_key(variable: Mapping[str, Any]) -> tuple[Any, ...]:
    answers = ((variable.get("unit") or "").casefold(),) if variable["value_type"] == "continuous" else tuple(
        label.casefold() for label in variable["categories"])
    return (snake_case(variable["name"]), variable["value_type"], answers)


def coalesce_exact_duplicates(
    entries: Sequence[Mapping[str, Any]], *, entry_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge variables with the same name and answer format without a model review."""

    by_key: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for entry in sorted(entries, key=lambda item: (not item.get("locked"), item["entry_id"])):
        by_key.setdefault(_format_key(entry["definition"]), []).append(entry)
    output, summaries = [], []
    for same in by_key.values():
        if len(same) == 1 or sum(bool(entry.get("locked")) for entry in same) > 1:
            output.extend(dict(entry) for entry in same)
            continue
        canonical = same[0]["definition"]
        labels = {label.casefold(): label for label in canonical["categories"]}
        label_map = {key: (labels[current.casefold()] if current is not None else None)
                     for entry in same for key, current in entry["label_map"].items()}
        entry = merged_entry(f"{entry_prefix}_{len(summaries) + 1:03d}", same, definition=canonical,
                             label_map=label_map)
        output.append(entry)
        summaries.append({"entry_id": entry["entry_id"], "members": [item["entry_id"] for item in same],
                          "name": canonical["name"], "reason": "same name and answer format"})
    return output, summaries


def pool_view_name(entry: Mapping[str, Any]) -> str:
    """A unique, sortable batching key."""

    return f"{entry['definition']['name']}#{entry['entry_id']}"


def embedding_text(entry: Mapping[str, Any]) -> str:
    variable = entry["definition"]
    return f"{clinical_prompts.label(variable['name'])}: {variable['description']} ({answer_format(variable)})"


def unique_feature_names(entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """entry_id -> distinct feature name; locked names never change."""

    names: dict[str, str] = {}
    taken = {entry["definition"]["name"] for entry in entries if entry.get("locked")}
    for entry in sorted(entries, key=lambda item: (not item.get("locked"), item["definition"]["name"],
                                                   item["entry_id"])):
        base = entry["definition"]["name"]
        if entry.get("locked"):
            names[entry["entry_id"]] = base
            continue
        name, suffix = base, 2
        while name in taken:
            name, suffix = f"{base}_{suffix}", suffix + 1
        taken.add(name)
        names[entry["entry_id"]] = name
    return names


def feature_from_entry(entry: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """A Stage 2 feature definition (``categories_or_unit`` form) from a pool entry."""

    if entry.get("explicit_feature"):
        return dict(entry["explicit_feature"])
    variable = entry["definition"]
    continuous = variable["value_type"] == "continuous"
    packets, architectures = entry["supporting_packet_ids"], entry["supporting_architectures"]
    return {
        "name": name,
        "description": variable["description"],
        "value_type": variable["value_type"],
        "categories_or_unit": ([variable["unit"]] if variable["unit"] else []) if continuous
        else list(variable["categories"]),
        "label_definitions": {} if continuous else {
            label: dict(variable["label_definitions"][label]) for label in variable["categories"]},
        "measurement_definition": variable["measurement_definition"],
        "missing_value_rule": variable["missing_value_rule"],
        "conflict_resolution": dict(variable["conflict_resolution"]),
        "stability_summary": (f"Supported by {len(packets)} evidence packet(s) across "
                              f"{len(architectures)} Stage 1 architecture(s)."),
        "caveats": "",
    }


def locked_definition(feature: Mapping[str, Any]) -> dict[str, Any]:
    """The pool form of an investigator-configured feature (its definition is fixed)."""

    value_type = str(feature["value_type"])
    values = [str(item) for item in feature.get("categories_or_unit") or []]
    rule = dict(feature.get("conflict_resolution") or {})
    return {
        "name": snake_case(feature["name"]),
        "description": str(feature.get("description") or ""),
        "value_type": value_type,
        "unit": (values[0] if values else None) if value_type == "continuous" else None,
        "categories": [] if value_type == "continuous" else values,
        "label_definitions": {},
        "measurement_definition": str(feature.get("measurement_definition") or ""),
        "missing_value_rule": str(feature.get("missing_value_rule") or ""),
        "conflict_resolution": {"strategy": rule.get("strategy"), "positive_category": rule.get("positive_category")},
    }


def dispositions(entries: Sequence[Mapping[str, Any]], names: Mapping[str, str],
                 dropped: Mapping[str, str], all_candidate_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """candidate ID -> how it reached the final features (retained, merged, decomposed, or dropped).

    Within a feature built from several candidates, the first origin is
    retained and the others are merged; a candidate split across several
    features is decomposed.
    """

    features_by_candidate: dict[str, list[str]] = {}
    shared_by: Counter[str] = Counter()
    for entry in entries:
        for candidate_id in entry["origin_candidate_ids"]:
            features_by_candidate.setdefault(candidate_id, []).append(names[entry["entry_id"]])
        shared_by.update(entry["origin_candidate_ids"][1:])
    result: dict[str, dict[str, Any]] = {}
    for candidate_id in all_candidate_ids:
        features = features_by_candidate.get(candidate_id, [])
        if not features:
            result[candidate_id] = {"status": "dropped", "feature_name": None, "feature_names": [],
                                    "reason": dropped.get(candidate_id)
                                    or "No closed-format variable remained for this candidate."}
            continue
        status = "decomposed" if len(features) > 1 else ("merged" if shared_by[candidate_id] else "retained")
        result[candidate_id] = {
            "status": status, "feature_name": features[0], "feature_names": features,
            "reason": {"decomposed": "Decomposed into closed-format component variables.",
                       "merged": "Merged with variables measuring the same attribute.",
                       "retained": "Defined as one closed-format variable."}[status],
        }
    return result


def entry_view(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The checkpointed identity of one pool entry (its definition and lock)."""

    return {"entry_id": entry["entry_id"], "locked": bool(entry.get("locked")),
            "definition": entry["definition"]}
