"""Post-hoc attribution of Stage 2 variables to the Stage 1 architectures behind them.

The report reads frozen Stage 2 outputs only; it never fits, selects, or
rewrites anything a run reads. For every outer fold it follows each variable
back through the lineage fields Stage 2 records:

* ``stage2/evidence_compilation/outer_00k/cards.jsonl``: each evidence card and
  its ``source_architectures``;
* ``outer_00k/interpreted_candidates.json``: each candidate and the cards
  (``supporting_packet_ids``) it was proposed from;
* ``outer_00k/feature_definitions.json``: the defined variables, their
  ``supporting_packet_ids`` and ``supporting_architectures``, and the
  ``candidate_dispositions`` that map every candidate to its variable;
* ``extraction/candidate_missingness_filter.json``: the informativeness and
  extraction-failure filter;
* ``final_definitions.json``: the selected variables and their roles.

An architecture supports a variable when one of the variable's cards, or a
card of one of its candidates, came from that architecture (or the variable
records it in ``supporting_architectures``). Variables derived later inherit
lineage: split and estimand-search variants through their parent definition,
consolidated latents through their measurement dependencies. A variable is
sole-source when exactly one architecture supports it and it is not
investigator-specified (``stage2.explicit_features``).

With ``--oracle`` the report first hashes every Stage 2 artifact it reads and
closes the reader, then loads the oracle columns. Recovery is measured by data:
on each fold's outer held-out rows, the oracle column is regressed on the
variable's encoded columns from the frozen held-out extraction (ordinary least
squares with an intercept), and the variable recovers the oracle concept when
R^2 reaches ``--recovery-r2``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..inference.stage1_architectures import STAGE1_ARCHITECTURES

ATTRIBUTION_SCHEMA_VERSION = "stage2_architecture_attribution_v1"
ATTRIBUTION_MANIFEST_SCHEMA_VERSION = "stage2_architecture_attribution_manifest_v1"
DEFAULT_RECOVERY_R2 = 0.5
LEVELS = (
    "cards",
    "candidates",
    "variables_defined",
    "variables_passing_filter",
    "nuisance_set",
    "selected_modifiers",
)
RECOVERY_LEVELS = ("selected_variables", "nuisance_set", "selected_modifiers")
ROW_ID = "_oci_row_id"
EXTRACTION_FAILURE_SUFFIX = "__oci_extraction_failed"
CONFIGURED_CANDIDATE_PREFIX = "configured_explicit_feature_"
ANALYSIS_COHORT = Path("input") / "analysis_cohort.parquet"
# Oracle columns that describe the data-generating outcome model rather than a
# patient characteristic; they are never recovery targets.
EFFECT_ORACLE_COLUMNS = frozenset(
    {
        "true_treatment_prob",
        "true_outcome_prob",
        "true_y0",
        "true_y1",
        "true_ite",
        "true_y0_prob",
        "true_y1_prob",
        "true_ite_prob",
    }
)
LINEAGE_RULE = (
    "An architecture supports a variable when a card cited by the variable "
    "(supporting_packet_ids) or by one of its candidates (candidate_dispositions) has that "
    "architecture in source_architectures, or when the variable records it in "
    "supporting_architectures. Split and estimand-search variants inherit their parent "
    "definition's candidates; consolidated latents take the union of their measurement "
    "dependencies. A variable is sole-source when exactly one architecture supports it and "
    "it is not investigator-specified."
)
RECOVERY_RULE = (
    "On each fold's outer held-out rows, the oracle column is regressed on the variable's "
    "columns from encode_measurements applied to the frozen held-out extraction (ordinary "
    "least squares with an intercept; missing continuous values are set to 0 beside their "
    "missing-state indicator). Continuous oracle columns are numeric targets; other oracle "
    "columns are one-hot encoded and R^2 is 1 - sum(SSE) / sum(SST) over their levels. A "
    "variable recovers the concept when R^2 >= the threshold."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    from ..inference.stage2_diagnostics import json_safe

    return json_safe(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    rows = list(rows)
    frame = pd.DataFrame(rows, columns=list(columns))
    for column in frame.columns:
        values = [row[column] for row in rows if row.get(column) is not None]
        # Keep integer counts integral when some rows leave them blank.
        if values and all(
            isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            for value in values
        ):
            frame[column] = frame[column].astype("Int64")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_architectures(values: Iterable[Any]) -> list[str]:
    """Known Stage 1 architecture names among ``values``, in registry order."""

    present = {str(value) for value in values}
    return [name for name in STAGE1_ARCHITECTURES if name in present]


def assert_output_outside_run_inputs(output_dir: Path, run_dir: Path) -> None:
    """Refuse output paths that a run reads.

    Outside the run directory anything goes; inside it, reports belong under
    ``evaluation/``, which no pipeline step reads.
    """

    output_dir = Path(output_dir).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    try:
        relative = output_dir.relative_to(run_dir)
    except ValueError:
        return
    if relative.parts and relative.parts[0] == "evaluation":
        return
    raise ValueError(
        f"refusing to write evaluation outputs to {output_dir}: inside the run directory "
        f"they must be under {run_dir / 'evaluation'}"
    )


class FrozenArtifactReader:
    """Read run artifacts once, hashing the exact bytes that are parsed.

    ``freeze()`` closes the reader. Oracle data may be loaded only after every
    reader in use is frozen, so no artifact can be read after oracle values are
    in memory.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.hashes: dict[str, str] = {}
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def _key(self, path: Path) -> str:
        try:
            return os.path.relpath(Path(path).resolve(), start=self.root.resolve())
        except ValueError:
            return str(Path(path).resolve())

    def read_bytes(self, path: Path) -> bytes | None:
        if self._frozen:
            raise RuntimeError(
                "artifact reads are closed: the reader was frozen before oracle data loaded"
            )
        path = Path(path)
        if not path.is_file():
            return None
        data = path.read_bytes()
        self.hashes[self._key(path)] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, path: Path) -> Any:
        data = self.read_bytes(path)
        return None if data is None else json.loads(data.decode("utf-8"))

    def jsonl(self, path: Path) -> list[dict[str, Any]] | None:
        data = self.read_bytes(path)
        if data is None:
            return None
        return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]

    def measurements(self, path: Path, mask_path: Path | None) -> pd.DataFrame | None:
        """Held-out measurements as strings, with extraction-failure companions.

        Mirrors ``stage2_diagnostics._read_measurements`` on the hashed bytes.
        """

        data = self.read_bytes(path)
        if data is None:
            return None
        frame = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=True)
        mask_bytes = self.read_bytes(mask_path) if mask_path is not None else None
        if mask_bytes is None or ROW_ID not in frame.columns:
            return frame
        mask = pd.read_csv(io.BytesIO(mask_bytes))
        if ROW_ID not in mask.columns:
            return frame
        companions = {
            column: f"{column}{EXTRACTION_FAILURE_SUFFIX}"
            for column in mask.columns
            if column != ROW_ID and f"{column}{EXTRACTION_FAILURE_SUFFIX}" not in frame.columns
        }
        if not companions:
            return frame
        mask = mask[[ROW_ID, *companions]].rename(columns=companions)
        mask[ROW_ID] = pd.to_numeric(mask[ROW_ID]).astype(int)
        frame[ROW_ID] = pd.to_numeric(frame[ROW_ID]).astype(int)
        return frame.merge(mask, on=ROW_ID, how="left")

    def freeze(self) -> dict[str, str]:
        self._frozen = True
        return dict(sorted(self.hashes.items()))


def _features(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, Mapping) and isinstance(payload.get("features"), list):
        return [dict(item) for item in payload["features"] if isinstance(item, Mapping)]
    if isinstance(payload, Mapping) and isinstance(payload.get("definitions"), list):
        return [dict(item) for item in payload["definitions"] if isinstance(item, Mapping)]
    return None


def _strings(value: Any) -> list[str]:
    if value is None or isinstance(value, (str, bytes)):
        return [str(value)] if isinstance(value, str) and value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def _roles(feature: Mapping[str, Any]) -> set[str]:
    return {str(role) for role in feature.get("roles") or []}


@dataclass
class FoldLineage:
    """One outer fold's frozen lineage artifacts."""

    fold: str
    cards: dict[str, list[str]] | None = None
    candidates: list[dict[str, Any]] | None = None
    dispositions: dict[str, dict[str, Any]] = field(default_factory=dict)
    definitions: list[dict[str, Any]] | None = None
    catalog: dict[str, dict[str, Any]] = field(default_factory=dict)
    filter_kept: list[str] | None = None
    final_features: list[dict[str, Any]] | None = None
    heldout_frame: pd.DataFrame | None = None
    heldout_definitions: list[dict[str, Any]] | None = None
    heldout_source: str | None = None
    unavailable: dict[str, str] = field(default_factory=dict)

    @property
    def definition_ids(self) -> dict[str, dict[str, Any]]:
        return {str(item.get("feature_id")): item for item in self.definitions or []}

    @property
    def candidates_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(item.get("candidate_id")): item for item in self.candidates or []}


@dataclass
class Stage2Lineage:
    """Lineage artifacts of one Stage 2 directory, read through one reader."""

    stage2_dir: Path
    reader: FrozenArtifactReader
    folds: dict[str, FoldLineage]
    architectures: list[str]
    architecture_source: str


def _fold_label(directory: Path) -> str | None:
    name = directory.name
    if not name.startswith("outer_"):
        return None
    suffix = name[len("outer_"):]
    return str(int(suffix)) if suffix.isdigit() else None


def _fold_sort_key(fold: str) -> tuple[int, str]:
    return (int(fold), fold) if str(fold).isdigit() else (10**9, str(fold))


def collect_stage2_lineage(
    stage2_dir: Path,
    *,
    reader: FrozenArtifactReader | None = None,
    include_heldout: bool = False,
    selected_architectures: Sequence[str] | None = None,
) -> Stage2Lineage:
    """Read every lineage artifact of a Stage 2 directory (no oracle data)."""

    stage2_dir = Path(stage2_dir).expanduser().resolve()
    if not stage2_dir.is_dir():
        raise FileNotFoundError(f"Stage 2 output directory not found: {stage2_dir}")
    reader = reader or FrozenArtifactReader(stage2_dir.parent)
    compilation = stage2_dir / "evidence_compilation"
    labels = {
        label
        for root in (stage2_dir, compilation)
        if root.is_dir()
        for path in root.iterdir()
        if path.is_dir()
        for label in [_fold_label(path)]
        if label is not None
    }
    packets_by_fold: dict[str, list[dict[str, Any]]] | None = None
    folds: dict[str, FoldLineage] = {}
    for fold in sorted(labels, key=_fold_sort_key):
        fold_name = f"outer_{int(fold):03d}"
        lineage = FoldLineage(fold=fold)
        cards = reader.jsonl(compilation / fold_name / "cards.jsonl")
        if cards is None:
            # Without per-fold card manifests, read the cards from the packets.
            if packets_by_fold is None:
                packets_by_fold = {}
                for packet in reader.jsonl(compilation / "packets.jsonl") or []:
                    packets_by_fold.setdefault(str(int(packet["outer_fold"])), []).append(packet)
            packets = packets_by_fold.get(fold) or reader.jsonl(
                stage2_dir / fold_name / "input_packets.jsonl"
            )
            if packets is not None:
                cards = [
                    {
                        "card_id": packet.get("packet_id"),
                        "source_architectures": (packet.get("content") or {}).get(
                            "source_architectures"
                        )
                        or [packet.get("architecture")],
                    }
                    for packet in packets
                ]
        if cards is None:
            lineage.unavailable["cards"] = "no evidence cards were found for this fold"
        else:
            lineage.cards = {
                str(card["card_id"]): canonical_architectures(card.get("source_architectures") or [])
                for card in cards
            }
        fold_dir = stage2_dir / fold_name
        candidates = reader.json(fold_dir / "interpreted_candidates.json")
        if isinstance(candidates, list):
            lineage.candidates = [dict(item) for item in candidates if isinstance(item, Mapping)]
        else:
            lineage.unavailable["candidates"] = "interpreted_candidates.json is missing"
        definitions_payload = reader.json(fold_dir / "feature_definitions.json")
        lineage.definitions = _features(definitions_payload)
        if lineage.definitions is None:
            lineage.unavailable["variables_defined"] = "feature_definitions.json is missing"
        else:
            raw = definitions_payload.get("candidate_dispositions") or {}
            lineage.dispositions = {
                str(key): dict(value) for key, value in raw.items() if isinstance(value, Mapping)
            }
        filter_audit = reader.json(fold_dir / "extraction" / "candidate_missingness_filter.json")
        estimand_audit = reader.json(
            fold_dir / "extraction" / "estimand_candidate_missingness_filter.json"
        )
        if isinstance(filter_audit, Mapping) and isinstance(filter_audit.get("features"), list):
            kept: list[str] = []
            for audit in (filter_audit, estimand_audit):
                if not isinstance(audit, Mapping):
                    continue
                for decision in audit.get("features") or []:
                    key = str(decision.get("feature_id"))
                    if decision.get("action") == "keep" and key not in kept:
                        kept.append(key)
            lineage.filter_kept = kept
        else:
            lineage.unavailable["variables_passing_filter"] = (
                "extraction/candidate_missingness_filter.json is missing"
            )
        final_payload = reader.json(fold_dir / "final_definitions.json")
        lineage.final_features = _features(final_payload)
        if lineage.final_features is None:
            reason = "final_definitions.json is missing"
            lineage.unavailable["nuisance_set"] = reason
            lineage.unavailable["selected_modifiers"] = reason
        # Every definition source contributes to the feature-id catalog used to
        # resolve filtered variables, latents, and their measurement dependencies.
        sources: list[list[dict[str, Any]] | None] = [
            lineage.final_features,
            _features({"features": (final_payload or {}).get("measurement_dependencies")})
            if isinstance(final_payload, Mapping)
            else None,
            _features(reader.json(fold_dir / "selection" / "nuisance_candidates.json")),
            _features(reader.json(fold_dir / "selection" / "input.json")),
            lineage.definitions,
        ]
        for features in sources:
            for feature in features or []:
                key = str(feature.get("feature_id") or "")
                if key and key not in lineage.catalog:
                    lineage.catalog[key] = feature
        if include_heldout:
            _collect_heldout(reader, fold_dir, lineage)
        folds[fold] = lineage
    architectures, source = _report_architectures(
        reader, stage2_dir, folds, selected_architectures
    )
    return Stage2Lineage(
        stage2_dir=stage2_dir,
        reader=reader,
        folds=folds,
        architectures=architectures,
        architecture_source=source,
    )


def _collect_heldout(reader: FrozenArtifactReader, fold_dir: Path, lineage: FoldLineage) -> None:
    mask = fold_dir / "extraction" / "failure_mask_heldout.csv"
    candidates = fold_dir / "estimation" / "nuisance_candidate_heldout.csv"
    definitions = None
    for path in (
        fold_dir / "estimation" / "nuisance_candidate_definitions.json",
        fold_dir / "selection" / "nuisance_candidates.json",
    ):
        definitions = _features(reader.json(path))
        if definitions is not None:
            break
    frame = reader.measurements(candidates, mask) if definitions is not None else None
    if frame is not None and definitions is not None:
        lineage.heldout_frame = frame
        lineage.heldout_definitions = definitions
        lineage.heldout_source = "estimation/nuisance_candidate_heldout.csv"
        return
    retained = fold_dir / "extraction" / "heldout" / "harmonized.csv"
    frame = reader.measurements(retained, mask)
    if frame is not None and lineage.final_features is not None:
        lineage.heldout_frame = frame
        lineage.heldout_definitions = lineage.final_features
        lineage.heldout_source = "extraction/heldout/harmonized.csv"
        return
    lineage.unavailable["oracle_recovery"] = "no frozen held-out extraction for this fold"


def _report_architectures(
    reader: FrozenArtifactReader,
    stage2_dir: Path,
    folds: Mapping[str, FoldLineage],
    selected: Sequence[str] | None,
) -> tuple[list[str], str]:
    present = {
        architecture
        for fold in folds.values()
        for architectures in (fold.cards or {}).values()
        for architecture in architectures
    }
    if selected is not None:
        return canonical_architectures([*selected, *present]), "argument"
    config = reader.json(stage2_dir / "config.json")
    included = config.get("included_architectures") if isinstance(config, Mapping) else None
    if included:
        return canonical_architectures([*included, *present]), "stage2/config.json"
    manifest = reader.json(stage2_dir.parent / "stage1_architectures" / "manifest.json")
    if isinstance(manifest, Mapping) and manifest.get("selected_architectures"):
        return (
            canonical_architectures([*manifest["selected_architectures"], *present]),
            "stage1_architectures/manifest.json",
        )
    return canonical_architectures(present), "evidence_cards"


def _definition_ancestor(
    feature_id: str,
    feature: Mapping[str, Any],
    lineage: FoldLineage,
) -> Mapping[str, Any] | None:
    """The feature_definitions.json entry a later variable descends from."""

    definitions = lineage.definition_ids
    current_id, current = feature_id, feature
    seen: set[str] = set()
    while current_id not in definitions:
        if current_id in seen:
            return None
        seen.add(current_id)
        parent = current.get("ontology_parent_feature_id") or (
            (current.get("estimand_ontology") or {}).get("source_feature_id")
            if isinstance(current.get("estimand_ontology"), Mapping)
            else None
        )
        if not parent and "__" in current_id:
            parent = current_id.split("__", 1)[0]
        if not parent:
            return None
        current_id = str(parent)
        current = lineage.catalog.get(current_id, {})
    return definitions[current_id]


def variable_lineage(
    feature: Mapping[str, Any],
    lineage: FoldLineage,
    _visiting: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Cards, candidates, and supporting architectures of one Stage 2 variable."""

    feature_id = str(feature.get("feature_id") or feature.get("name") or "")
    name = str(feature.get("name") or feature_id)
    if feature.get("derived_structured_latent"):
        cards: set[str] = set()
        candidates: set[str] = set()
        architectures: set[str] = set()
        unresolved: set[str] = set()
        investigator = False
        for dependency in _strings(feature.get("measurement_dependency_feature_ids")):
            if dependency in _visiting:
                continue
            part = variable_lineage(
                lineage.catalog.get(dependency, {"feature_id": dependency}),
                lineage,
                _visiting | {feature_id},
            )
            cards.update(part["card_ids"])
            candidates.update(part["candidate_ids"])
            architectures.update(part["architectures"])
            unresolved.update(part["unresolved_card_ids"])
            investigator = investigator or part["investigator_specified"]
        kind = "latent"
    else:
        cards = set(_strings(feature.get("supporting_packet_ids")))
        recorded = set(_strings(feature.get("supporting_architectures")))
        investigator = feature.get("configured_explicit_feature") is True
        ancestor = _definition_ancestor(feature_id, feature, lineage)
        if ancestor is not None:
            cards.update(_strings(ancestor.get("supporting_packet_ids")))
            recorded.update(_strings(ancestor.get("supporting_architectures")))
            investigator = investigator or ancestor.get("configured_explicit_feature") is True
            ancestor_name = str(ancestor.get("name") or "")
        else:
            ancestor_name = name
        candidates = {
            candidate_id
            for candidate_id, disposition in lineage.dispositions.items()
            if str(disposition.get("feature_name") or "") == ancestor_name
        }
        by_id = lineage.candidates_by_id
        for candidate_id in candidates:
            if candidate_id.startswith(CONFIGURED_CANDIDATE_PREFIX):
                investigator = True
            candidate = by_id.get(candidate_id, {})
            cards.update(_strings(candidate.get("supporting_packet_ids")))
            recorded.update(_strings(candidate.get("supporting_architectures")))
            recorded.update(_strings(candidate.get("architecture")))
        known_cards = lineage.cards or {}
        unresolved = {card for card in cards if card not in known_cards}
        architectures = {
            architecture for card in cards for architecture in known_cards.get(card, [])
        } | set(canonical_architectures(recorded))
        kind = "investigator_specified" if investigator and not cards else "measurement"
    ordered = canonical_architectures(architectures)
    return {
        "feature_id": feature_id,
        "name": name,
        "kind": kind,
        "card_ids": sorted(cards),
        "candidate_ids": sorted(candidates),
        "architectures": ordered,
        "investigator_specified": bool(investigator),
        "sole_source_architecture": (
            ordered[0] if len(ordered) == 1 and not investigator else None
        ),
        "unresolved_card_ids": sorted(unresolved),
    }


def _candidate_item(candidate: Mapping[str, Any], lineage: FoldLineage) -> dict[str, Any]:
    cards = _strings(candidate.get("supporting_packet_ids"))
    known = lineage.cards or {}
    architectures = canonical_architectures(
        [
            *(architecture for card in cards for architecture in known.get(card, [])),
            *_strings(candidate.get("supporting_architectures")),
            *_strings(candidate.get("architecture")),
        ]
    )
    return {
        "id": str(candidate.get("candidate_id")),
        "architectures": architectures,
        "investigator_specified": False,
    }


def level_items(lineage: FoldLineage) -> dict[str, list[dict[str, Any]] | None]:
    """Attributable items of every level in one fold; None when unavailable."""

    def variable(feature: Mapping[str, Any]) -> dict[str, Any]:
        traced = variable_lineage(feature, lineage)
        return {
            "id": traced["feature_id"],
            "architectures": traced["architectures"],
            "investigator_specified": traced["investigator_specified"],
            "lineage": traced,
        }

    items: dict[str, list[dict[str, Any]] | None] = {}
    items["cards"] = (
        None
        if lineage.cards is None
        else [
            {"id": card, "architectures": architectures, "investigator_specified": False}
            for card, architectures in sorted(lineage.cards.items())
        ]
    )
    items["candidates"] = (
        None
        if lineage.candidates is None
        else [_candidate_item(candidate, lineage) for candidate in lineage.candidates]
    )
    items["variables_defined"] = (
        None if lineage.definitions is None else [variable(f) for f in lineage.definitions]
    )
    items["variables_passing_filter"] = (
        None
        if lineage.filter_kept is None
        else [
            variable(lineage.catalog.get(key, {"feature_id": key}))
            for key in lineage.filter_kept
        ]
    )
    if lineage.final_features is None:
        items["nuisance_set"] = None
        items["selected_modifiers"] = None
    else:
        items["nuisance_set"] = [
            variable(f) for f in lineage.final_features if "confounder" in _roles(f)
        ]
        items["selected_modifiers"] = [
            variable(f) for f in lineage.final_features if "effect_modifier" in _roles(f)
        ]
    return items


def _count(items: Sequence[Mapping[str, Any]], architectures: Sequence[str]) -> dict[str, Any]:
    counts = {
        architecture: {
            "any_source": sum(architecture in item["architectures"] for item in items),
            "sole_source": sum(
                list(item["architectures"]) == [architecture]
                and not item["investigator_specified"]
                for item in items
            ),
        }
        for architecture in architectures
    }
    totals = {
        "items": len(items),
        "multi_source": sum(len(item["architectures"]) > 1 for item in items),
        "investigator_specified": sum(bool(item["investigator_specified"]) for item in items),
        "unattributed": sum(
            not item["architectures"] and not item["investigator_specified"] for item in items
        ),
    }
    return {"by_architecture": counts, "totals": totals}


def attribution_counts(lineage: Stage2Lineage) -> dict[str, Any]:
    """Per fold and total any-source and sole-source counts at every level."""

    architectures = lineage.architectures
    per_fold: dict[str, Any] = {}
    variables: list[dict[str, Any]] = []
    totals: dict[str, Any] = {}
    for fold, fold_lineage in lineage.folds.items():
        items = level_items(fold_lineage)
        fold_result: dict[str, Any] = {}
        membership: dict[str, dict[str, Any]] = {}
        for level in LEVELS:
            values = items[level]
            if values is None:
                fold_result[level] = {
                    "status": "unavailable",
                    "reason": fold_lineage.unavailable.get(level, "artifact is missing"),
                }
                continue
            fold_result[level] = {"status": "ok", **_count(values, architectures)}
            if level in {"cards", "candidates"}:
                continue
            for item in values:
                entry = membership.setdefault(
                    item["id"],
                    {**item["lineage"], **{name: False for name in LEVELS[2:]}},
                )
                entry[level] = True
        per_fold[fold] = fold_result
        for entry in membership.values():
            variables.append({"outer_fold": fold, **entry})
    for level in LEVELS:
        available = [
            fold for fold, result in per_fold.items() if result[level]["status"] == "ok"
        ]
        if not available:
            totals[level] = {"status": "unavailable", "folds": []}
            continue
        by_architecture = {
            architecture: {
                kind: sum(
                    per_fold[fold][level]["by_architecture"][architecture][kind]
                    for fold in available
                )
                for kind in ("any_source", "sole_source")
            }
            for architecture in architectures
        }
        level_totals = {
            key: sum(per_fold[fold][level]["totals"][key] for fold in available)
            for key in ("items", "multi_source", "investigator_specified", "unattributed")
        }
        totals[level] = {
            "status": "ok",
            "folds": available,
            "folds_unavailable": [fold for fold in per_fold if fold not in available],
            "by_architecture": by_architecture,
            "totals": level_totals,
        }
    return {"per_fold": per_fold, "totals": totals, "variables": variables}


# ---------------------------------------------------------------------------
# Oracle recovery


@dataclass(frozen=True)
class OracleConcept:
    name: str
    column: str
    value_type: str
    roles: tuple[str, ...]


@dataclass
class OracleData:
    dataset: Path
    dataset_sha256: str
    metadata: Path | None
    metadata_sha256: str | None
    concepts: list[OracleConcept]
    frame: pd.DataFrame
    absent_concepts: list[str]


def _dataset_columns(path: Path) -> list[str]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    raise ValueError("oracle dataset must be Parquet or CSV")


def _read_dataset_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    wanted = list(dict.fromkeys(columns))
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, columns=wanted)
    else:
        frame = pd.read_csv(path, usecols=lambda column: column in set(wanted))
    frame = frame.reset_index(drop=True)
    frame[ROW_ID] = np.arange(len(frame), dtype=int)
    return frame


def load_oracle_data(
    dataset_path: Path,
    metadata_path: Path | None,
    *,
    readers: Sequence[FrozenArtifactReader],
) -> OracleData:
    """Load oracle columns; every artifact reader must already be frozen."""

    if not readers or any(not reader.frozen for reader in readers):
        raise RuntimeError(
            "hash and freeze every Stage 2 artifact before loading oracle data"
        )
    from .stage1 import _metadata_features

    dataset_path = Path(dataset_path).expanduser().resolve(strict=True)
    metadata_value: Mapping[str, Any] = {}
    if metadata_path is not None:
        metadata_path = Path(metadata_path).expanduser().resolve(strict=True)
        metadata_value = json.loads(metadata_path.read_text(encoding="utf-8"))
    available = _dataset_columns(dataset_path)
    concepts: list[OracleConcept] = []
    absent: list[str] = []
    features = _metadata_features(metadata_value) if metadata_value else []
    if features:
        for feature in features:
            name = str(feature["name"])
            column = f"true_{name}"
            if column not in available:
                absent.append(name)
                continue
            concepts.append(
                OracleConcept(
                    name=name,
                    column=column,
                    value_type=str(feature.get("type") or "categorical").strip().lower(),
                    roles=tuple(sorted(str(role) for role in feature.get("roles") or [])),
                )
            )
    else:
        for column in available:
            # The generator names each variable's truth true_<variable>.
            if not str(column).startswith("true_") or column in EFFECT_ORACLE_COLUMNS:
                continue
            concepts.append(
                OracleConcept(
                    name=str(column)[len("true_"):],
                    column=str(column),
                    value_type="infer",
                    roles=(),
                )
            )
    frame = _read_dataset_columns(dataset_path, [concept.column for concept in concepts])
    resolved: list[OracleConcept] = []
    for concept in concepts:
        if concept.value_type != "infer":
            resolved.append(concept)
            continue
        numeric = pd.to_numeric(frame[concept.column], errors="coerce")
        continuous = bool(numeric.notna().all()) and numeric.nunique(dropna=True) > 10
        resolved.append(
            OracleConcept(
                concept.name,
                concept.column,
                "continuous" if continuous else "categorical",
                concept.roles,
            )
        )
    return OracleData(
        dataset=dataset_path,
        dataset_sha256=file_sha256(dataset_path),
        metadata=metadata_path,
        metadata_sha256=None if metadata_path is None else file_sha256(metadata_path),
        concepts=resolved,
        frame=frame,
        absent_concepts=absent,
    )


def _oracle_targets(values: pd.Series, value_type: str) -> tuple[np.ndarray, np.ndarray]:
    """(target matrix over observed rows, observed-row mask)."""

    if value_type == "continuous":
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        observed = np.isfinite(numeric)
        return numeric[observed].reshape(-1, 1), observed
    observed = values.notna().to_numpy(dtype=bool)
    text = values[observed].astype(str).to_numpy()
    levels = sorted(set(text.tolist()))
    matrix = np.column_stack([(text == level).astype(float) for level in levels]) if levels else (
        np.zeros((0, 0))
    )
    return matrix, observed


def multivariate_r_squared(targets: np.ndarray, design: np.ndarray) -> float | None:
    """1 - sum(SSE) / sum(SST) of OLS (with intercept) over the target columns."""

    from ..inference.stage2_diagnostics import _r_squared

    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[0] < 3 or targets.shape[1] == 0:
        return None
    design = np.nan_to_num(np.asarray(design, dtype=float).reshape(len(targets), -1), nan=0.0)
    explained = 0.0
    total = 0.0
    for column in range(targets.shape[1]):
        target = targets[:, column]
        sst = float(np.sum((target - target.mean()) ** 2))
        if sst <= 0:
            continue
        explained += sst * _r_squared(target, design)
        total += sst
    return None if total <= 0 else float(explained / total)


def _recovered(
    keys: Sequence[str],
    best: Mapping[tuple[str, str], float],
    concept_names: Sequence[str],
    threshold: float,
) -> list[str]:
    """Concepts (in concept order) some variable in ``keys`` recovers."""

    hits = {
        concept
        for key in keys
        for concept in concept_names
        if best.get((key, concept), -1.0) >= threshold
    }
    return [name for name in concept_names if name in hits]


def _gathered(lists: Sequence[Sequence[str]], concept_names: Sequence[str]) -> dict[str, Any]:
    union = {name for values in lists for name in values}
    return {
        "fold_concept_count": sum(len(values) for values in lists),
        "concepts": [name for name in concept_names if name in union],
    }


def oracle_recovery(
    lineage: Stage2Lineage,
    oracle: OracleData,
    *,
    threshold: float = DEFAULT_RECOVERY_R2,
) -> dict[str, Any]:
    """R^2 of every oracle concept on every held-out variable, and recovery by architecture."""

    from ..inference.stage2_diagnostics import encode_measurements

    oracle_by_row = oracle.frame.set_index(ROW_ID)
    r2_rows: list[dict[str, Any]] = []
    per_fold: dict[str, Any] = {}
    concept_names = [concept.name for concept in oracle.concepts]
    for fold, fold_lineage in lineage.folds.items():
        frame = fold_lineage.heldout_frame
        definitions = fold_lineage.heldout_definitions
        if frame is None or definitions is None or ROW_ID not in frame.columns:
            per_fold[fold] = {
                "status": "unavailable",
                "reason": fold_lineage.unavailable.get(
                    "oracle_recovery", "no frozen held-out extraction for this fold"
                ),
            }
            continue
        frame = frame.reset_index(drop=True)
        row_ids = pd.to_numeric(frame[ROW_ID]).astype(int).to_numpy()
        missing_rows = sorted(set(row_ids.tolist()) - set(oracle_by_row.index.tolist()))
        if missing_rows:
            raise ValueError(
                f"fold {fold} held-out rows {missing_rows[:5]} are outside the oracle dataset; "
                "evaluate with the dataset file the run used"
            )
        encoded, meta = encode_measurements(frame, definitions)
        columns_by_feature: dict[str, list[str]] = {}
        for item in meta:
            columns_by_feature.setdefault(str(item["feature_id"]), []).append(str(item["column"]))
        traced = {
            str(feature.get("feature_id") or feature.get("name")): variable_lineage(
                feature, fold_lineage
            )
            for feature in definitions
        }
        selected_roles = {
            str(feature.get("feature_id") or feature.get("name")): _roles(feature)
            for feature in fold_lineage.final_features or []
        }
        targets_by_concept = {
            concept.name: _oracle_targets(
                oracle_by_row.loc[row_ids, concept.column].reset_index(drop=True),
                concept.value_type,
            )
            for concept in oracle.concepts
        }
        best: dict[tuple[str, str], float] = {}
        for feature in definitions:
            key = str(feature.get("feature_id") or feature.get("name"))
            columns = columns_by_feature.get(key, [])
            design = encoded[columns].to_numpy(dtype=float) if columns else np.zeros((len(frame), 0))
            roles = selected_roles.get(key, set())
            for concept in oracle.concepts:
                targets, observed = targets_by_concept[concept.name]
                r2 = multivariate_r_squared(targets, design[observed]) if columns else None
                if r2 is not None:
                    best[(key, concept.name)] = r2
                r2_rows.append(
                    {
                        "outer_fold": fold,
                        "feature_id": key,
                        "name": str(feature.get("name") or key),
                        "concept": concept.name,
                        "oracle_column": concept.column,
                        "concept_roles": ";".join(concept.roles),
                        "r_squared": r2,
                        "rows": int(observed.sum()),
                        "design_columns": len(columns),
                        "recovered": bool(r2 is not None and r2 >= threshold),
                        "nuisance_set": "confounder" in roles,
                        "selected_modifier": "effect_modifier" in roles,
                        "architectures": ";".join(traced[key]["architectures"]),
                        "sole_source_architecture": traced[key]["sole_source_architecture"] or "",
                    }
                )
        levels = {
            "selected_variables": [k for k, roles in selected_roles.items() if roles],
            "nuisance_set": [k for k, roles in selected_roles.items() if "confounder" in roles],
            "selected_modifiers": [
                k for k, roles in selected_roles.items() if "effect_modifier" in roles
            ],
        }
        missing_selected = sorted(
            key for keys in levels.values() for key in keys if key not in traced
        )
        fold_result: dict[str, Any] = {
            "status": "ok",
            "heldout_source": fold_lineage.heldout_source,
            "heldout_rows": len(frame),
            "selected_variables_missing_from_heldout_extraction": sorted(set(missing_selected)),
            "levels": {},
        }
        for level, keys in levels.items():
            keys = [key for key in keys if key in traced]
            fold_result["levels"][level] = {
                "all_variables": _recovered(keys, best, concept_names, threshold),
                "by_architecture": {
                    architecture: {
                        "any_source": _recovered(
                            [key for key in keys if architecture in traced[key]["architectures"]],
                            best,
                            concept_names,
                            threshold,
                        ),
                        "sole_source": _recovered(
                            [
                                key
                                for key in keys
                                if traced[key]["sole_source_architecture"] == architecture
                            ],
                            best,
                            concept_names,
                            threshold,
                        ),
                    }
                    for architecture in lineage.architectures
                },
            }
        per_fold[fold] = fold_result
    ok_folds = [fold for fold, result in per_fold.items() if result["status"] == "ok"]
    totals: dict[str, Any] = {}
    for level in RECOVERY_LEVELS:
        sections = [per_fold[fold]["levels"][level] for fold in ok_folds]
        totals[level] = {
            "all_variables": _gathered(
                [section["all_variables"] for section in sections], concept_names
            ),
            "by_architecture": {
                architecture: {
                    kind: _gathered(
                        [section["by_architecture"][architecture][kind] for section in sections],
                        concept_names,
                    )
                    for kind in ("any_source", "sole_source")
                }
                for architecture in lineage.architectures
            },
        }
    return {
        "status": "ok" if ok_folds else "unavailable",
        "threshold_r_squared": float(threshold),
        "method": RECOVERY_RULE,
        "concepts": [
            {
                "name": concept.name,
                "oracle_column": concept.column,
                "value_type": concept.value_type,
                "roles": list(concept.roles),
            }
            for concept in oracle.concepts
        ],
        "metadata_concepts_without_oracle_column": list(oracle.absent_concepts),
        "folds": ok_folds,
        "per_fold": per_fold,
        "totals": totals,
        "r_squared": r2_rows,
    }


# ---------------------------------------------------------------------------
# Report


def _run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _resolve_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path)


def resolve_oracle_inputs(
    run_dir: Path,
    *,
    dataset: Path | None = None,
    metadata: Path | None = None,
) -> tuple[Path, Path | None]:
    """The oracle-bearing dataset and DGP metadata of a run or ablation directory."""

    run_dir = Path(run_dir)
    ablation_path = run_dir / "ablation.json"
    ablation = (
        json.loads(ablation_path.read_text(encoding="utf-8")) if ablation_path.is_file() else {}
    )
    if isinstance(ablation, Mapping) and ablation.get("source_run_dir"):
        # An ablation shares its source run's rows; read that run's oracle inputs.
        return resolve_oracle_inputs(
            Path(str(ablation["source_run_dir"])), dataset=dataset, metadata=metadata
        )
    config = _run_config(run_dir)
    source_dataset = config.get("dataset")
    if dataset is None:
        cohort = run_dir / ANALYSIS_COHORT
        if cohort.is_file():
            dataset = cohort
        elif source_dataset:
            dataset = _resolve_path(source_dataset, run_dir)
        else:
            raise ValueError("no oracle dataset is recorded for this run; pass --dataset")
    dataset = Path(dataset).expanduser().resolve(strict=True)
    if metadata is None:
        candidates = []
        if source_dataset:
            candidates.append(_resolve_path(source_dataset, run_dir).parent / "metadata.json")
        candidates.append(dataset.parent / "metadata.json")
        metadata = next((path for path in candidates if path.is_file()), None)
    return dataset, (None if metadata is None else Path(metadata).expanduser().resolve())


def _count_rows(result: Mapping[str, Any], architectures: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sections = [
        *((fold, value) for fold, value in result["per_fold"].items()),
        ("all", result["totals"]),
    ]
    for fold, section in sections:
        for level in LEVELS:
            entry = section[level]
            if entry.get("status") != "ok":
                rows.append({"outer_fold": fold, "level": level, "status": "unavailable"})
                continue
            for architecture in architectures:
                counts = entry["by_architecture"][architecture]
                rows.append(
                    {
                        "outer_fold": fold,
                        "level": level,
                        "status": "ok",
                        "architecture": architecture,
                        "any_source": counts["any_source"],
                        "sole_source": counts["sole_source"],
                        "level_items": entry["totals"]["items"],
                        "level_investigator_specified": entry["totals"][
                            "investigator_specified"
                        ],
                        "level_unattributed": entry["totals"]["unattributed"],
                    }
                )
    return rows


COUNT_COLUMNS = (
    "outer_fold",
    "level",
    "status",
    "architecture",
    "any_source",
    "sole_source",
    "level_items",
    "level_investigator_specified",
    "level_unattributed",
)
VARIABLE_COLUMNS = (
    "outer_fold",
    "feature_id",
    "name",
    "kind",
    "architectures",
    "sole_source_architecture",
    "investigator_specified",
    "variables_defined",
    "variables_passing_filter",
    "nuisance_set",
    "selected_modifiers",
    "candidate_ids",
    "card_count",
    "unresolved_card_ids",
)
RECOVERY_COLUMNS = (
    "outer_fold",
    "level",
    "architecture",
    "kind",
    "recovered_count",
    "recovered_concepts",
)
R2_COLUMNS = (
    "outer_fold",
    "feature_id",
    "name",
    "concept",
    "oracle_column",
    "concept_roles",
    "r_squared",
    "rows",
    "design_columns",
    "recovered",
    "nuisance_set",
    "selected_modifier",
    "architectures",
    "sole_source_architecture",
)


def _variable_rows(variables: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "outer_fold": item["outer_fold"],
            "feature_id": item["feature_id"],
            "name": item["name"],
            "kind": item["kind"],
            "architectures": ";".join(item["architectures"]),
            "sole_source_architecture": item["sole_source_architecture"] or "",
            "investigator_specified": item["investigator_specified"],
            "variables_defined": item["variables_defined"],
            "variables_passing_filter": item["variables_passing_filter"],
            "nuisance_set": item["nuisance_set"],
            "selected_modifiers": item["selected_modifiers"],
            "candidate_ids": ";".join(item["candidate_ids"]),
            "card_count": len(item["card_ids"]),
            "unresolved_card_ids": ";".join(item["unresolved_card_ids"]),
        }
        for item in variables
    ]


def recovery_rows(recovery: Mapping[str, Any], architectures: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, section in recovery["per_fold"].items():
        if section.get("status") != "ok":
            continue
        for level, entry in section["levels"].items():
            rows.append(
                {
                    "outer_fold": fold,
                    "level": level,
                    "architecture": "all",
                    "kind": "any_variable",
                    "recovered_count": len(entry["all_variables"]),
                    "recovered_concepts": ";".join(entry["all_variables"]),
                }
            )
            for architecture in architectures:
                for kind in ("any_source", "sole_source"):
                    names = entry["by_architecture"][architecture][kind]
                    rows.append(
                        {
                            "outer_fold": fold,
                            "level": level,
                            "architecture": architecture,
                            "kind": kind,
                            "recovered_count": len(names),
                            "recovered_concepts": ";".join(names),
                        }
                    )
    for level, entry in recovery["totals"].items():
        rows.append(
            {
                "outer_fold": "all",
                "level": level,
                "architecture": "all",
                "kind": "any_variable",
                "recovered_count": entry["all_variables"]["fold_concept_count"],
                "recovered_concepts": ";".join(entry["all_variables"]["concepts"]),
            }
        )
        for architecture in architectures:
            for kind in ("any_source", "sole_source"):
                values = entry["by_architecture"][architecture][kind]
                rows.append(
                    {
                        "outer_fold": "all",
                        "level": level,
                        "architecture": architecture,
                        "kind": kind,
                        "recovered_count": values["fold_concept_count"],
                        "recovered_concepts": ";".join(values["concepts"]),
                    }
                )
    return rows


def evaluate_attribution(
    *,
    run_dir: Path,
    oracle: bool = False,
    output_dir: Path | None = None,
    recovery_r2: float = DEFAULT_RECOVERY_R2,
    dataset_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Write the attribution report for a workflow output or ablation directory."""

    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    if not 0.0 <= float(recovery_r2) <= 1.0:
        raise ValueError("--recovery-r2 must be between 0 and 1")
    stage2_dir = run_dir / "stage2"
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else run_dir / "evaluation" / "attribution"
    )
    assert_output_outside_run_inputs(output_dir, run_dir)
    if oracle:
        # Resolve (but do not open) the oracle inputs first so a missing
        # dataset fails before any work.
        dataset, metadata = resolve_oracle_inputs(
            run_dir, dataset=dataset_path, metadata=metadata_path
        )
    lineage = collect_stage2_lineage(stage2_dir, include_heldout=oracle)
    result = attribution_counts(lineage)
    # Every Stage 2 artifact the report uses has now been read and hashed.
    artifact_hashes = lineage.reader.freeze()
    recovery: dict[str, Any] = {"status": "not_requested"}
    oracle_manifest: dict[str, Any] = {"requested": False}
    if oracle:
        oracle_data = load_oracle_data(dataset, metadata, readers=[lineage.reader])
        recovery = oracle_recovery(lineage, oracle_data, threshold=recovery_r2)
        oracle_manifest = {
            "requested": True,
            "dataset": str(oracle_data.dataset),
            "dataset_sha256": oracle_data.dataset_sha256,
            "metadata": None if oracle_data.metadata is None else str(oracle_data.metadata),
            "metadata_sha256": oracle_data.metadata_sha256,
            "oracle_columns": [concept.column for concept in oracle_data.concepts],
            "stage2_artifacts_hashed_before_oracle_load": True,
        }
    report = {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "stage2_dir": str(stage2_dir),
        "architectures": lineage.architectures,
        "architecture_source": lineage.architecture_source,
        "levels": list(LEVELS),
        "lineage_rule": LINEAGE_RULE,
        "level_definitions": {
            "cards": "evidence cards compiled for the fold",
            "candidates": "candidates proposed by interpretation",
            "variables_defined": "variables in feature_definitions.json",
            "variables_passing_filter": (
                "variables kept by the informativeness and extraction-failure filter"
            ),
            "nuisance_set": "selected variables with the confounder role",
            "selected_modifiers": "selected variables with the effect_modifier role",
        },
        "folds": list(lineage.folds),
        "per_fold": result["per_fold"],
        "totals": result["totals"],
        "oracle_recovery": {
            key: value for key, value in recovery.items() if key != "r_squared"
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "attribution.json", report)
    write_csv(
        output_dir / "attribution_counts.csv",
        _count_rows(result, lineage.architectures),
        COUNT_COLUMNS,
    )
    write_csv(output_dir / "variables.csv", _variable_rows(result["variables"]), VARIABLE_COLUMNS)
    outputs = ["attribution.json", "attribution_counts.csv", "variables.csv"]
    if oracle:
        write_csv(
            output_dir / "oracle_recovery.csv",
            recovery_rows(recovery, lineage.architectures),
            RECOVERY_COLUMNS,
        )
        write_csv(output_dir / "oracle_recovery_r2.csv", recovery["r_squared"], R2_COLUMNS)
        outputs.extend(["oracle_recovery.csv", "oracle_recovery_r2.csv"])
    manifest = {
        "schema_version": ATTRIBUTION_MANIFEST_SCHEMA_VERSION,
        "created_at": _now(),
        "run_dir": str(run_dir),
        "stage2_dir": str(stage2_dir),
        "stage2_artifact_sha256": artifact_hashes,
        "oracle": oracle_manifest,
        "recovery_threshold_r_squared": float(recovery_r2) if oracle else None,
        "outputs": outputs,
        "report_does_not_modify_run_artifacts": True,
    }
    write_json(output_dir / "attribution_manifest.json", manifest)
    return {
        "output_dir": str(output_dir),
        "architectures": lineage.architectures,
        "folds": list(lineage.folds),
        "outputs": outputs,
        "oracle_recovery_status": recovery.get("status"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Credit frozen Stage 2 variables to the Stage 1 architectures that supported "
            "them, optionally with oracle recovery on synthetic cohorts."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="workflow output or ablation directory containing stage2/")
    parser.add_argument("--oracle", action="store_true",
                        help="add oracle recovery (synthetic cohorts with true_* columns)")
    parser.add_argument("--out", type=Path, help="output directory "
                        "(default RUN_DIR/evaluation/attribution)")
    parser.add_argument("--recovery-r2", type=float, default=DEFAULT_RECOVERY_R2,
                        help="held-out R^2 at which a variable recovers an oracle concept")
    parser.add_argument("--dataset", type=Path, help="oracle-bearing dataset (with --oracle)")
    parser.add_argument("--metadata", type=Path, help="generator metadata.json (with --oracle)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_attribution(
        run_dir=args.run_dir,
        oracle=args.oracle,
        output_dir=args.out,
        recovery_r2=args.recovery_r2,
        dataset_path=args.dataset,
        metadata_path=args.metadata,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTRIBUTION_MANIFEST_SCHEMA_VERSION",
    "ATTRIBUTION_SCHEMA_VERSION",
    "DEFAULT_RECOVERY_R2",
    "FrozenArtifactReader",
    "LEVELS",
    "OracleConcept",
    "OracleData",
    "Stage2Lineage",
    "attribution_counts",
    "build_parser",
    "collect_stage2_lineage",
    "evaluate_attribution",
    "load_oracle_data",
    "main",
    "multivariate_r_squared",
    "oracle_recovery",
    "resolve_oracle_inputs",
    "variable_lineage",
]
