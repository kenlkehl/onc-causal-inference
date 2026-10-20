"""Run the shared causal estimators on declared table columns, without an LLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping
import concurrent.futures
import json
import re

import numpy as np
import pandas as pd

INPUT_FILE = "tabular_inputs.json"


def _category(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            raise ValueError("tabular categories must be finite")
        return str(int(value)) if float(value).is_integer() else str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("tabular categories must be nonempty strings or finite numbers")
    return value.strip()


@dataclass(frozen=True)
class TabularFeature:
    column: str
    value_type: str
    roles: tuple[str, ...]
    categories: tuple[str, ...] = ()
    description: str = ""

    @property
    def name(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.column.lower()).strip("_")

    def definition(self, outer_fold: int) -> dict[str, Any]:
        return {
            "feature_id": f"outer_{outer_fold:03d}_tabular_{self.name}",
            "name": self.name,
            "source_column": self.column,
            "description": self.description or self.column,
            "value_type": self.value_type,
            "categories_or_unit": list(self.categories),
            "roles": list(self.roles),
            "nuisance_model_roles": (
                ["treatment", "outcome"] if "confounder" in self.roles else ["outcome"]
            ),
            "configured_explicit_feature": True,
            "measurement_source": "tabular",
            **({"modeling_strategy": "continuous"} if self.value_type == "continuous" else {}),
        }


@dataclass(frozen=True)
class TabularConfig:
    features: tuple[TabularFeature, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def tabular_config_from_mapping(raw: Any) -> TabularConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != {"features"}:
        raise ValueError("tabular must be an object containing only a features list")
    entries = raw["features"]
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError("tabular.features must be a nonempty list")
    features = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise ValueError("each tabular feature must be an object")
        unknown = set(item) - {"column", "value_type", "roles", "categories", "description"}
        if unknown:
            raise ValueError(f"unsupported tabular feature fields: {sorted(unknown)}")
        column = item.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ValueError("each tabular feature requires a nonempty column")
        kind = item.get("value_type")
        if kind not in {"continuous", "binary", "categorical", "ordinal"}:
            raise ValueError(
                "tabular value_type must be continuous, binary, categorical, or ordinal"
            )
        roles = item.get("roles")
        if isinstance(roles, str):
            roles = [roles]
        if not isinstance(roles, (list, tuple)) or not roles:
            raise ValueError(f"tabular column {column!r} requires causal roles")
        roles = tuple(
            dict.fromkeys(
                role
                for value in roles
                for role in (("confounder", "effect_modifier") if value == "both" else (value,))
            )
        )
        if set(roles) - {"confounder", "effect_modifier"}:
            raise ValueError(f"unsupported roles for tabular column {column!r}")
        categories = item.get("categories", [0, 1] if kind == "binary" else [])
        if not isinstance(categories, (list, tuple)):
            raise ValueError("tabular categories must be a list")
        categories = tuple(_category(value) for value in categories)
        if len(set(categories)) != len(categories):
            raise ValueError(f"duplicate categories for tabular column {column!r}")
        if any(value.casefold() in {"__missing__", "documented_unknown"} for value in categories):
            raise ValueError(
                "tabular categories cannot use reserved missing-state labels; use null"
            )
        if kind == "continuous" and categories:
            raise ValueError("continuous tabular features must not declare categories")
        if kind == "binary" and len(categories) != 2:
            raise ValueError("binary tabular features require exactly two categories")
        if kind in {"ordinal", "categorical"} and len(categories) < 2:
            raise ValueError("ordinal and categorical tabular features require declared categories")
        feature = TabularFeature(
            column, kind, roles, categories, str(item.get("description") or "")
        )
        if not feature.name or feature.name == "oci_row_id":
            raise ValueError(f"invalid or reserved tabular feature name: {column!r}")
        if feature.name.startswith(("true_", "oracle_", "ground_truth")):
            raise ValueError("oracle columns cannot be declared as tabular covariates")
        features.append(feature)
    if len({feature.name for feature in features}) != len(features):
        raise ValueError("tabular columns must have distinct normalized names")
    return TabularConfig(tuple(features))


def prepare_tabular_inputs(config: Any, dataset: pd.DataFrame, *, persist: bool):
    """Validate supplied values and freeze the input identity and nested splits."""
    from .plain_handoff_stage2_analysis import _frame_fingerprint
    from .stage2_explicit_inputs import freeze_nested_splits

    data = dataset.reset_index(drop=True)
    declared = config.tabular.features
    labels = [config.unit_id_column, config.treatment_column, config.outcome_column]
    columns = [feature.column for feature in declared]
    if len(set(labels)) != len(labels) or set(labels) & set(columns):
        raise ValueError("tabular covariates must be distinct from unit ID, treatment, and outcome")
    missing = sorted(set(labels + columns) - set(data.columns))
    if missing:
        raise ValueError(f"tabular dataset is missing declared columns: {missing}")
    if data[config.unit_id_column].isna().any() or data[config.unit_id_column].duplicated().any():
        raise ValueError("tabular unit IDs must be nonmissing and unique")
    treatment = pd.to_numeric(data[config.treatment_column], errors="coerce")
    outcome = pd.to_numeric(data[config.outcome_column], errors="coerce")
    if treatment.isna().any() or set(treatment.unique()) != {0, 1}:
        raise ValueError("tabular treatment must be complete 0/1 with both arms present")
    if not np.isfinite(outcome.to_numpy(dtype=float)).all():
        raise ValueError("tabular outcome must be complete, finite, and numeric")
    if config.outcome_type not in {"binary", "continuous"}:
        raise ValueError("tabular outcome_type must be binary or continuous")
    if config.outcome_type == "binary" and not set(outcome.unique()) <= {0, 1}:
        raise ValueError("tabular binary outcome must contain only 0 and 1")
    if config.outer_folds < 2 or config.inner_folds < 2:
        raise ValueError("tabular analyses require outer_folds and inner_folds >= 2")
    policy = config.stage2.statistical_selection.multi_model.modifier_selection
    if config.outcome_type != "binary" and policy.resolved_estimator_policy == "dina":
        raise ValueError("DINA requires a binary outcome")

    measurements = pd.DataFrame({"_oci_row_id": np.arange(len(data))})
    for feature in declared:
        values = data[feature.column]
        if feature.value_type == "continuous":
            numeric = pd.to_numeric(values, errors="coerce")
            if (values.notna() & numeric.isna()).any() or np.isinf(
                numeric.to_numpy(dtype=float)
            ).any():
                raise ValueError(
                    f"tabular continuous column {feature.column!r} has nonnumeric or infinite values"
                )
            measurements[feature.name] = numeric.astype(float)
        else:
            normalized = values.map(lambda value: None if pd.isna(value) else _category(value))
            unknown = sorted(set(normalized.dropna()) - set(feature.categories))
            if unknown:
                raise ValueError(
                    f"tabular column {feature.column!r} has undeclared categories: {unknown[:8]}"
                )
            measurements[feature.name] = normalized
    identity = {
        "schema_version": "stage2_tabular_inputs_v1",
        "dataset_fingerprint": _frame_fingerprint(data[labels + columns]),
        "rows": len(data),
        "columns": labels + columns,
        "outcome_type": config.outcome_type,
        "clinical_question": config.clinical_question,
        "outer_folds": config.outer_folds,
        "inner_folds": config.inner_folds,
        "seed": config.seed,
        "features": config.tabular.as_dict()["features"],
    }
    # JSON-native values give the saved and in-memory representations identical types.
    identity = json.loads(json.dumps(identity))
    manifest = freeze_nested_splits(
        dataset=data,
        output_dir=config.output_dir / "stage2",
        identity=identity,
        treatment_column=config.treatment_column,
        outcome_column=config.outcome_column,
        filename=INPUT_FILE,
        mode="tabular",
        persist=persist,
    )
    return measurements, manifest


def run_tabular_stage2(config: Any, dataset: pd.DataFrame) -> Mapping[str, Any]:
    from .plain_handoff_stage2 import _write_json, summarize_stage2_results
    from .plain_handoff_stage2_analysis import estimate_outer_fold, _write_frame
    from .stage2_modeling_pool import modeling_session
    from .stage2_sensitivity import benchmark_fold

    output_dir = config.output_dir / "stage2"
    measurements, manifest = prepare_tabular_inputs(config, dataset, persist=True)
    stage2 = config.stage2
    _write_json(output_dir / "config.json", stage2.public_dict())
    _write_frame(output_dir / "measurements.csv", measurements)
    splits = manifest["splits"]
    workers = min(stage2.workers, len(splits))
    estimator = (
        stage2.statistical_selection.multi_model.modifier_selection.resolved_estimator_policy
    )

    def run_fold(split):
        fold = split["outer_fold"]
        directory = output_dir / f"outer_{fold:03d}"
        definitions = [feature.definition(fold) for feature in config.tabular.features]
        fit = measurements.iloc[split["fit_row_ids"]].reset_index(drop=True)
        heldout = measurements.iloc[split["heldout_row_ids"]].reset_index(drop=True)
        _write_json(
            directory / "final_definitions.json",
            {
                "schema_version": "stage2_tabular_definitions_v1",
                "features": definitions,
                "final_estimator": estimator,
                "selection": "fixed_investigator_specified_roles",
                "measurement_source": "tabular",
            },
        )
        _write_frame(directory / "measurements" / "fit.csv", fit)
        _write_frame(directory / "measurements" / "heldout.csv", heldout)
        seed = config.seed + 100_000 * fold
        diagnostics = estimate_outer_fold(
            dataset=dataset,
            extracted_fit=fit,
            extracted_heldout=heldout,
            definitions=definitions,
            split=split,
            unit_id_column=config.unit_id_column,
            treatment_column=config.treatment_column,
            outcome_column=config.outcome_column,
            outcome_type=config.outcome_type,
            inner_folds=config.inner_folds,
            seed=seed,
            propensity_clip=stage2.propensity_clip,
            estimation_trees=stage2.estimation_trees,
            min_propensity=stage2.min_propensity,
            max_propensity=stage2.max_propensity,
            estimator=estimator,
            statistical_policy=stage2.statistical_selection,
            output_dir=directory / "estimation",
            nuisance_candidate_definitions=definitions,
            nuisance_candidate_fit=fit,
            nuisance_candidate_heldout=heldout,
            nuisance_learner=stage2.nuisance_learner_spec(),
        )
        benchmark_fold(
            config=stage2,
            estimation_dir=directory / "estimation",
            dataset=dataset,
            candidate_definitions=definitions,
            candidate_fit=fit,
            candidate_heldout=heldout,
            selected_definitions=definitions,
            split=split,
            treatment_column=config.treatment_column,
            outcome_column=config.outcome_column,
            outcome_type=config.outcome_type,
            seed=seed,
            propensity_clip=stage2.propensity_clip,
        )
        result = {"outer_fold": fold, "features": definitions, "estimation": diagnostics}
        _write_json(
            directory / "complete.json",
            {"status": "complete", "phase": "causal_estimation", **result},
        )
        return result

    with modeling_session(stage2.modeling_workers, parallel_fits=workers):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run_fold, splits))
    return summarize_stage2_results(
        config=stage2,
        dataset=dataset,
        fold_results=results,
        output_dir=output_dir,
        outcome_type=config.outcome_type,
        seed=config.seed,
        source_summary={
            "candidate_discovery_source": "tabular",
            "measurement_source": "supplied_columns",
            "selection": "fixed_investigator_specified_roles",
            "split_provenance_path": str(output_dir / INPUT_FILE),
            "measurements_path": str(output_dir / "measurements.csv"),
            "llm_requests": 0,
        },
    )
