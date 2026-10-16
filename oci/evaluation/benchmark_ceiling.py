"""Oracle-covariate ceiling for the benchmark harness.

The ceiling refits a finished run's final Stage 2 estimation with the true
covariates in place of the LLM-extracted variables. It calls
:func:`oci.inference.plain_handoff_stage2_analysis.estimate_outer_fold`, the
function every Stage 2 outer fold calls after selection, with:

* the run's frozen outer and inner folds (``components/tfidf/split_provenance.jsonl``);
* the run's fold seeds (``science.seed + 100000 * outer_fold``), inner-fold
  count, propensity clip and bounds, forest size, statistical-selection policy
  and nuisance learner (``stage2/config.json``);
* each fold's recorded final estimator (``final_definitions.json``), which is
  the estimator ensemble under the default ``orthogonal_tests`` rule;
* in place of the extracted measurements, each patient's ``true_<variable>``
  values.

The covariate design follows the data-generating roles. The nuisance-candidate
catalog holds every metadata feature with a role (confounder, effect modifier,
instrument or prognostic); unassigned features, proxies and unmeasured
confounders are left out. The selected variables carry the Stage 2 roles a
perfect selector would give them: ``confounder`` for true confounders and
prognostic-only variables, ``effect_modifier`` for true effect modifiers (both
for dual-role variables). Instruments enter the nuisance models only.

No LLM is called and nothing is written inside the run directory. The ceiling
runs only after every run artifact has been hashed and the reader frozen,
because its inputs are oracle columns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

CEILING_SCHEMA_VERSION = "oci_benchmark_oracle_ceiling_v1"
CEILING_DGP_ROLES = ("confounder", "effect_modifier", "instrument", "prognostic")
FOLD_SEED_STRIDE = 100_000
ROW_ID = "_oci_row_id"
CEILING_DESIGN_RULE = (
    "Nuisance-candidate catalog: every metadata feature with a data-generating role "
    "(confounder, effect_modifier, instrument, prognostic), measured by its true_<variable> "
    "column; unassigned features, proxies and unmeasured confounders are excluded. Selected "
    "variables: confounder role for true confounders and prognostic-only variables, "
    "effect_modifier role for true effect modifiers (both for dual-role variables); "
    "instruments enter the nuisance models only. The estimation code path is Stage 2's "
    "estimate_outer_fold with the run's folds, fold seeds, nuisance learner, statistical "
    "policy, propensity settings, forest size and each fold's final estimator."
)


def _sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ceiling_definitions(
    features: Sequence[Mapping[str, Any]],
    values: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(selected definitions with Stage 2 roles, nuisance-candidate catalog).

    ``features`` are metadata features (``name``, ``type``, ``categories``,
    ``roles``); ``values`` holds their ``true_<name>`` columns.
    """

    catalog: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for feature in features:
        name = str(feature.get("name") or "").strip()
        dgp_roles = {str(role) for role in feature.get("roles") or []}
        column = f"true_{name}"
        if not name or not dgp_roles.intersection(CEILING_DGP_ROLES) or column not in values:
            continue
        stage2_roles = []
        if dgp_roles.intersection({"confounder", "prognostic"}):
            stage2_roles.append("confounder")
        if "effect_modifier" in dgp_roles:
            stage2_roles.append("effect_modifier")
        continuous = str(feature.get("type") or "").strip().lower() == "continuous"
        definition: dict[str, Any] = {
            "feature_id": f"ceiling_feature_{len(catalog) + 1:03d}",
            "name": name,
            "description": str(feature.get("description") or name),
            "value_type": "continuous" if continuous else "categorical",
            "roles": stage2_roles,
            "data_generating_roles": sorted(dgp_roles),
        }
        if continuous:
            definition["modeling_strategy"] = "continuous"
        else:
            declared = [str(level) for level in feature.get("categories") or []]
            observed = sorted(
                {str(value) for value in values[column].dropna().tolist()} - set(declared)
            )
            definition["categories_or_unit"] = [*declared, *observed]
        catalog.append(definition)
        if stage2_roles:
            selected.append(definition)
    return selected, catalog


def _measurements(
    values: pd.DataFrame, definitions: Sequence[Mapping[str, Any]], row_ids: Sequence[int]
) -> pd.DataFrame:
    """True values as a Stage 2 measurement frame in ``row_ids`` order."""

    frame = pd.DataFrame({ROW_ID: [int(row) for row in row_ids]})
    for definition in definitions:
        column = values[f"true_{definition['name']}"].iloc[list(map(int, row_ids))]
        if definition["value_type"] == "continuous":
            frame[definition["name"]] = pd.to_numeric(column, errors="coerce").to_numpy(float)
        else:
            frame[definition["name"]] = [
                None if pd.isna(value) else str(value) for value in column.tolist()
            ]
    return frame


def ceiling_settings(
    stage2_config: Mapping[str, Any], run_config: Mapping[str, Any]
) -> dict[str, Any]:
    """The run's estimation settings, parsed with Stage 2's own parsers and defaults."""

    from ..inference.plain_handoff_stage2 import PlainHandoffStage2Config
    from ..inference.stage2_elastic_net_selection import (
        statistical_selection_config_from_mapping,
    )
    from ..models.super_learner_nuisance import (
        learner_spec,
        super_learner_config_from_mapping,
    )

    fields = PlainHandoffStage2Config.__dataclass_fields__

    def value(key: str) -> Any:
        return stage2_config[key] if key in stage2_config else fields[key].default

    policy = statistical_selection_config_from_mapping(
        stage2_config.get("statistical_selection") or {}
    )
    min_propensity, max_propensity = value("min_propensity"), value("max_propensity")
    policy = replace(policy, min_propensity=min_propensity, max_propensity=max_propensity)
    return {
        "policy": policy,
        "learner": learner_spec(
            str(value("nuisance_learner")),
            super_learner_config_from_mapping(stage2_config.get("super_learner")),
        ),
        "propensity_clip": float(value("propensity_clip")),
        "min_propensity": min_propensity,
        "max_propensity": max_propensity,
        "estimation_trees": int(value("estimation_trees")),
        "lead_estimand": str(value("lead_estimand")),
        "seed": int(run_config.get("seed", 42)),
        "inner_folds": int(run_config.get("inner_folds", 5)),
        "outcome_type": str(run_config.get("outcome_type", "binary")),
        "unit_id_column": str(run_config.get("unit_id_column", "patient_id")),
        "treatment_column": str(run_config.get("treatment_column", "treatment_indicator")),
        "outcome_column": str(run_config.get("outcome_column", "outcome_indicator")),
    }


def default_estimator(policy: Any) -> str:
    """The estimator Stage 2 applies when a fold records none."""

    selection = policy.multi_model.modifier_selection
    if selection.rule == "orthogonal_tests":
        return str(selection.resolved_estimator_policy)
    return "causal_forest"


def run_oracle_ceiling(
    *,
    dataset: pd.DataFrame,
    features: Sequence[Mapping[str, Any]],
    splits: Mapping[int, Mapping[str, Any]],
    settings: Mapping[str, Any],
    final_estimators: Mapping[int, str | None],
    output_root: Path,
    identity_inputs: Mapping[str, Any],
    workers: int = 1,
) -> dict[str, Any]:
    """Fit the Stage 2 final estimation on the true covariates for every outer fold.

    ``dataset`` is the run's analysis dataset in row order (with the
    ``true_<variable>`` columns); ``splits`` maps each outer fold to its
    ``fit_row_ids``, ``heldout_row_ids`` and ``inner_splits``.
    """

    from ..inference import stage2_modeling_pool as modeling
    from ..inference.plain_handoff_stage2_analysis import estimate_outer_fold
    from ..inference.stage2_diagnostics import (
        build_heterogeneity_diagnostics,
        co_primary_estimands,
    )

    selected, catalog = ceiling_definitions(features, dataset)
    if not catalog:
        return {"status": "unavailable", "reason": "no metadata feature with a role has a true_ column"}
    columns = [settings["unit_id_column"], settings["treatment_column"], settings["outcome_column"]]
    labels = dataset[columns].reset_index(drop=True)
    estimators = {
        int(fold): str(final_estimators.get(int(fold)) or default_estimator(settings["policy"]))
        for fold in splits
    }
    identity = {
        "schema_version": CEILING_SCHEMA_VERSION,
        "design_rule": CEILING_DESIGN_RULE,
        "selected": selected,
        "catalog": catalog,
        "estimators": {str(fold): name for fold, name in sorted(estimators.items())},
        "policy": settings["policy"].public_dict(),
        "learner": settings["learner"],
        **{
            key: settings[key]
            for key in (
                "propensity_clip", "min_propensity", "max_propensity", "estimation_trees",
                "lead_estimand", "seed", "inner_folds", "outcome_type", "unit_id_column",
                "treatment_column", "outcome_column",
            )
        },
        **dict(identity_inputs),
    }
    fingerprint = _sha256_json(identity)
    root = Path(output_root) / fingerprint[:16]
    root.mkdir(parents=True, exist_ok=True)
    (root / "ceiling_identity.json").write_text(
        json.dumps({**identity, "fingerprint": fingerprint}, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )

    def fit_all() -> list[pd.DataFrame]:
        frames = []
        for fold in sorted(splits):
            split = splits[fold]
            fit_ids = [int(row) for row in split["fit_row_ids"]]
            heldout_ids = [int(row) for row in split["heldout_row_ids"]]
            fit_frame = _measurements(dataset, catalog, fit_ids)
            heldout_frame = _measurements(dataset, catalog, heldout_ids)
            selected_names = ["_oci_row_id", *(item["name"] for item in selected)]
            estimate_outer_fold(
                dataset=labels,
                extracted_fit=fit_frame[selected_names],
                extracted_heldout=heldout_frame[selected_names],
                definitions=selected,
                split=dict(split),
                unit_id_column=settings["unit_id_column"],
                treatment_column=settings["treatment_column"],
                outcome_column=settings["outcome_column"],
                outcome_type=settings["outcome_type"],
                inner_folds=settings["inner_folds"],
                seed=settings["seed"] + FOLD_SEED_STRIDE * int(fold),
                propensity_clip=settings["propensity_clip"],
                min_propensity=settings["min_propensity"],
                max_propensity=settings["max_propensity"],
                estimation_trees=settings["estimation_trees"],
                estimator=estimators[int(fold)],
                statistical_policy=settings["policy"],
                output_dir=root / f"outer_{int(fold):03d}" / "estimation",
                nuisance_candidate_definitions=catalog,
                nuisance_candidate_fit=fit_frame,
                nuisance_candidate_heldout=heldout_frame,
                nuisance_model_store=None,
                nuisance_learner=settings["learner"],
            )
            frame = pd.read_csv(root / f"outer_{int(fold):03d}" / "estimation" / "predictions.csv")
            frame.insert(1, "outer_fold", int(fold))
            frames.append(frame)
        return frames

    if int(workers) > 1:
        with modeling.modeling_session(int(workers)):
            frames = fit_all()
    else:
        frames = fit_all()
    predictions = (
        pd.concat(frames, ignore_index=True).sort_values(ROW_ID, kind="mergesort")
        .reset_index(drop=True)
    )
    predictions.to_csv(root / "cross_fitted_predictions.csv", index=False)
    co_primary = co_primary_estimands(
        predictions,
        propensity_clip=settings["propensity_clip"],
        lead_estimand=settings["lead_estimand"],
    )
    heterogeneity = build_heterogeneity_diagnostics(
        predictions,
        propensity_clip=settings["propensity_clip"],
        seed=settings["seed"],
        n_bootstrap=2,
        outcome_type=settings["outcome_type"],
    )
    summary = {
        "schema_version": CEILING_SCHEMA_VERSION,
        "status": "ok",
        "fingerprint": fingerprint,
        "design_rule": CEILING_DESIGN_RULE,
        "estimators": {str(fold): name for fold, name in sorted(estimators.items())},
        "selected_features": [
            {"name": item["name"], "roles": item["roles"],
             "data_generating_roles": item["data_generating_roles"]}
            for item in selected
        ],
        "nuisance_candidates": [item["name"] for item in catalog],
        "co_primary_estimands": co_primary,
    }
    (root / "ceiling_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        **summary,
        "output_dir": str(root),
        "predictions": predictions,
        "heterogeneity": heterogeneity,
    }


def _json_ready(value: Any) -> Any:
    from ..inference.stage2_diagnostics import json_safe

    return json_safe(value)


__all__ = [
    "CEILING_DESIGN_RULE",
    "CEILING_SCHEMA_VERSION",
    "ceiling_definitions",
    "ceiling_settings",
    "default_estimator",
    "run_oracle_ceiling",
]
