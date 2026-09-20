"""Isolated, checkpointed four-way Stage 2 selector experiment.

Run with the existing /home/klkehl/thisenv/bin/python environment. Production
artifacts are read only. Oracle values are opened only by the evaluate phase,
after all experimental predictions have been frozen and hashed.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/oci_selection_comparison_matplotlib")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

METHODS = ("current_joint", "llm_adjudication", "permissive_union", "all_candidates")
SCHEMA = "frozen_nuisance_selection_comparison_v1"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Incompatible resume input: {path}")
    else:
        write_json(path, value)


def progress(out, phase, **details):
    value = {"updated_at": now(), "phase": phase, **details}
    write_json(out / "status.json", value)
    print(json.dumps(value), flush=True)


def align(frame, row_ids, names=None):
    """Never silently fill absent columns or missing/duplicated row IDs."""
    ids = list(map(int, row_ids))
    if len(ids) != len(set(ids)) or frame["_oci_row_id"].duplicated().any():
        raise ValueError("Duplicated row IDs")
    if set(map(int, frame["_oci_row_id"])) != set(ids):
        raise ValueError("Measurement row coverage differs from the frozen split")
    if names is not None and set(names) - set(frame):
        raise ValueError("Missing measurement columns")
    frame = frame.set_index("_oci_row_id", drop=False).loc[ids].reset_index(drop=True)
    return frame if names is None else frame[["_oci_row_id", *names]]


def admissions(definitions, statistical, llm_decisions):
    """Policy uses only training evidence, never clinical names or oracle roles."""
    ids = [d["feature_id"] for d in definitions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicated feature IDs")
    llm = {d["feature_id"]: d for d in llm_decisions}
    if len(llm) != len(llm_decisions) or set(llm) != set(ids):
        raise ValueError("LLM decisions must cover every candidate exactly once")
    effect_votes = statistical["multivariable_modifier_elastic_net_screen"]["votes"]
    if set(effect_votes) != set(ids):
        raise ValueError("Joint votes must cover every candidate")
    gains = {fid: [] for fid in ids}
    for fold in statistical["effect_modifier_screen"]["folds"]:
        for test in fold["tests"]:
            gain = test.get("heldout_r_loss_improvement")
            if test.get("status") == "ok" and gain is not None and np.isfinite(gain):
                gains[test["feature_id"]].append(float(gain))
    selected = {method: [] for method in METHODS}
    records = []
    for feature in definitions:
        fid = feature["feature_id"]
        joint = effect_votes[fid] > 0
        llm_effect = "effect_modifier" in llm[fid]["roles"]
        individual = (len(gains[fid]) == 5 and sum(g > 0 for g in gains[fid]) >= 3
                      and float(np.mean(gains[fid])) > 0)
        chosen = dict(zip(METHODS, (joint, llm_effect, joint or llm_effect or individual, True)))
        locked = feature.get("configured_explicit_feature") is True
        if locked:
            chosen = {method: "effect_modifier" in feature["roles"] for method in METHODS}
        for method, include in chosen.items():
            if include:
                selected[method].append(fid)
        records.append({"feature_id": fid, "name": feature["name"],
                        "joint_effect_votes": int(effect_votes[fid]),
                        "llm_effect_assignment": llm_effect,
                        "individual_positive_folds": sum(g > 0 for g in gains[fid]),
                        "individual_mean_gain": float(np.mean(gains[fid])) if gains[fid] else None,
                        "individual_rescue": bool(individual), "investigator_locked": locked,
                        "included": chosen})
    return {"methods": selected, "decisions": records}


def loss_metrics(t, y, e, m, cate, constant):
    arrays = [np.asarray(a, dtype=float).reshape(-1) for a in (t, y, e, m, cate)]
    if len({len(a) for a in arrays}) != 1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Nonfinite or misaligned loss inputs")
    t, y, e, m, cate = arrays
    loss = float(np.mean(((y - m) - (t - e) * cate) ** 2))
    base = float(np.mean(((y - m) - (t - e) * constant) ** 2))
    return {"rows": len(t), "r_loss": loss, "constant_r_loss": base,
            "r_score": 1 - loss / base if base else None,
            "mean_cate": float(np.mean(cate)), "sd_cate": float(np.std(cate))}


def prepare(source, out, args):
    out.mkdir(parents=True, exist_ok=True)
    run = read_json(source / "run_config.json")
    original = read_json(source / "stage2/config.json")
    files = [source / "run_config.json", source / "stage2/config.json",
             source / "stage2/model_identity.json", Path(run["dataset"]),
             HERE / "PROTOCOL_2026-09-19.md", Path(__file__), HERE / "launch.py"]
    files += [ROOT / rel for rel in (
        "oci/inference/plain_handoff_stage2.py", "oci/inference/plain_handoff_stage2_analysis.py",
        "oci/inference/stage2_elastic_net_selection.py", "oci/inference/stage2_role_adjudication.py",
        "oci/inference/stage2_taskwise_policy.py", "oci/models/causal_forest_head.py",
        "oci/models/elastic_net_nuisance.py")]
    # Freeze the producer's Stage 1 handoff manifests/provenance as well. The
    # experiment consumes frozen Stage 2 definitions, not raw Stage 1 models.
    files += sorted((source / "handoff").glob("*.json"))
    splits = []
    for fold in range(1, 6):
        p = source / "stage2" / f"outer_{fold:03d}"
        files.extend([p / "feature_definitions.json", p / "selection/input.json"])
        inp = read_json(p / "selection/input.json")
        inner = inp["inner_splits"]
        fit = sorted({int(i) for s in inner for i in s["heldout_row_ids"]})
        heldout = pd.read_csv(p / "estimation/predictions.csv", usecols=["_oci_row_id"])["_oci_row_id"].tolist()
        files.append(p / "estimation/predictions.csv")
        if set(fit) & set(heldout):
            raise ValueError("Outer-fold overlap")
        if len(set(fit) | set(heldout)) != len(fit) + len(heldout):
            raise ValueError("Bad row partition")
        for s in inner:
            if set(s["fit_row_ids"]) & set(s["heldout_row_ids"]):
                raise ValueError("Inner-fold overlap")
            if set(s["fit_row_ids"]) | set(s["heldout_row_ids"]) != set(fit):
                raise ValueError("Inner partition differs from outer training")
        splits.append({"outer_fold": fold, "fit_row_ids": fit, "heldout_row_ids": heldout,
                       "inner_splits": inner})
        immutable_json(out / "inputs" / f"outer_{fold:03d}_definitions.json",
                       read_json(p / "feature_definitions.json"))
    all_heldout = [i for split in splits for i in split["heldout_row_ids"]]
    if len(all_heldout) != len(set(all_heldout)):
        raise ValueError("Patients occur in more than one outer-heldout fold")
    manifest = {str(p.resolve()): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}
    immutable_json(out / "inputs/source_manifest.json", manifest)
    immutable_json(out / "inputs/splits.json", splits)
    config = copy.deepcopy(original)
    config.update(endpoint=args.primary_endpoint, model=args.primary_model, api_key="EMPTY", vllm=None,
                  workers=args.workers)
    config["extraction_llm"] = dict(endpoint=args.extractor_endpoint, model=args.extractor_model,
                                     api_key="EMPTY", workers=args.workers, vllm=None)
    config["role_adjudication"]["enabled"] = False  # Numerical reference needs no advisory call.
    config["statistical_selection"]["selection_mode"] = "independent_tasks"
    config["selection_consolidation"]["enabled"] = False
    immutable_json(out / "inputs/refresh_config.json", config)
    immutable_json(out / "inputs/run_settings.json", {k: run[k] for k in
        ("dataset", "clinical_question", "unit_id_column", "text_column", "treatment_column",
         "outcome_column", "outcome_type", "seed", "inner_folds")})
    immutable_json(out / "inputs/experiment.json", {
        "schema": SCHEMA, "source": str(source.resolve()), "methods": list(METHODS),
        "measurement_policy": "refresh_all_with_one_new_extractor",
        "forest_replicates": 3, "permissive_minimum_positive_folds": 3,
        "permissive_mean_gain_must_be_positive": True,
        "nuisances": "all_candidate_group_elastic_net_shared_residuals",
        "production_artifacts_read_only": True, "oracle_used_in_fitting": False})
    immutable_json(out / "inputs/software.json", {"python": sys.version, "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("pandas", "numpy", "scikit-learn", "econml", "transformers")}})
    progress(out, "prepared", outer_folds=len(splits), rows=len(all_heldout))


def make_runtime(config, out):
    from oci.inference.plain_handoff_stage2 import (PlainHandoffStage2, plain_stage2_config_from_mapping,
                                                   _request_json, _LazyStage2ExtractionTokenizer)
    from oci.inference.plain_handoff_stage2_analysis import prompt_token_count
    cfg = plain_stage2_config_from_mapping(config, default_workers=config["workers"])
    runtime = PlainHandoffStage2(config=cfg, clinical_question="")
    runtime._check_and_record_model_identity(out)
    identity = runtime.model_identity["extraction"]
    identity_text = json.dumps(identity).lower()
    if "26b" not in identity_text or "a4b" not in identity_text:
        raise ValueError("The extraction service is not the requested Gemma 4 26B A4B model")
    tokenizer_model = identity.get("actual_model_identity", {}).get("root") or cfg.extraction_llm.model
    if tokenizer_model != cfg.extraction_llm.model:
        runtime.extraction_tokenizer = _LazyStage2ExtractionTokenizer(model=tokenizer_model, cache_dir="")
    immutable_json(out / "tokenizer_identity.json", {"model": tokenizer_model,
        "source": "verified_extraction_service_root"})

    def request(messages, validate, *, request_kind="interpretation", **kwargs):
        extraction = request_kind == "extraction"
        rcfg = runtime.extraction_request_config if extraction else runtime.config
        rcfg = dataclasses.replace(rcfg, max_prompt_chars=(runtime.config.extraction_max_prompt_chars
                                                         if extraction else runtime.config.max_prompt_chars))
        return _request_json(messages=messages, config=rcfg,
            completion=runtime.extraction_completion if extraction else runtime.completion,
            validate=validate, request_kind=request_kind,
            prompt_token_counter=(lambda m: prompt_token_count(runtime.extraction_tokenizer, m)) if extraction else None,
            context_window_tokens=runtime.config.extraction_context_window_tokens if extraction else None,
            context_margin_tokens=runtime.config.extraction_context_margin_tokens if extraction else 0, **kwargs)
    return runtime, request


def bounded_role_adjudication(*, definitions, statistical_report, request_json,
                              output_dir, policy, max_prompt_chars):
    """Fit whole-candidate role prompts within the existing transport limit."""
    from oci.inference.stage2_role_adjudication import (
        ROLE_ADJUDICATION_SYSTEM_PROMPT, _canonical_json, _fingerprint,
        _role_request_payload, adjudicate_stage2_roles, build_stage2_role_evidence,
    )
    from oci.inference.stage2_taskwise_policy import independent_tasks_enabled
    policy.validate()
    if not policy.enabled:
        raise ValueError("Binding role adjudication must be enabled")
    if independent_tasks_enabled(statistical_report):
        raise ValueError("Binding role prompt preflight requires llm_roles evidence")
    if isinstance(max_prompt_chars, bool) or not isinstance(max_prompt_chars, int) or max_prompt_chars < 1:
        raise ValueError("Role prompt character limit must be a positive integer")
    evidence = build_stage2_role_evidence(definitions=definitions,
        statistical_report=statistical_report, policy=policy)
    candidates = evidence["candidates"]
    configured_limit = int(policy.max_candidates_per_request)
    configured_sizes = None
    for candidate_limit in range(configured_limit, 0, -1):
        batches = [candidates[start:start + candidate_limit]
                   for start in range(0, len(candidates), candidate_limit)]
        prompt_sizes = []
        for index, batch in enumerate(batches, start=1):
            payload = _role_request_payload(evidence={**evidence, "candidates": batch},
                batch_index=index, batch_count=len(batches))
            prompt_sizes.append(len(ROLE_ADJUDICATION_SYSTEM_PROMPT) + len(_canonical_json(payload)))
        if configured_sizes is None:
            configured_sizes = prompt_sizes
        if all(size <= max_prompt_chars for size in prompt_sizes):
            break
    else:
        raise ValueError("One complete role-evidence candidate exceeds the prompt character limit; "
                         f"largest singleton={max(prompt_sizes)}, limit={max_prompt_chars}. "
                         "No evidence was truncated and no adjudication request was sent.")
    bounded_policy = dataclasses.replace(policy, max_candidates_per_request=candidate_limit)
    # Only request grouping changes. Keep all allowlisted evidence, candidate
    # order, decision authority, and the adjudicator's checkpoint validation.
    if build_stage2_role_evidence(definitions=definitions, statistical_report=statistical_report,
                                 policy=bounded_policy) != evidence:
        raise ValueError("Role request batching changed the evidence")
    immutable_json(Path(output_dir) / "request_budget.json", {
        "schema": "comparison_role_prompt_budget_v1", "max_prompt_chars": max_prompt_chars,
        "configured_max_candidates_per_request": configured_limit,
        "effective_max_candidates_per_request": candidate_limit,
        "configured_prompt_chars": configured_sizes, "rendered_prompt_chars": prompt_sizes,
        "evidence_fingerprint": _fingerprint(evidence),
        "candidate_batches": [[c["feature_id"] for c in batch] for batch in batches],
        "evidence_truncated": False,
    })
    return adjudicate_stage2_roles(definitions=definitions, statistical_report=statistical_report,
        request_json=request_json, output_dir=output_dir, policy=bounded_policy)


def safe_dataset(settings):
    columns = [settings[k] for k in ("unit_id_column", "text_column", "treatment_column", "outcome_column")]
    if len(columns) != len(set(columns)):
        raise ValueError("Dataset column roles overlap")
    return pd.read_parquet(settings["dataset"], columns=columns)


def frozen_training_frame(fold_dir, inp, definitions):
    from oci.inference.plain_handoff_stage2_analysis import _apply_harmonization_plans, _frame_fingerprint
    exact = fold_dir / "comparison_measurements/fit.pkl"
    exact_audit = exact.with_suffix(".json")
    if exact.exists() and exact_audit.exists():
        audit = read_json(exact_audit)
        if sha256(exact) != audit["sha256"]:
            raise ValueError("Exact training-frame checkpoint changed")
        frame = pd.read_pickle(exact)
        if _frame_fingerprint(frame) != inp["extracted_fit_fingerprint"]:
            raise ValueError("Exact training frame differs from selection input")
        return frame
    candidates = [fold_dir / "extraction/all_candidates_fit_harmonization/extracted_harmonized.csv",
                  fold_dir / "extraction/all_candidates_fit/extracted.csv"]
    # Earlier rounds also retain the exact matrix when the final round does not change definitions.
    candidates += sorted((fold_dir / "ontology_supervision").glob("round_*/harmonization/extracted_harmonized.csv"), reverse=True)
    for p in candidates:
        if not p.exists():
            continue
        frame = pd.read_csv(p)
        variants = [frame, _apply_harmonization_plans(frame, definitions, scope="outer_training")[0]]
        for variant in variants:
            if _frame_fingerprint(variant) == inp["extracted_fit_fingerprint"]:
                return variant
    raise ValueError("Could not reproduce the exact matrix used for statistical selection")


def complete_measurements(dataset, definitions, split, fold_dir, runtime, request, settings):
    from oci.inference.plain_handoff_stage2_analysis import (
        extract_rows, _feature_extraction_fingerprint, _apply_harmonization_plans, _configured_serial_extraction)
    output = fold_dir / "comparison_measurements"
    final = read_json(fold_dir / "final_definitions.json")["features"]
    by_id = {d["feature_id"]: d for d in final}
    reusable = [d for d in definitions if d["feature_id"] in by_id and
                _feature_extraction_fingerprint(d) == _feature_extraction_fingerprint(by_id[d["feature_id"]])]
    names = [d["name"] for d in reusable]
    existing = align(pd.read_csv(fold_dir / "extraction/heldout/extracted.csv"), split["heldout_row_ids"])
    existing = existing[["_oci_row_id", *names]]
    missing = [d for d in definitions if d["name"] not in names]
    cfg = runtime.config
    more = extract_rows(dataset=dataset, row_ids=split["heldout_row_ids"], text_column=settings["text_column"],
        definitions=missing, output_dir=output / "heldout_additional", request_json=request,
        workers=cfg.extraction_llm.workers, max_prompt_chars=cfg.extraction_max_prompt_chars,
        feature_batch_size=cfg.extraction_feature_batch_size,
        request_identity={"model": cfg.extraction_llm.model, "experiment_schema": SCHEMA},
        tokenizer=runtime.extraction_tokenizer, **_configured_serial_extraction(cfg))
    raw = existing.merge(more, on="_oci_row_id", validate="one_to_one")
    raw = align(raw, split["heldout_row_ids"], [d["name"] for d in definitions])
    harmonized, audit = _apply_harmonization_plans(raw, definitions, scope="outer_heldout")
    harmonized.to_csv(output / "heldout.csv", index=False)
    write_json(output / "heldout_harmonization.json", audit)
    return harmonized


def fit_common_nuisances(dataset, fit, heldout, definitions, stat, split, cfg, settings, output):
    from oci.inference.stage2_elastic_net_selection import _encode_design, _logistic_elastic_net
    output.mkdir(parents=True, exist_ok=True)
    guard = fingerprint({"definitions": definitions, "stat_policy": stat["policy"],
                         "training_predictions": stat["cross_fitted_nuisance_models"]["predictions"],
                         "fit": fit.to_json(), "heldout": heldout.to_json(), "split": split,
                         "observed_fit": dataset.iloc[split["fit_row_ids"]][[settings["treatment_column"], settings["outcome_column"]]].to_json(),
                         "observed_test": dataset.iloc[split["heldout_row_ids"]][[settings["treatment_column"], settings["outcome_column"]]].to_json()})
    complete = output / "complete.json"
    if complete.exists():
        if read_json(complete)["fingerprint"] != guard:
            raise ValueError("Nuisance checkpoint mismatch")
        return pd.read_csv(output / "fit.csv"), pd.read_csv(output / "heldout.csv")
    train_predictions = align(pd.DataFrame(stat["cross_fitted_nuisance_models"]["predictions"]), split["fit_row_ids"])
    design = _encode_design(fit, heldout, definitions, categorical_min_count=cfg.statistical_selection.categorical_min_count)
    policy = dataclasses.replace(cfg.statistical_selection, min_propensity=cfg.min_propensity, max_propensity=cfg.max_propensity)
    if settings["outcome_type"] != "binary":
        raise ValueError("This preregistered cohort comparison expects a binary outcome")
    seed = settings["seed"] + 100000 * split["outer_fold"]
    efit = _logistic_elastic_net(design.train, dataset.iloc[split["fit_row_ids"]][settings["treatment_column"]].to_numpy(),
        design.valid, design.column_feature_ids, config=policy, seed=seed + 70000,
        one_standard_error_rule=policy.nuisance_prediction_one_standard_error_rule)
    mfit = _logistic_elastic_net(design.train, dataset.iloc[split["fit_row_ids"]][settings["outcome_column"]].to_numpy(),
        design.valid, design.column_feature_ids, config=policy, seed=seed + 70001,
        one_standard_error_rule=policy.nuisance_prediction_one_standard_error_rule)
    from oci.inference.nuisance_diagnostics import propensity_eligibility
    test_predictions = pd.DataFrame({"_oci_row_id": split["heldout_row_ids"],
        "treatment": dataset.iloc[split["heldout_row_ids"]][settings["treatment_column"]].to_numpy(),
        "outcome": dataset.iloc[split["heldout_row_ids"]][settings["outcome_column"]].to_numpy(),
        "propensity": efit.valid_prediction, "outcome_prediction": mfit.valid_prediction,
        "effect_eligible": propensity_eligibility(efit.valid_prediction, cfg.min_propensity, cfg.max_propensity)})
    train_predictions.to_csv(output / "fit.csv", index=False)
    test_predictions.to_csv(output / "heldout.csv", index=False)
    audit = {role: {k: getattr(model, k) for k in ("regularization", "cv_folds", "status", "iterations", "converged")}
             for role, model in (("treatment", efit), ("outcome", mfit))}
    write_json(output / "fit_audit.json", {"heldout_prediction_models": audit,
        "training_cross_fitted_models": stat["cross_fitted_nuisance_models"]["folds"],
        "policy": policy.public_dict(), "encoded_columns": len(design.column_names)})
    write_json(complete, {"fingerprint": guard, "completed_at": now(),
                         "fit_sha256": sha256(output / "fit.csv"), "heldout_sha256": sha256(output / "heldout.csv")})
    return train_predictions, test_predictions


def fit_forests(fit, heldout, definitions, routing, train_nuisance, test_nuisance, split, settings, output):
    from econml.grf import CausalForest
    from oci.inference.plain_handoff_stage2_analysis import _FeatureEncoder
    output.mkdir(parents=True, exist_ok=True)
    tr = train_nuisance["effect_eligible"].to_numpy(dtype=bool)
    te = test_nuisance["effect_eligible"].to_numpy(dtype=bool)
    if tr.sum() < 40 or te.sum() < 2:
        raise ValueError("Insufficient common overlap rows")
    t_res = (train_nuisance.treatment - train_nuisance.propensity).to_numpy()[tr]
    y_res = (train_nuisance.outcome - train_nuisance.outcome_prediction).to_numpy()[tr]
    constant = float(np.dot(t_res, y_res) / np.dot(t_res, t_res))
    nuisance_hash = fingerprint({"train": train_nuisance.to_json(), "test": test_nuisance.to_json()})
    all_summaries = []
    for method in METHODS:
        wanted = set(routing["methods"][method])
        subset = [d for d in definitions if d["feature_id"] in wanted]
        encoder = _FeatureEncoder(subset).fit(fit.loc[tr].reset_index(drop=True))
        x_train = encoder.transform(fit.loc[tr].reset_index(drop=True))
        x_test = encoder.transform(heldout.loc[te].reset_index(drop=True))
        if x_train.shape[1] == 0:
            x_train, x_test = np.ones((tr.sum(), 1)), np.ones((te.sum(), 1))
        for replicate in range(3):
            seed = settings["seed"] + 100000 * split["outer_fold"] + 20000 + 1000000 * replicate
            dest = output / method / f"seed_{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            params = dict(n_estimators=200, max_depth=None, min_samples_leaf=10, max_features="sqrt",
                          honest=True, inference=True, subforest_size=4, n_jobs=1, random_state=seed)
            guard = fingerprint({"features": subset, "params": params, "nuisance_hash": nuisance_hash,
                                 "train_design": hashlib.sha256(x_train.tobytes()).hexdigest(),
                                 "test_design": hashlib.sha256(x_test.tobytes()).hexdigest()})
            if (dest / "complete.json").exists():
                done = read_json(dest / "complete.json")
                if done["fingerprint"] != guard or done["predictions_sha256"] != sha256(dest / "predictions.csv"):
                    raise ValueError("Forest resume fingerprint mismatch")
                all_summaries.append(read_json(dest / "metrics.json"))
                continue
            model = CausalForest(**params).fit(x_train, t_res, y_res)
            cate, lower, upper = model.predict(x_test, interval=True)
            pred = test_nuisance.loc[te].reset_index(drop=True).copy()
            pred["outer_fold"], pred["method"], pred["replicate"] = split["outer_fold"], method, replicate
            pred["estimated_cate"], pred["lower_95"], pred["upper_95"] = cate.ravel(), lower.ravel(), upper.ravel()
            if not np.isfinite(pred[["estimated_cate", "lower_95", "upper_95"]].to_numpy()).all():
                raise ValueError("Nonfinite forest predictions")
            pred.to_csv(dest / "predictions.csv", index=False)
            metrics = loss_metrics(pred.treatment, pred.outcome, pred.propensity, pred.outcome_prediction,
                                   pred.estimated_cate, constant)
            metrics.update(method=method, outer_fold=split["outer_fold"], replicate=replicate,
                           features=len(subset), encoded_columns=x_train.shape[1], fit_rows=int(tr.sum()),
                           constant_effect=constant, nuisance_hash=nuisance_hash, forest_parameters=params)
            write_json(dest / "metrics.json", metrics)
            write_json(dest / "complete.json", {"fingerprint": guard, "predictions_sha256": sha256(dest / "predictions.csv"),
                                                "completed_at": now()})
            all_summaries.append(metrics)
    return all_summaries


def validate_run_inputs(out, revision_path=None):
    """Validate original frozen inputs or an explicitly recorded operational revision."""
    original_path = out / "inputs/source_manifest.json"
    manifest = read_json(original_path)
    config_path = out / "inputs/refresh_config.json"
    if revision_path is not None:
        revision = read_json(revision_path)
        if revision["schema"] != "selection_comparison_operational_revision_v1":
            raise ValueError("Unknown experiment revision schema")
        if revision["original_manifest_sha256"] != sha256(original_path):
            raise ValueError("Original experiment manifest changed")
        for path, record in revision["source_overrides"].items():
            if path not in manifest or record["before_sha256"] != manifest[path]["sha256"]:
                raise ValueError(f"Revision does not extend the original source manifest: {path}")
            if sha256(record["before_archive"]) != record["before_sha256"]:
                raise ValueError(f"Original source archive changed: {path}")
            manifest[path] = {"sha256": record["after_sha256"]}
        for path, record in revision["additional_sources"].items():
            if path in manifest:
                raise ValueError(f"Revision source added twice: {path}")
            manifest[path] = record
        for path, record in revision["frozen_inputs"].items():
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Frozen experimental input changed: {path}")
        if not {str(p.resolve()) for p in (out / "inputs").glob("*.json")} <= set(revision["frozen_inputs"]):
            raise ValueError("Revision does not protect all frozen experiment inputs")
        config_path = Path(revision["config"]["path"])
        if sha256(config_path) != revision["config"]["sha256"]:
            raise ValueError("Revised runtime configuration changed")
        original_config = read_json(out / "inputs/refresh_config.json")
        revised_config = read_json(config_path)
        changes = {k for k in set(original_config) | set(revised_config)
                   if original_config.get(k) != revised_config.get(k)}
        if changes != set(revision["config_changes"]) or not changes <= {
            "extraction_max_tokens", "extraction_reasoning_max_tokens",
            "extraction_stream", "extraction_deferred_retry_passes"
        }:
            raise ValueError("Operational revision changed scientific configuration")
        for key, values in revision["config_changes"].items():
            if values != {"before": original_config.get(key), "after": revised_config.get(key)}:
                raise ValueError(f"Inaccurate configuration change record: {key}")
    for path, record in manifest.items():
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Source changed after preparation/revision: {path}")
    return read_json(config_path)


def run(source, out, args):
    import oci.inference.plain_handoff_stage2_analysis as analysis
    config = validate_run_inputs(out, getattr(args, "revision", None))
    if getattr(args, "revision", None) is not None:
        write_json(out / "active_revision.json", {"path": str(args.revision.resolve()),
                   "sha256": sha256(args.revision), "validated_at": now()})
    settings = read_json(out / "inputs/run_settings.json")
    splits = read_json(out / "inputs/splits.json")
    runtime, request = make_runtime(config, out)
    dataset = safe_dataset(settings)
    summaries = []
    for split in splits:
        fold = split["outer_fold"]
        directory = out / "refresh" / f"outer_{fold:03d}"
        progress(out, "refreshing_measurements_and_selection", outer_fold=fold)
        definitions = read_json(out / "inputs" / f"outer_{fold:03d}_definitions.json")["features"]
        original_selector = analysis._run_stage2_statistical_selection

        def capture_selector(arguments):
            # Preserve exact values/dtypes at the selector boundary. CSV reloads
            # can change a mixed categorical/numeric column's dtype and hash.
            exact = directory / "comparison_measurements/fit.pkl"
            exact.parent.mkdir(parents=True, exist_ok=True)
            arguments["extracted_fit"].to_pickle(exact)
            write_json(exact.with_suffix(".json"), {"sha256": sha256(exact),
                "frame_fingerprint": analysis._frame_fingerprint(arguments["extracted_fit"])})
            return original_selector(arguments)

        analysis._run_stage2_statistical_selection = capture_selector
        try:
            analysis.run_fold_analysis(dataset=dataset, definitions=definitions, split=split,
                clinical_question=settings["clinical_question"], unit_id_column=settings["unit_id_column"],
                text_column=settings["text_column"], treatment_column=settings["treatment_column"],
                outcome_column=settings["outcome_column"], outcome_type=settings["outcome_type"],
                inner_folds=settings["inner_folds"], seed=settings["seed"] + 100000 * fold,
                output_dir=directory, request_json=request, config=runtime.config,
                extraction_tokenizer=runtime.extraction_tokenizer)
        finally:
            analysis._run_stage2_statistical_selection = original_selector
        inp = read_json(directory / "selection/input.json")
        definitions = inp["definitions"]
        names = [d["name"] for d in definitions]
        fit = align(frozen_training_frame(directory, inp, definitions), split["fit_row_ids"], names)
        stat = read_json(directory / "selection/statistical_evidence.json")
        progress(out, "binding_llm_adjudication", outer_fold=fold)
        legacy_evidence = copy.deepcopy(stat)
        legacy_evidence["policy"]["selection_mode"] = "llm_roles"
        # This changes decision authority only; evidence and nuisance fits are frozen.
        _, role_report, _ = bounded_role_adjudication(definitions=definitions, statistical_report=legacy_evidence,
            request_json=request, output_dir=directory / "comparison_llm",
            policy=dataclasses.replace(runtime.config.role_adjudication, enabled=True),
            max_prompt_chars=int(runtime.config.max_prompt_chars))
        routing = admissions(definitions, stat, role_report["decisions"])
        write_json(directory / "comparison_admissions.json", routing)
        progress(out, "extracting_additional_heldout_measurements", outer_fold=fold,
                 feature_counts={m: len(v) for m, v in routing["methods"].items()})
        heldout = complete_measurements(dataset, definitions, split, directory, runtime, request, settings)
        progress(out, "fitting_shared_nuisances", outer_fold=fold)
        train_nuisance, test_nuisance = fit_common_nuisances(dataset, fit, heldout, definitions, stat, split,
            runtime.config, settings, directory / "comparison_nuisances")
        progress(out, "fitting_four_forests_three_seeds", outer_fold=fold)
        summaries.extend(fit_forests(fit, heldout, definitions, routing, train_nuisance, test_nuisance,
                                    split, settings, directory / "comparison_forests"))
        write_json(out / "heldout_r_loss_results.json", summaries)
    predictions = sorted((out / "refresh").glob("outer_*/comparison_forests/*/seed_*/predictions.csv"))
    if len(predictions) != 60:
        raise ValueError("Expected 5 folds × 4 methods × 3 seeds")
    write_json(out / "predictions_frozen.json", {"completed_at": now(), "schema": SCHEMA,
        "files": {str(p.resolve()): sha256(p) for p in predictions}})
    progress(out, "all_predictions_frozen", forest_fits=60)


def evaluate(source, out):
    frozen = read_json(out / "predictions_frozen.json")
    for p, expected in frozen["files"].items():
        if sha256(p) != expected:
            raise ValueError("Prediction changed after freezing")
    # First oracle read in the experiment. Only evaluation below sees these columns.
    oracle_path = source / "stage2/posthoc_predictions_with_oracle_ite.csv"
    oracle = pd.read_csv(oracle_path)
    oracle_cols = [c for c in oracle if c.lower() in {"oracle_ite", "true_ite", "oracle_cate", "true_ite_prob"}]
    if len(oracle_cols) != 1:
        raise ValueError(f"Expected one explicit oracle effect column; found {oracle_cols}")
    truth = oracle[["_oci_row_id", oracle_cols[0]]].rename(columns={oracle_cols[0]: "oracle_effect"})
    if truth["_oci_row_id"].duplicated().any():
        raise ValueError("Oracle row IDs are duplicated")
    frames = [pd.read_csv(p) for p in frozen["files"]]
    predictions = pd.concat(frames, ignore_index=True).merge(truth, on="_oci_row_id", validate="many_to_one")
    if len(predictions) != sum(len(f) for f in frames) or predictions.oracle_effect.isna().any():
        raise ValueError("Oracle alignment failure")
    results = []
    for (method, replicate), frame in predictions.groupby(["method", "replicate"]):
        error = frame.estimated_cate - frame.oracle_effect
        results.append({"method": method, "replicate": int(replicate), "rows": len(frame),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)), "correlation": float(frame.estimated_cate.corr(frame.oracle_effect)),
            "predicted_sd": float(frame.estimated_cate.std(ddof=0)), "oracle_sd": float(frame.oracle_effect.std(ddof=0)),
            "coverage_95": float(((frame.lower_95 <= frame.oracle_effect) & (frame.oracle_effect <= frame.upper_95)).mean())})
    write_json(out / "oracle_evaluation.json", {"oracle_sha256": sha256(oracle_path), "results": results})
    predictions.to_csv(out / "predictions_with_oracle.csv", index=False)
    progress(out, "evaluation_complete", results=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "run", "evaluate"])
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/research_all_evidence/five_conf_five_mod_nsclc_full")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--primary-endpoint", default="http://sn4622130540:8000/v1")
    parser.add_argument("--primary-model", default="gemma4-31b")
    parser.add_argument("--extractor-endpoint", default="http://sn4622130540:8001/v1")
    parser.add_argument("--extractor-model", required=False)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--revision", type=Path,
                        help="Explicit operational revision manifest; original inputs remain frozen")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.source.resolve()):
        raise ValueError("Experiment output must be separate from the production source")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.output / "execution.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.phase == "prepare":
        if args.revision:
            parser.error("--revision applies to run, never preparation")
        if not args.extractor_model:
            parser.error("--extractor-model must be the verified served model ID")
        prepare(args.source, args.output, args)
    elif args.phase == "run":
        run(args.source, args.output, args)
    else:
        evaluate(args.source, args.output)


if __name__ == "__main__":
    main()
