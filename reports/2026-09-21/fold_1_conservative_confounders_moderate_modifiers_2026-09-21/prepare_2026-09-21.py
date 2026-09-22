"""Assemble the user-requested role union without reading oracle values."""
from copy import deepcopy
from pathlib import Path
from common import HERE, BASE, PRIOR, DATE, n


def main():
    frozen_path = HERE / f"selection_frozen_{DATE}.json"
    if frozen_path.exists():
        from common import verify_inputs
        verify_inputs()
        return
    parent = n.read(BASE / f"input_manifest_{DATE}.json")
    n.verify(parent["sources"])
    n.verify(parent["input_files"])
    upstream = {}
    for folder, filename in [(BASE, f"selection_frozen_{DATE}.json"),
                             (PRIOR, f"selection_frozen_{DATE}.json"),
                             (PRIOR, f"predictions_frozen_{DATE}.json")]:
        path = folder / filename
        freeze = n.read(path)
        n.verify(freeze["files"])
        upstream.update(freeze["files"])
        upstream[str(path)] = n.sha(path)
    definitions = n.read(n.INPUTS / "inputs/definitions.json")
    original = n.read(BASE / "selected_definitions.json")["features"]
    confounders = {d["feature_id"] for d in original if "confounder" in d["roles"]}
    review = n.read(PRIOR / "global_response.json")
    ranking = ["outer_001_feature_" + d["id"] for d in review["modifier_ranking"]]
    size = n.read(PRIOR / "size_choice.json")
    assert size["best_mean_size"] == 16
    modifiers = set(ranking[:16])
    assert len(confounders) == 189 and len(modifiers) == 16
    assert confounders | modifiers <= {d["feature_id"] for d in definitions}
    selected, decisions = [], []
    for d in definitions:
        key = d["feature_id"]
        roles = (["confounder"] if key in confounders else []) + (["effect_modifier"] if key in modifiers else [])
        decisions.append({"feature_id": key, "name": d["name"], "roles": roles,
                          "modifier_rank": ranking.index(key) + 1 if key in ranking else None})
        if roles:
            item = deepcopy(d)
            item["roles"] = roles
            item["nuisance_model_roles"] = ["treatment", "outcome"] if key in confounders else []
            item["selection_source"] = "broad_confounders_and_frozen_global_top16_modifiers"
            selected.append(item)
    n.write(HERE / "selected_definitions.json", {"features": selected})
    n.write(HERE / "role_report.json", {"decisions": decisions})
    local_sources = [Path(__file__), HERE / "common.py", HERE / f"fit_{DATE}.py", HERE / f"PROTOCOL_{DATE}.md"]
    manifest = {
        "created_at": n.now(), "outer_fold": 1, "training_rows": 800, "heldout_rows": 200,
        "evaluation_plan": parent["evaluation_plan"], "input_files": parent["input_files"],
        "sources": {**parent["sources"], **{str(p): n.sha(p) for p in local_sources}},
        "upstream_files": upstream, "versions": parent["versions"],
        "prior_fold_1_oracle_results_seen": True, "exploratory_post_hoc_variant": True,
        "oracle_values_read_by_selection_or_fitting": False, "extraction_refreshed_again": False,
        "rule": "All original 189 confounders plus the existing global ranking's first 16 modifiers; minimum prior mean inner-fold R-loss.",
        "prior_best_inner_r_loss": size["options"]["16"]["mean_r_loss"],
        "prior_inner_size_choice_was": size["chosen_size"],
    }
    manifest_path = HERE / f"input_manifest_{DATE}.json"
    n.write(manifest_path, manifest)
    n.write(frozen_path, {
        "frozen_at": n.now(), "input_manifest_sha256": n.sha(manifest_path),
        "files": {str(p): n.sha(p) for p in [HERE / "selected_definitions.json", HERE / "role_report.json"]},
        "selected_count": len(selected), "candidate_count": len(definitions),
        "roles": {"confounder": len(confounders), "effect_modifier": len(modifiers), "both": len(confounders & modifiers)},
        "modifier_order": ranking[:16], "prior_oracle_exposure_acknowledged": True,
    })
    n.write(HERE / "status.json", {"phase": "selection_frozen", "updated_at": n.now()})
    print({"features": len(selected), "confounders": len(confounders), "modifiers": len(modifiers), "both": len(confounders & modifiers)})


if __name__ == "__main__":
    main()
