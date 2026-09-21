"""Post-fit direct oracle-concept recovery audit; does not fit or select models."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DATE = "2026-09-21"
THRESHOLD = 1e-8
LINEAGE = HERE.parent / f"fold_1_interim_oracle_review_{DATE}" / f"fold_1_feature_recovery_{DATE}.json"
FREEZE = HERE / f"predictions_frozen_{DATE}.json"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


frozen = json.loads(FREEZE.read_text())
lineage = json.loads(LINEAGE.read_text())
assert len(lineage["concepts"]) == 10
sources = {str(LINEAGE): digest(LINEAGE), str(FREEZE): digest(FREEZE)}
cohorts = {}
rows = []
proxy_ids = {
    "gender_identity": ("outer_001_feature_140", "Separate construct; not a direct biological-sex match."),
    "brain_lesion_count": ("outer_001_feature_049", "Possible proxy for brain-metastasis presence; not a direct match."),
    "hematocrit": ("outer_001_feature_153", "Possible proxy for hemoglobin; not a direct match."),
    "estimated_glomerular_filtration_rate": ("outer_001_feature_125", "Possible proxy for creatinine clearance; not a direct match."),
}

for cohort in ("all_800", "eligible_720"):
    path = HERE / f"fits/{cohort}/elastic_net/coefficients.csv"
    assert digest(path) == frozen["files"][str(path)]
    sources[str(path)] = digest(path)
    coefficients = pd.read_csv(path)
    selected = coefficients[coefficients.coefficient.abs() > THRESHOLD].copy()
    concepts = []
    for concept in lineage["concepts"]:
        direct_ids = set(concept["direct_feature_ids"])
        matches = selected[selected.feature_id.isin(direct_ids)]
        role = concept["oracle_roles"]
        assert len(role) == 1
        main = matches[matches.block == "main"]
        interactions = matches[matches.block == "interaction"]
        record = {
            "concept": concept["concept"],
            "oracle_role": role[0],
            "direct_feature_ids": sorted(direct_ids),
            "main_selected": not main.empty,
            "interaction_selected": not interactions.empty,
            "any_selected": not matches.empty,
            "selected_coefficients": matches.to_dict(orient="records"),
            "measurement_lineage_note": concept["lineage_review"],
        }
        concepts.append(record)
        rows.append({
            "training_cohort": cohort,
            "concept": record["concept"],
            "oracle_role": record["oracle_role"],
            "direct_feature_ids": ";".join(record["direct_feature_ids"]),
            "main_selected": record["main_selected"],
            "interaction_selected": record["interaction_selected"],
            "any_selected": record["any_selected"],
            "selected_main_terms": ";".join(main.encoded_term),
            "selected_interaction_terms": ";".join(interactions.encoded_term),
        })
    confounders = [c for c in concepts if c["oracle_role"] == "confounder"]
    modifiers = [c for c in concepts if c["oracle_role"] == "effect_modifier"]
    assert len(confounders) == len(modifiers) == 5
    main = selected[selected.block == "main"]
    interactions = selected[selected.block == "interaction"]
    proxies = []
    for name, (feature_id, note) in proxy_ids.items():
        matches = selected[selected.feature_id == feature_id]
        proxies.append({
            "name": name, "feature_id": feature_id, "note": note,
            "main_selected": bool((matches.block == "main").any()),
            "interaction_selected": bool((matches.block == "interaction").any()),
            "selected_coefficients": matches.to_dict(orient="records"),
        })
    cohorts[cohort] = {
        "main_coefficients": len(main),
        "main_candidate_ids": int(main.feature_id.nunique()),
        "interaction_coefficients": len(interactions),
        "interaction_candidate_ids": int(interactions.feature_id.nunique()),
        "confounders_main_selected": sum(c["main_selected"] for c in confounders),
        "confounders_any_selected": sum(c["any_selected"] for c in confounders),
        "modifiers_interaction_selected": sum(c["interaction_selected"] for c in modifiers),
        "modifiers_any_selected": sum(c["any_selected"] for c in modifiers),
        "concepts": concepts,
        "possible_proxies_not_counted_as_direct_recovery": proxies,
    }

for path, expected in sources.items():
    assert digest(Path(path)) == expected
result = {
    "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "prediction_freeze_preceded_oracle_audit": True,
    "predictions_frozen_at": frozen["frozen_at"],
    "selection_rule": f"At least one encoded coefficient with absolute value greater than {THRESHOLD} in the indicated block.",
    "source_hashes": sources,
    "audit_script_sha256": digest(Path(__file__)),
    "notes": [
        "This audits the final penalized outcome models, not the earlier statistical-selection or nuisance models.",
        "Direct identities come from the saved lineage review and were read only after model fitting and prediction freezing.",
        "Direct concept retention does not establish faithful extraction, full category recovery, or sufficient adjustment.",
        "Possible proxies are not credited as direct matches; their correlations with oracle values were not evaluated here.",
        "The designated DGP modifier role refers to log-odds treatment interactions. Main effects can also change probability-scale treatment differences through the inverse link.",
        "Unmatched interactions are not automatically established causal false positives; predictive proxies, measurement error, and model misspecification complicate that interpretation.",
        "Categorical indicators and missingness columns are separate coefficients; all counts distinguish coefficients from candidate identities.",
    ],
    "cohorts": cohorts,
}
(HERE / f"variable_selection_audit_{DATE}.json").write_text(json.dumps(result, indent=2) + "\n")
pd.DataFrame(rows).to_csv(HERE / f"oracle_variable_selection_{DATE}.csv", index=False)
for cohort, values in cohorts.items():
    print(cohort, json.dumps({key: value for key, value in values.items() if isinstance(value, int)}))
