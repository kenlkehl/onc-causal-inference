from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oci.evaluation.stage1 import (
    evaluate_stage1_architectures,
)
from oci.inference.stage1_architecture_artifacts import (
    materialize_stage1_architecture_artifacts,
)


def test_posthoc_evaluation_uses_frozen_architecture_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    dataset_path = tmp_path / "dataset.parquet"
    metadata_path = tmp_path / "metadata.json"
    rows = 12
    age = np.arange(rows, dtype=float)
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index}" for index in range(rows)],
            "clinical_text": [f"age marker {index}" for index in range(rows)],
            "treatment_indicator": [0, 1] * (rows // 2),
            "outcome_indicator": [0, 0, 1, 1] * (rows // 4),
            "true_age": age,
            "true_marker": ["low"] * 6 + ["high"] * 6,
            "true_ite_prob": np.linspace(-0.2, 0.3, rows),
        }
    )
    dataset.to_parquet(dataset_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "name": "age",
                        "type": "continuous",
                        "description": "patient age in years",
                        "roles": ["confounder"],
                    },
                    {
                        "name": "marker",
                        "type": "categorical",
                        "categories": ["low", "high"],
                        "roles": ["effect_modifier"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "treatment_column": "treatment_indicator",
                "outcome_column": "outcome_indicator",
                "outcome_type": "binary",
            }
        ),
        encoding="utf-8",
    )
    evidence_path = run_dir / "components" / "text_models" / "evidence.jsonl"
    evidence_path.parent.mkdir(parents=True)
    raw = {
        "source": "text_models",
        "outer_fold": 1,
        "inner_fold": None,
        "scope": "full_outer_train",
        "evidence": {
            "importance": {
                "views": [
                    {
                        "view_name": "word_linear",
                        "treatment_positive": [{"feature": "age marker", "score": 2.0}],
                        "outcome_positive": [],
                        "pseudo_target_positive": [],
                    }
                ]
            }
        },
    }
    evidence_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    score_path = (
        run_dir
        / "components"
        / "text_models"
        / "outer_001_full"
        / "worker_artifacts"
        / "fold_000001"
        / "predictions.parquet"
    )
    score_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "split_role": "test_outer_train_fit",
            "source_name": "bow__linear__nuisance",
            "architecture": "bow_nuisance",
            "e_hat": np.asarray(dataset["treatment_indicator"], dtype=float) * 0.8 + 0.1,
            "m_hat": np.asarray(dataset["outcome_indicator"], dtype=float) * 0.8 + 0.1,
            "age_score": age,
        }
    ).to_parquet(score_path, index=False)
    materialize_stage1_architecture_artifacts(
        output_dir=run_dir,
        raw_handoff_rows=[raw],
        selected_architectures=("bow_nuisance",),
        source_artifacts={"text_models": evidence_path},
        selection_mode="explicit",
    )

    result = evaluate_stage1_architectures(
        run_dir=run_dir,
        metadata_path=metadata_path,
    )

    output = Path(result["output_dir"])
    assert output == run_dir / "evaluation" / "stage1"
    for name in (
        "metrics.jsonl",
        "architectures/bow_nuisance/metrics.jsonl",
        "comparison.csv",
        "lexical_recovery_null.csv",
        "timing.csv",
        "summary.json",
        "evaluation_manifest.json",
    ):
        assert (output / name).is_file(), name
    metrics = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        row["metric"] == "best_score_abs_spearman"
        and row["target"] == "true_age"
        and row["value"] == 1.0
        for row in metrics
    )
    assert any(
        row["metric"] == "best_treatment_orientation_free_auc" and row["value"] == 1.0
        for row in metrics
    )
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["oracle_columns_available_to_stage1"] is False
    assert manifest["stage1_artifacts_frozen_before_oracle_load"] is True
    assert manifest["stage1_evidence_sha256"]["bow_nuisance"]
