from __future__ import annotations

from pathlib import Path

import pytest

from oci.config import ExperimentConfig


def test_generic_experiment_validation_rejects_stage1_only_default(tmp_path: Path):
    dataset = tmp_path / "dataset.parquet"
    dataset.touch()
    config = ExperimentConfig.from_dict(
        {"applied_inference": {"dataset_path": str(dataset)}}
    )

    assert config.applied_inference.architecture.model_type == "multi_model_forest"
    with pytest.raises(ValueError, match="ResearchAllEvidenceWorkflow"):
        config.validate()
