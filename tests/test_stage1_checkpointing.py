from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oci.inference.stage1_checkpointing import (
    Stage1CheckpointStore,
    row_id_identity,
)


def _store(tmp_path: Path, *, identity: str = "context-a") -> Stage1CheckpointStore:
    return Stage1CheckpointStore(
        tmp_path / "checkpoints" / "v1",
        context_identity={
            "context": identity,
            "fit": row_id_identity([4, 2, 9]),
            "heldout": row_id_identity([1, 8]),
        },
        base_seed=123,
    )


def test_completed_leaf_survives_interruption_and_resumes_independently(tmp_path):
    store = _store(tmp_path)
    completed_calls = 0

    def completed_compute():
        nonlocal completed_calls
        completed_calls += 1
        return np.asarray([10.0, 20.0])

    store.run("bow/nuisance/fold_001", completed_compute)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        store.run(
            "bow/nuisance/fold_002",
            lambda: (_ for _ in ()).throw(RuntimeError("simulated interruption")),
        )

    resumed = _store(tmp_path)
    reused = resumed.run("bow/nuisance/fold_001", completed_compute)
    second = resumed.run(
        "bow/nuisance/fold_002",
        lambda: np.asarray([30.0, 40.0]),
    )

    assert completed_calls == 1
    assert reused.reused is True
    assert second.reused is False
    progress = json.loads(
        (resumed.root / "checkpoint_progress.json").read_text(encoding="utf-8")
    )
    assert progress["units"]["bow/nuisance/fold_001"]["status"] == "reused"
    assert progress["units"]["bow/nuisance/fold_002"]["status"] == "completed"


def test_context_or_row_identity_change_invalidates_without_scanning_payloads(tmp_path):
    first = _store(tmp_path, identity="config-a")
    first.run(
        "unit",
        lambda: np.asarray([1.0]),
        rows={"fit": row_id_identity([1, 2, 3])},
    )
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return np.asarray([2.0])

    changed_config = _store(tmp_path, identity="config-b")
    changed_config.run(
        "unit",
        compute,
        rows={"fit": row_id_identity([1, 2, 3])},
    )
    changed_rows = _store(tmp_path, identity="config-a")
    changed_rows.run(
        "unit",
        compute,
        rows={"fit": row_id_identity([3, 2, 1])},
    )

    assert calls == 2
