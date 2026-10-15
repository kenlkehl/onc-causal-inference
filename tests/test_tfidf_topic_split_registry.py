import pytest

from oci.inference.tfidf_topic_discovery import row_set_fingerprint
from oci.inference.tfidf_topic_split_registry import (
    SplitRegistryError,
    validate_handoff_rows_against_split_registry,
)


def _registry() -> dict:
    heldout_groups = ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11])
    all_rows = list(range(12))
    outer_folds = []
    for outer_fold, heldout in enumerate(heldout_groups, start=1):
        fit = [row_id for row_id in all_rows if row_id not in heldout]
        midpoint = len(fit) // 2
        outer_folds.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": fit,
                "heldout_row_ids": list(heldout),
                "inner_folds": [
                    {
                        "inner_fold": 1,
                        "fit_row_ids": fit[midpoint:],
                        "heldout_row_ids": fit[:midpoint],
                    },
                    {
                        "inner_fold": 2,
                        "fit_row_ids": fit[:midpoint],
                        "heldout_row_ids": fit[midpoint:],
                    },
                ],
            }
        )
    return {"outer_folds": outer_folds}


def _handoff_rows(registry: dict) -> list[dict]:
    rows = []
    for outer in registry["outer_folds"]:
        for split, scope in [
            *[(inner, "candidate_selection_inner_fit") for inner in outer["inner_folds"]],
            (outer, "full_outer_train"),
        ]:
            inner_fold = split.get("inner_fold")
            fit_ids = list(split["fit_row_ids"])
            heldout_ids = list(split["heldout_row_ids"])
            rows.append(
                {
                    "fold_key": (
                        outer["outer_fold"]
                        if inner_fold is None
                        else 1000 * outer["outer_fold"] + inner_fold
                    ),
                    "outer_fold": outer["outer_fold"],
                    "inner_fold": inner_fold,
                    "scope": scope,
                    "fit_row_ids": fit_ids,
                    "heldout_row_ids": heldout_ids,
                    "fit_row_fingerprint": row_set_fingerprint(fit_ids),
                    "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
                    "discovery": {
                        "fit_row_ids": fit_ids,
                        "heldout_row_ids": heldout_ids,
                        "fit_row_fingerprint": row_set_fingerprint(fit_ids),
                        "heldout_row_fingerprint": row_set_fingerprint(heldout_ids),
                    },
                }
            )
    return rows


def test_handoff_rows_must_match_saved_splits_including_row_order():
    registry = _registry()
    rows = _handoff_rows(registry)
    validate_handoff_rows_against_split_registry(rows, registry)

    # A row-order change preserves set fingerprints, so exact matching must
    # still reject it.
    full = next(
        row for row in rows if row["outer_fold"] == 1 and row["scope"] == "full_outer_train"
    )
    assert row_set_fingerprint(full["fit_row_ids"]) == row_set_fingerprint(
        list(reversed(full["fit_row_ids"]))
    )
    full["fit_row_ids"] = list(reversed(full["fit_row_ids"]))
    full["discovery"]["fit_row_ids"] = list(full["fit_row_ids"])
    with pytest.raises(SplitRegistryError, match="row ids/order"):
        validate_handoff_rows_against_split_registry(rows, registry)
