import pytest

from oci.extraction.complete_paged import (
    COMPLETE_PAGED_RESPONSE_SCHEMA,
    CompleteFeatureContract,
    CompletePageResponse,
    CompletePagingGeometry,
    build_complete_paged_coverage_ledger,
    plan_complete_note_pages,
    plan_complete_paged_requests,
    reconcile_complete_page_responses,
    validate_absolute_citations,
    validate_complete_note_page_plan,
)


def test_page_cores_cover_every_character_once_with_bounded_overlap():
    text = "0123456789" * 4000
    geometry = CompletePagingGeometry(
        core_chars=13_488,
        context_chars=256,
        max_page_chars=14_000,
    )
    pages = plan_complete_note_pages(text, geometry=geometry)
    assert "".join(text[p.core_start:p.core_end] for p in pages) == text
    assert all(
        p.context_end - p.context_start <= geometry.max_page_chars
        for p in pages
    )
    validate_complete_note_page_plan(text, pages, geometry=geometry)


def test_page_geometry_is_required_config_and_arbitrary_sizes_remain_lossless():
    text = "clinical evidence " * 37
    geometry = CompletePagingGeometry(
        core_chars=47,
        context_chars=9,
        max_page_chars=65,
    )
    pages = plan_complete_note_pages(text, geometry=geometry)
    assert "".join(text[page.core_start:page.core_end] for page in pages) == text
    validate_complete_note_page_plan(text, pages, geometry=geometry)


def test_citations_are_absolute_and_overlap_is_deduplicated():
    text = "abc target xyz"
    citation = {"start": 4, "end": 10, "text": "target"}
    assert len(validate_absolute_citations(text, [citation, citation])) == 1
    with pytest.raises(ValueError, match="exact absolute"):
        validate_absolute_citations(text, [{**citation, "text": "wrong!"}])


@pytest.mark.parametrize(
    "citation",
    (
        {"start": "4", "end": 10, "text": "target"},
        {"start": 4, "end": 10.0, "text": "target"},
        {"start": True, "end": 10, "text": "target"},
        {"start": 4, "end": 10, "text": "target", "page": 0},
    ),
)
def test_citations_use_a_closed_schema_without_offset_coercion(citation):
    with pytest.raises(ValueError, match="closed-schema|exact absolute"):
        validate_absolute_citations("abc target xyz", [citation])


def test_authenticated_normalized_citation_schema_is_separate_and_hash_checked():
    text = "abc target xyz"
    normalized = {
        "start": 4,
        "end": 10,
        "text": "target",
        "sha256": "34a04005bcaf206eec990bd9637d9fdb6725e0a0c0d4aebf003f17f4c956eb5c",
    }
    assert validate_absolute_citations(
        text,
        [normalized],
        authenticated_citations=True,
    )[0] == normalized
    with pytest.raises(ValueError, match="SHA-256"):
        validate_absolute_citations(
            text,
            [{**normalized, "sha256": "0" * 64}],
            authenticated_citations=True,
        )
    with pytest.raises(ValueError, match="closed-schema"):
        validate_absolute_citations(
            text,
            [normalized],
            authenticated_citations=False,
        )


def _positive_response(*, start: int, text: str) -> CompletePageResponse:
    quote = text[start : start + 1]
    return CompletePageResponse.validate(
        {
            "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
            "status": "positive",
            "normalized_value": True,
            "reason": None,
            "citations": [
                {
                    "start": start,
                    "end": start + 1,
                    "text": quote,
                }
            ],
        },
        text=text,
        page=None,
    )


def test_recursive_reconciliation_accounts_for_all_leaves():
    text = "abcdefghijk"
    leaves = tuple(
        (f"leaf-{index}", _positive_response(start=index, text=text))
        for index in range(len(text))
    )
    seen_child_ids = []

    def keep_first(children):
        seen_child_ids.extend(child["node_id"] for child in children)
        return {
            "child_ids": [child["node_id"] for child in children],
            **children[0]["response"],
        }

    result, ledger = reconcile_complete_page_responses(
        leaves,
        reducer=keep_first,
        fan_in=4,
    )
    assert result.status == "positive"
    assert ledger["leaf_count"] == len(text)
    assert max(node["depth"] for node in ledger["nodes"]) == 1
    assert ledger["every_child_referenced_exactly_once"] is True
    assert set(f"leaf-{index}" for index in range(len(text))) <= set(seen_child_ids)
    assert len(seen_child_ids) == len(set(seen_child_ids))


def test_reconciliation_cannot_invent_or_mutate_authenticated_citations():
    text = "abc"
    leaves = (
        ("leaf-a", _positive_response(start=0, text=text)),
        ("leaf-b", _positive_response(start=1, text=text)),
    )

    def invented(children):
        return {
            "child_ids": [child["node_id"] for child in children],
            "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
            "status": "positive",
            "normalized_value": True,
            "reason": None,
            "citations": [
                dict(_positive_response(start=2, text=text).citations[0])
            ],
        }

    with pytest.raises(ValueError, match="invented"):
        reconcile_complete_page_responses(
            leaves,
            reducer=invented,
            fan_in=2,
        )


@pytest.mark.parametrize(
    ("status", "normalized_value", "reason", "match"),
    [
        ("positive", None, None, "positive reconciliation"),
        ("missing", True, "not found", "missing/ambiguous"),
        ("ambiguous", None, None, "missing/ambiguous"),
        ("negative", None, "failed", "negative reconciliation"),
        ("unknown", None, None, "status is invalid"),
    ],
)
def test_reconciliation_revalidates_closed_status_semantics(
    status,
    normalized_value,
    reason,
    match,
):
    text = "ab"
    leaves = (
        ("leaf-a", _positive_response(start=0, text=text)),
        ("leaf-b", _positive_response(start=1, text=text)),
    )

    def invalid(children):
        return {
            "child_ids": [child["node_id"] for child in children],
            "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
            "status": status,
            "normalized_value": normalized_value,
            "reason": reason,
            "citations": [],
        }

    with pytest.raises(ValueError, match=match):
        reconcile_complete_page_responses(
            leaves,
            reducer=invalid,
            fan_in=2,
        )


def test_reconciliation_accepts_only_exact_ordered_children_and_leaf_witnesses():
    text = "ab"
    leaves = (
        ("leaf-a", _positive_response(start=0, text=text)),
        ("leaf-b", _positive_response(start=1, text=text)),
    )

    def valid(children):
        return {
            "child_ids": [child["node_id"] for child in children],
            **children[0]["response"],
        }

    result, ledger = reconcile_complete_page_responses(
        leaves,
        reducer=valid,
        fan_in=2,
    )
    assert result.status == "positive"
    assert [citation["text"] for citation in result.citations] == ["a"]
    assert ledger["leaf_count"] == 2
    assert ledger["every_child_referenced_exactly_once"] is True

    def reordered(children):
        return {
            "child_ids": [
                child["node_id"]
                for child in reversed(children)
            ],
            **children[0]["response"],
        }

    with pytest.raises(ValueError, match="every child"):
        reconcile_complete_page_responses(
            leaves,
            reducer=reordered,
            fan_in=2,
        )


def test_request_plan_covers_every_note_feature_page_and_records_geometry():
    notes = {
        "p1": "a" * 101,
        "p2": "b" * 17,
    }
    geometry = CompletePagingGeometry(
        core_chars=23,
        context_chars=4,
        max_page_chars=31,
    )
    feature = CompleteFeatureContract(
        name="marker",
        value_type="boolean",
        description="configured marker",
        temporal_rule="use evidence before treatment only",
        aggregation_rule="ever documented",
    )
    plan = plan_complete_paged_requests(
        notes,
        (feature,),
        geometry=geometry,
    )
    expected_count = sum(
        len(plan_complete_note_pages(text, geometry=geometry))
        for text in notes.values()
    )
    assert len(plan.requests) == expected_count
    assert plan.as_dict()["geometry"] == geometry.as_dict()
    terminal = {
        request.request_id: {
            "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
            "status": "negative",
            "normalized_value": None,
            "reason": None,
            "citations": [],
        }
        for request in plan.requests
    }
    ledger = build_complete_paged_coverage_ledger(plan, terminal)
    assert ledger["planned_request_count"] == expected_count
    assert ledger["every_planned_request_accounted_for_exactly_once"] is True


def test_request_plan_has_no_note_or_page_count_cap():
    notes = {
        f"patient-{index}": ("evidence " * (index + 1))
        for index in range(37)
    }
    geometry = CompletePagingGeometry(
        core_chars=29,
        context_chars=3,
        max_page_chars=35,
    )
    feature = CompleteFeatureContract(
        name="marker",
        value_type="boolean",
        description="configured marker",
        temporal_rule="use evidence before treatment only",
        aggregation_rule="ever documented",
    )
    plan = plan_complete_paged_requests(
        notes,
        (feature,),
        geometry=geometry,
    )
    expected_pages = sum(
        len(plan_complete_note_pages(text, geometry=geometry))
        for text in notes.values()
    )
    assert plan.patient_count == len(notes)
    assert plan.note_page_count == expected_pages
    assert len(plan.requests) == expected_pages


def test_empty_note_cannot_disappear_from_the_page_ledger():
    geometry = CompletePagingGeometry(
        core_chars=29,
        context_chars=3,
        max_page_chars=35,
    )
    feature = CompleteFeatureContract(
        name="marker",
        value_type="boolean",
        description="configured marker",
        temporal_rule="use evidence before treatment only",
        aggregation_rule="ever documented",
    )
    with pytest.raises(ValueError, match="neutral marker"):
        plan_complete_paged_requests(
            {"patient-with-unprepared-empty-note": ""},
            (feature,),
            geometry=geometry,
        )
