import pytest

from oci.extraction.complete_paged import (
    COMPLETE_PAGED_RESPONSE_SCHEMA,
    CompletePageResponse,
    CompletePagingGeometry,
    plan_complete_note_pages,
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


def test_citations_are_absolute_and_overlap_is_deduplicated():
    text = "abc target xyz"
    citation = {"start": 4, "end": 10, "text": "target"}
    assert len(validate_absolute_citations(text, [citation, citation])) == 1
    with pytest.raises(ValueError, match="exact absolute"):
        validate_absolute_citations(text, [{**citation, "text": "wrong!"}])


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
