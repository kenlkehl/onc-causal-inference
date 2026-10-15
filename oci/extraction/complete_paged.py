"""Lossless complete-note paging and citation/reconciliation validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

COMPLETE_PAGED_VERSION = "complete_paged_v1"
COMPLETE_PAGED_REQUEST_PLAN_SCHEMA = "complete_paged_request_plan_v1"
COMPLETE_PAGED_RESPONSE_SCHEMA = "complete_paged_closed_response_v1"
COMPLETE_PAGED_RECONCILIATION_SCHEMA = "complete_paged_reconciliation_v1"
COMPLETE_PAGED_COVERAGE_LEDGER_SCHEMA = "complete_paged_terminal_coverage_ledger_v1"
PAGE_RESPONSE_STATUSES = frozenset({"positive", "negative", "missing", "ambiguous"})


@dataclass(frozen=True)
class CompletePagingGeometry:
    """Scientific page geometry supplied by configuration.

    There are intentionally no source-code defaults.  A deployment must choose
    geometry appropriate for its tokenizer/model context and record that choice
    in the scientific identity.  The geometry bounds an individual request; it
    never bounds how much of a note is processed.
    """

    core_chars: int
    context_chars: int
    max_page_chars: int

    def __post_init__(self) -> None:
        for name in ("core_chars", "context_chars", "max_page_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"complete paging {name} must be an integer")
        if self.core_chars < 1:
            raise ValueError("complete paging core_chars must be positive")
        if self.context_chars < 0:
            raise ValueError("complete paging context_chars must be nonnegative")
        if self.max_page_chars < 1:
            raise ValueError("complete paging max_page_chars must be positive")
        if self.core_chars + 2 * self.context_chars > self.max_page_chars:
            raise ValueError(
                "complete paging core plus two-sided context exceeds max_page_chars"
            )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @property
    def content_sha256(self) -> str:
        return _value_sha(self.as_dict())


@dataclass(frozen=True)
class CompleteNotePage:
    page_index: int
    core_start: int
    core_end: int
    context_start: int
    context_end: int
    text_sha256: str
    core_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _value_sha(value: Any) -> str:
    return _sha(_canonical(value))


def plan_complete_note_pages(
    text: str,
    *,
    geometry: CompletePagingGeometry,
) -> tuple[CompleteNotePage, ...]:
    if not isinstance(geometry, CompletePagingGeometry):
        raise TypeError("complete-note paging requires configured CompletePagingGeometry")
    pages: list[CompleteNotePage] = []
    for index, start in enumerate(range(0, len(text), geometry.core_chars)):
        end = min(len(text), start + geometry.core_chars)
        context_start = max(0, start - geometry.context_chars)
        context_end = min(len(text), end + geometry.context_chars)
        rendered = text[context_start:context_end]
        if len(rendered) > geometry.max_page_chars:
            raise RuntimeError("complete-note page exceeds its configured character bound")
        pages.append(
            CompleteNotePage(
                index,
                start,
                end,
                context_start,
                context_end,
                _sha(rendered),
                _sha(text[start:end]),
            )
        )
    validate_complete_note_page_plan(text, pages, geometry=geometry)
    return tuple(pages)


def validate_complete_note_page_plan(
    text: str,
    pages: Sequence[CompleteNotePage],
    *,
    geometry: CompletePagingGeometry,
) -> None:
    if not isinstance(geometry, CompletePagingGeometry):
        raise TypeError("page-plan validation requires configured CompletePagingGeometry")
    expected = 0
    for index, page in enumerate(pages):
        expected_end = min(len(text), expected + geometry.core_chars)
        if (
            page.page_index != index
            or page.core_start != expected
            or page.core_end != expected_end
            or page.context_start != max(0, expected - geometry.context_chars)
            or page.context_end
            != min(len(text), expected_end + geometry.context_chars)
        ):
            raise ValueError("page cores are missing, duplicated, or reordered")
        if not (page.context_start <= page.core_start < page.core_end <= page.context_end):
            raise ValueError("invalid core/context page bounds")
        rendered = text[page.context_start:page.context_end]
        if (
            len(rendered) > geometry.max_page_chars
            or _sha(rendered) != page.text_sha256
        ):
            raise ValueError("page text changed or exceeds its bound")
        if _sha(text[page.core_start:page.core_end]) != page.core_sha256:
            raise ValueError("page core changed")
        expected = page.core_end
    if expected != len(text) or (text and not pages):
        raise ValueError("page cores do not cover the complete note exactly once")


def validate_absolute_citations(
    text: str,
    citations: Sequence[Mapping[str, Any]],
    *,
    page: CompleteNotePage | None = None,
    authenticated_citations: bool = False,
) -> tuple[dict[str, Any], ...]:
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    expected_fields = (
        {"start", "end", "text", "sha256"}
        if authenticated_citations
        else {"start", "end", "text"}
    )
    for citation in citations:
        if not isinstance(citation, Mapping) or set(citation) != expected_fields:
            raise ValueError("citation is not one closed-schema absolute witness")
        start, end = citation["start"], citation["end"]
        quote = citation["text"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not isinstance(quote, str)
            or not (0 <= start < end <= len(text))
            or text[start:end] != quote
        ):
            raise ValueError("citation cannot be located at its exact absolute offset")
        if authenticated_citations and citation["sha256"] != _sha(quote):
            raise ValueError("authenticated citation SHA-256 does not match its quote")
        if page is not None and not (page.context_start <= start and end <= page.context_end):
            raise ValueError("citation lies outside the page supplied to the model")
        unique[(start, end)] = {"start": start, "end": end, "text": quote, "sha256": _sha(quote)}
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True)
class CompleteFeatureContract:
    name: str
    value_type: str
    description: str
    temporal_rule: str
    aggregation_rule: str
    categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.name,
                self.value_type,
                self.description,
                self.temporal_rule,
                self.aggregation_rule,
            )
        ):
            raise ValueError("complete-page feature contracts require nonempty fields")
        if self.value_type not in {"categorical", "continuous", "boolean"}:
            raise ValueError("complete-page feature contract has unsupported value_type")
        if self.value_type == "categorical" and not self.categories:
            raise ValueError("categorical complete-page contracts require categories")
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("complete-page feature categories are duplicated")

    @property
    def contract_sha256(self) -> str:
        return _value_sha(asdict(self))


@dataclass(frozen=True)
class CompletePagedRequest:
    request_id: str
    patient_id: str
    note_sha256: str
    feature_name: str
    feature_contract_sha256: str
    page: CompleteNotePage
    prompt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "page": self.page.as_dict()}


@dataclass(frozen=True)
class CompletePagedRequestPlan:
    requests: tuple[CompletePagedRequest, ...]
    patient_count: int
    feature_count: int
    note_page_count: int
    geometry: CompletePagingGeometry

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": COMPLETE_PAGED_REQUEST_PLAN_SCHEMA,
            "patient_count": self.patient_count,
            "feature_count": self.feature_count,
            "note_page_count": self.note_page_count,
            "planned_request_count": len(self.requests),
            "geometry": self.geometry.as_dict(),
            "geometry_sha256": self.geometry.content_sha256,
            "requests": [request.as_dict() for request in self.requests],
            "raw_note_copies_persisted": False,
            "complete_note_text_truncation_allowed": False,
        }
        return {**body, "content_sha256": _value_sha(body)}


def build_complete_page_prompt(
    text: str,
    *,
    page: CompleteNotePage,
    feature: CompleteFeatureContract,
    geometry: CompletePagingGeometry,
) -> str:
    """Build one page request containing exactly one feature contract."""

    if not isinstance(geometry, CompletePagingGeometry):
        raise TypeError("complete-page prompting requires configured geometry")
    expected_start = page.page_index * geometry.core_chars
    expected_end = min(len(text), expected_start + geometry.core_chars)
    if (
        page.page_index < 0
        or page.core_start != expected_start
        or page.core_end != expected_end
        or page.context_start
        != max(0, expected_start - geometry.context_chars)
        or page.context_end
        != min(len(text), expected_end + geometry.context_chars)
    ):
        raise ValueError("complete-page prompt received a noncanonical page")
    rendered = text[page.context_start:page.context_end]
    if (
        len(rendered) > geometry.max_page_chars
        or _sha(rendered) != page.text_sha256
    ):
        raise ValueError("complete-page prompt source changed")
    contract = {
        "name": feature.name,
        "value_type": feature.value_type,
        "description": feature.description,
        "categories": list(feature.categories),
        "temporal_rule": feature.temporal_rule,
        "aggregation_rule": feature.aggregation_rule,
    }
    response_shape = {
        "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
        "status": "positive|negative|missing|ambiguous",
        "normalized_value": None,
        "reason": None,
        "citations": [
            {"start": 0, "end": 1, "text": "exact prepared-text substring"}
        ],
    }
    return (
        "Extract exactly one feature from this bounded page of one complete "
        "prepared note. Offsets in citations MUST be absolute offsets into the "
        "complete prepared note. A positive result requires at least one exact "
        "citation. Do not infer from information outside this page.\n"
        f"feature_contract={_canonical(contract)}\n"
        f"configured_page_geometry={_canonical(geometry.as_dict())}\n"
        f"page_bounds={_canonical(page.as_dict())}\n"
        f"closed_response_schema={_canonical(response_shape)}\n"
        f"page_text_absolute_base={page.context_start}\n"
        f"page_text={rendered}"
    )


def plan_complete_paged_requests(
    notes: Mapping[Any, str],
    features: Sequence[CompleteFeatureContract],
    *,
    geometry: CompletePagingGeometry,
) -> CompletePagedRequestPlan:
    """Compute the complete request plan before any extraction call."""

    if not notes or not features:
        raise ValueError("complete-page request planning requires notes and features")
    if len({feature.name for feature in features}) != len(features):
        raise ValueError("complete-page feature names are duplicated")
    requests: list[CompletePagedRequest] = []
    page_count = 0
    for raw_patient_id, raw_text in notes.items():
        patient_id = str(raw_patient_id)
        text = str(raw_text)
        if not patient_id:
            raise ValueError("complete-page patient IDs cannot be empty")
        if not text:
            raise ValueError(
                "complete-page notes cannot be empty; text preparation must "
                "materialize the configured neutral marker first"
            )
        pages = plan_complete_note_pages(text, geometry=geometry)
        page_count += len(pages)
        note_sha = _sha(text)
        for feature in features:
            for page in pages:
                prompt = build_complete_page_prompt(
                    text,
                    page=page,
                    feature=feature,
                    geometry=geometry,
                )
                request_body = {
                    "patient_id": patient_id,
                    "note_sha256": note_sha,
                    "feature_name": feature.name,
                    "feature_contract_sha256": feature.contract_sha256,
                    "page": page.as_dict(),
                    "prompt_sha256": _sha(prompt),
                }
                requests.append(
                    CompletePagedRequest(
                        request_id=_value_sha(request_body),
                        patient_id=patient_id,
                        note_sha256=note_sha,
                        feature_name=feature.name,
                        feature_contract_sha256=feature.contract_sha256,
                        page=page,
                        prompt_sha256=request_body["prompt_sha256"],
                    )
                )
    if len({request.request_id for request in requests}) != len(requests):
        raise RuntimeError("complete-page request IDs collided")
    plan = CompletePagedRequestPlan(
        requests=tuple(requests),
        patient_count=len(notes),
        feature_count=len(features),
        note_page_count=page_count,
        geometry=geometry,
    )
    validate_complete_paged_request_plan(plan, notes=notes, features=features)
    return plan


def validate_complete_paged_request_plan(
    plan: CompletePagedRequestPlan,
    *,
    notes: Mapping[Any, str],
    features: Sequence[CompleteFeatureContract],
) -> None:
    expected = plan_complete_paged_requests_without_validation(
        notes,
        features,
        geometry=plan.geometry,
    )
    if plan.as_dict() != expected.as_dict():
        raise ValueError("complete-page request plan changed or is incomplete")


def plan_complete_paged_requests_without_validation(
    notes: Mapping[Any, str],
    features: Sequence[CompleteFeatureContract],
    *,
    geometry: CompletePagingGeometry,
) -> CompletePagedRequestPlan:
    """Internal non-recursive reconstruction used by the plan validator."""

    requests: list[CompletePagedRequest] = []
    page_count = 0
    for raw_patient_id, raw_text in notes.items():
        patient_id, text = str(raw_patient_id), str(raw_text)
        if not patient_id or not text:
            raise ValueError(
                "complete-page plan validation requires nonempty IDs and "
                "prepared marker-backed notes"
            )
        pages = plan_complete_note_pages(text, geometry=geometry)
        page_count += len(pages)
        for feature in features:
            for page in pages:
                prompt_sha = _sha(
                    build_complete_page_prompt(
                        text,
                        page=page,
                        feature=feature,
                        geometry=geometry,
                    )
                )
                request_body = {
                    "patient_id": patient_id,
                    "note_sha256": _sha(text),
                    "feature_name": feature.name,
                    "feature_contract_sha256": feature.contract_sha256,
                    "page": page.as_dict(),
                    "prompt_sha256": prompt_sha,
                }
                requests.append(
                    CompletePagedRequest(
                        request_id=_value_sha(request_body),
                        patient_id=patient_id,
                        note_sha256=_sha(text),
                        feature_name=feature.name,
                        feature_contract_sha256=feature.contract_sha256,
                        page=page,
                        prompt_sha256=prompt_sha,
                    )
                )
    return CompletePagedRequestPlan(
        tuple(requests),
        len(notes),
        len(features),
        page_count,
        geometry,
    )


@dataclass(frozen=True)
class CompletePageResponse:
    schema_version: str
    status: str
    normalized_value: Any
    reason: str | None
    citations: tuple[Mapping[str, Any], ...]

    @classmethod
    def validate(
        cls,
        value: Mapping[str, Any],
        *,
        text: str,
        page: CompleteNotePage | None,
        authenticated_citations: bool = False,
    ) -> "CompletePageResponse":
        required = {
            "schema_version",
            "status",
            "normalized_value",
            "reason",
            "citations",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("complete-page response is not one closed-schema object")
        status = value["status"]
        if (
            value["schema_version"] != COMPLETE_PAGED_RESPONSE_SCHEMA
            or status not in PAGE_RESPONSE_STATUSES
        ):
            raise ValueError("complete-page response schema/status is invalid")
        reason = value["reason"]
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("complete-page response reason must be null or nonempty")
        raw_citations = value["citations"]
        if not isinstance(raw_citations, list):
            raise ValueError("complete-page citations must be a list")
        citations = validate_absolute_citations(
            text,
            raw_citations,
            page=page,
            authenticated_citations=authenticated_citations,
        )
        if status == "positive":
            if value["normalized_value"] is None or not citations or reason is not None:
                raise ValueError(
                    "positive complete-page responses require a value and citations"
                )
        elif status in {"missing", "ambiguous"}:
            if value["normalized_value"] is not None or reason is None:
                raise ValueError(
                    "missing/ambiguous complete-page responses require a reason and null value"
                )
        elif status == "negative" and reason is not None:
            raise ValueError("negative complete-page responses cannot carry a failure reason")
        return cls(
            schema_version=COMPLETE_PAGED_RESPONSE_SCHEMA,
            status=str(status),
            normalized_value=value["normalized_value"],
            reason=reason,
            citations=citations,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "normalized_value": self.normalized_value,
            "reason": self.reason,
            "citations": [dict(value) for value in self.citations],
        }


def _validate_reconciled_response_payload(
    value: Mapping[str, Any],
    *,
    allowed_citations: set[tuple[int, int, str, str]],
) -> dict[str, Any]:
    """Validate a reconciliation result without reopening the prepared note.

    Leaf citations have already been authenticated against the complete
    prepared text.  A reconciliation node may retain or discard those
    witnesses, but it may not create or alter one.
    """

    required = {
        "schema_version",
        "status",
        "normalized_value",
        "reason",
        "citations",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("complete-page reconciliation response schema is not closed")
    if value["schema_version"] != COMPLETE_PAGED_RESPONSE_SCHEMA:
        raise ValueError("complete-page reconciliation response schema changed")
    status = value["status"]
    if status not in PAGE_RESPONSE_STATUSES:
        raise ValueError("complete-page reconciliation response status is invalid")
    reason = value["reason"]
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError(
            "complete-page reconciliation reason must be null or nonempty"
        )
    raw_citations = value["citations"]
    if not isinstance(raw_citations, list):
        raise ValueError("complete-page reconciliation citations must be a list")

    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for citation in raw_citations:
        if not isinstance(citation, Mapping) or set(citation) != {
            "start",
            "end",
            "text",
            "sha256",
        }:
            raise ValueError(
                "complete-page reconciliation citation schema is not closed"
            )
        start, end = citation["start"], citation["end"]
        quote, quote_sha = citation["text"], citation["sha256"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or not isinstance(quote, str)
            or not quote
            or not isinstance(quote_sha, str)
            or quote_sha != _sha(quote)
        ):
            raise ValueError("complete-page reconciliation citation is invalid")
        witness = (start, end, quote, quote_sha)
        if witness not in allowed_citations:
            raise ValueError("complete-page reconciliation invented a new citation")
        key = (start, end)
        canonical = {
            "start": start,
            "end": end,
            "text": quote,
            "sha256": quote_sha,
        }
        if key in unique and unique[key] != canonical:
            raise ValueError(
                "complete-page reconciliation has conflicting offset witnesses"
            )
        unique[key] = canonical

    citations = [unique[key] for key in sorted(unique)]
    normalized_value = value["normalized_value"]
    if status == "positive":
        if normalized_value is None or not citations or reason is not None:
            raise ValueError(
                "positive reconciliation responses require a value and citations"
            )
    elif status in {"missing", "ambiguous"}:
        if normalized_value is not None or reason is None:
            raise ValueError(
                "missing/ambiguous reconciliation responses require a reason "
                "and null value"
            )
    elif status == "negative" and reason is not None:
        raise ValueError("negative reconciliation responses cannot carry a reason")

    # Reject NaN, infinities, and other values that cannot enter the canonical
    # authenticated ledger.
    _canonical(normalized_value)
    return {
        "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
        "status": status,
        "normalized_value": normalized_value,
        "reason": reason,
        "citations": citations,
    }


def reconcile_complete_page_responses(
    responses: Sequence[tuple[str, CompletePageResponse]],
    *,
    reducer: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
    fan_in: int,
) -> tuple[CompletePageResponse, Mapping[str, Any]]:
    """Reconcile through a fan-in tree whose nodes reference every child once."""

    if not responses or fan_in < 2:
        raise ValueError("complete-page reconciliation requires leaves and fan_in >= 2")
    if len({request_id for request_id, _ in responses}) != len(responses):
        raise ValueError("complete-page reconciliation leaf IDs are duplicated")
    level = [
        {
            "node_id": request_id,
            "response": response.as_dict(),
        }
        for request_id, response in responses
    ]
    nodes: list[dict[str, Any]] = []
    depth = 0
    while len(level) > 1:
        next_level: list[dict[str, Any]] = []
        for start in range(0, len(level), fan_in):
            children = level[start : start + fan_in]
            expected_ids = [child["node_id"] for child in children]
            allowed_citations = {
                (
                    int(citation["start"]),
                    int(citation["end"]),
                    str(citation["text"]),
                    str(citation["sha256"]),
                )
                for child in children
                for citation in child["response"]["citations"]
            }
            reduced = reducer(tuple(children))
            if not isinstance(reduced, Mapping):
                raise ValueError("complete-page reconciliation reducer returned no object")
            if set(reduced) != {
                "child_ids",
                "schema_version",
                "status",
                "normalized_value",
                "reason",
                "citations",
            }:
                raise ValueError("complete-page reconciliation node schema is not closed")
            if reduced["child_ids"] != expected_ids:
                raise ValueError(
                    "complete-page reconciliation did not reference every child exactly once"
                )
            response_payload = {
                key: reduced[key]
                for key in (
                    "schema_version",
                    "status",
                    "normalized_value",
                    "reason",
                    "citations",
                )
            }
            response_payload = _validate_reconciled_response_payload(
                response_payload,
                allowed_citations=allowed_citations,
            )
            node_body = {
                "depth": depth,
                "child_ids": expected_ids,
                "response": response_payload,
            }
            node_id = _value_sha(node_body)
            nodes.append({**node_body, "node_id": node_id})
            next_level.append({"node_id": node_id, "response": response_payload})
        level = next_level
        depth += 1
    final_payload = level[0]["response"]
    final = CompletePageResponse(
        schema_version=COMPLETE_PAGED_RESPONSE_SCHEMA,
        status=final_payload["status"],
        normalized_value=final_payload["normalized_value"],
        reason=final_payload["reason"],
        citations=tuple(final_payload["citations"]),
    )
    body = {
        "schema_version": COMPLETE_PAGED_RECONCILIATION_SCHEMA,
        "leaf_count": len(responses),
        "fan_in": fan_in,
        "nodes": nodes,
        "root_node_id": level[0]["node_id"],
        "every_child_referenced_exactly_once": True,
    }
    return final, {**body, "content_sha256": _value_sha(body)}


def build_complete_paged_coverage_ledger(
    plan: CompletePagedRequestPlan,
    terminal_responses: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    planned_ids = [request.request_id for request in plan.requests]
    if set(terminal_responses) != set(planned_ids) or len(terminal_responses) != len(
        planned_ids
    ):
        raise ValueError("terminal response ledger does not equal the complete request plan")
    rows = []
    for request in plan.requests:
        response = terminal_responses[request.request_id]
        rows.append(
            {
                "request_id": request.request_id,
                "patient_id_sha256": _sha(request.patient_id),
                "feature_name": request.feature_name,
                "page_index": request.page.page_index,
                "prompt_sha256": request.prompt_sha256,
                "normalized_response_sha256": _value_sha(response),
            }
        )
    body = {
        "schema_version": COMPLETE_PAGED_COVERAGE_LEDGER_SCHEMA,
        "planned_request_count": len(planned_ids),
        "terminal_response_count": len(rows),
        "requests": rows,
        "planned_and_terminal_counts_equal": True,
        "every_planned_request_accounted_for_exactly_once": True,
        "raw_note_copies_persisted": False,
    }
    return {**body, "content_sha256": _value_sha(body)}


__all__ = [
    "COMPLETE_PAGED_COVERAGE_LEDGER_SCHEMA",
    "COMPLETE_PAGED_RECONCILIATION_SCHEMA",
    "COMPLETE_PAGED_REQUEST_PLAN_SCHEMA",
    "COMPLETE_PAGED_RESPONSE_SCHEMA",
    "COMPLETE_PAGED_VERSION",
    "CompleteFeatureContract",
    "CompleteNotePage",
    "CompletePageResponse",
    "CompletePagingGeometry",
    "CompletePagedRequest",
    "CompletePagedRequestPlan",
    "build_complete_page_prompt",
    "build_complete_paged_coverage_ledger",
    "plan_complete_note_pages",
    "plan_complete_paged_requests",
    "reconcile_complete_page_responses",
    "validate_absolute_citations",
    "validate_complete_note_page_plan",
    "validate_complete_paged_request_plan",
]
