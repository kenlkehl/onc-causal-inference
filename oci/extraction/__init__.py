# oci/extraction/__init__.py
"""Clinical-text extraction tools.

This module provides LLM-based extraction of explicit features from clinical text.
"""

from .explicit_features import (
    ExplicitFeatureValue,
    VLLMFeatureExtractor,
    build_extraction_prompt,
    build_extraction_repair_prompt,
    infer_vllm_reasoning_parser,
    parse_extraction_response,
    resolve_vllm_reasoning_parser,
    strip_reasoning_trace,
)
from .cache import ExtractionCache
from .contract_lexical_context import (
    CONTRACT_LEXICAL_CONTEXT_VERSION,
    EXTRACTION_GROUPING_VERSION,
    ContractLexicalContext,
    RetrievedContractExcerpt,
    compact_contract_lexical_context,
)
from .complete_paged import (
    COMPLETE_PAGED_COVERAGE_LEDGER_SCHEMA,
    COMPLETE_PAGED_RECONCILIATION_SCHEMA,
    COMPLETE_PAGED_REQUEST_PLAN_SCHEMA,
    COMPLETE_PAGED_RESPONSE_SCHEMA,
    COMPLETE_PAGED_VERSION,
    CompleteFeatureContract,
    CompleteNotePage,
    CompletePageResponse,
    CompletePagingGeometry,
    CompletePagedRequest,
    CompletePagedRequestPlan,
    build_complete_page_prompt,
    build_complete_paged_coverage_ledger,
    plan_complete_note_pages,
    plan_complete_paged_requests,
    reconcile_complete_page_responses,
    validate_absolute_citations,
    validate_complete_note_page_plan,
    validate_complete_paged_request_plan,
)
from .llm_routing import (
    OpenAIClientPool,
    call_with_exponential_backoff,
    parse_server_urls,
)

__all__ = [
    "ExplicitFeatureValue",
    "VLLMFeatureExtractor",
    "build_extraction_prompt",
    "build_extraction_repair_prompt",
    "infer_vllm_reasoning_parser",
    "parse_extraction_response",
    "resolve_vllm_reasoning_parser",
    "strip_reasoning_trace",
    "ExtractionCache",
    "CONTRACT_LEXICAL_CONTEXT_VERSION",
    "EXTRACTION_GROUPING_VERSION",
    "ContractLexicalContext",
    "RetrievedContractExcerpt",
    "compact_contract_lexical_context",
    "COMPLETE_PAGED_VERSION",
    "COMPLETE_PAGED_COVERAGE_LEDGER_SCHEMA",
    "COMPLETE_PAGED_RECONCILIATION_SCHEMA",
    "COMPLETE_PAGED_REQUEST_PLAN_SCHEMA",
    "COMPLETE_PAGED_RESPONSE_SCHEMA",
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
    "OpenAIClientPool",
    "call_with_exponential_backoff",
    "parse_server_urls",
]
