# oci/extraction/explicit_features.py
"""LLM-based explicit feature extraction from clinical text.

This module extracts researcher-specified feature variables from clinical text
using a large language model (via vLLM). The extracted features are returned
as structured data that can be featurized and used alongside text embeddings
for causal inference.

Three vLLM modes are supported:
- "server": Connect to a running vLLM OpenAI-compatible server
- "start_server": Start vLLM server subprocess, then connect (cleans up after)
- "python_api": Use vLLM Python API directly (no server, in-process inference)

Example usage:
    from oci.extraction.explicit_features import VLLMFeatureExtractor
    from oci.config import ExplicitFeatureSpec

    specs = [
        ExplicitFeatureSpec(
            name="performance_status",
            type="categorical",
            categories=["0", "1", "2", "3", "4"],
            description="ECOG performance status",
            roles=["confounder", "effect_modifier"],
        ),
        ExplicitFeatureSpec(
            name="age_at_diagnosis",
            type="continuous",
            description="Patient age at diagnosis in years",
            roles=["confounder"],
        )
    ]

    extractor = VLLMFeatureExtractor(
        specs=specs,
        mode="python_api",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=2
    )

    results = extractor.extract(clinical_texts)
    # results: List[Dict[str, ExplicitFeatureValue]]
"""

import json
import logging
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from tqdm import tqdm

from ..config import ExplicitFeatureSpec
from ..colbert_config import ColBERTConfig, colbert_config_from_mapping
from .contract_lexical_context import (
    CONTRACT_LEXICAL_CONTEXT_VERSION,
    compact_contract_lexical_context,
)
from .complete_paged import (
    COMPLETE_PAGED_RESPONSE_SCHEMA,
    CompleteFeatureContract,
    CompleteNotePage,
    CompletePageResponse,
    CompletePagingGeometry,
    build_complete_page_prompt,
)
from .llm_routing import (
    OpenAIClientPool,
    google_json_response_format_kwargs,
    is_retryable_llm_exception,
    parse_server_urls,
    retry_delay,
)

logger = logging.getLogger(__name__)

_DISABLED_REASONING_PARSER_VALUES = {"", "none", "off", "false", "disabled", "no"}


def infer_vllm_reasoning_parser(model_name: Optional[str]) -> Optional[str]:
    """Infer the vLLM reasoning parser name from a model name."""
    if not model_name:
        return None
    model_key = str(model_name).lower()
    if "qwen" in model_key:
        return "qwen3"
    if "gemma" in model_key:
        return "gemma4"
    if "gpt-oss" in model_key or "gptoss" in model_key:
        return "openai_gptoss"
    return None


def resolve_vllm_reasoning_parser(
    reasoning_parser: Optional[str],
    model_name: Optional[str],
) -> Optional[str]:
    """Resolve an explicit/auto parser setting to a vLLM parser name."""
    if reasoning_parser is None:
        return None
    value = str(reasoning_parser).strip()
    if value.lower() in _DISABLED_REASONING_PARSER_VALUES:
        return None
    if value.lower() == "auto":
        return infer_vllm_reasoning_parser(model_name)
    return value


def strip_reasoning_trace(response: str) -> str:
    """Remove common inline reasoning blocks before JSON parsing."""
    text = response.strip()
    if not text:
        return text
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    lower_text = text.lower()
    end_marker = "</think>"
    end_idx = lower_text.rfind(end_marker)
    if end_idx >= 0:
        text = text[end_idx + len(end_marker) :].strip()
    return text


def _category_match_key(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("\u2265", ">=").replace("\u2264", "<=")
    text = re.sub(r"[\s_-]+", "", text)
    return text


_UNDECLARED_CATEGORICAL_MISSING_SENTINEL_KEYS = frozenset(
    _category_match_key(value) for value in ("unknown", "not_documented")
)


def _is_null_like(value: Any) -> bool:
    """Accept common quoted JSON-null spellings as conservative missing values."""
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"null", "none", "n/a", "na"}


def _categorical_value_map(spec: ExplicitFeatureSpec) -> Dict[str, str]:
    categories = spec.categories or []
    value_map = {_category_match_key(category): str(category) for category in categories}
    by_category_key = {_category_match_key(category): str(category) for category in categories}
    aliases = getattr(spec, "value_aliases", None) or {}
    if isinstance(aliases, dict):
        for raw_category, raw_aliases in aliases.items():
            category = by_category_key.get(_category_match_key(raw_category))
            if category is None:
                continue
            alias_values = raw_aliases if isinstance(raw_aliases, list) else [raw_aliases]
            for alias in alias_values:
                value_map[_category_match_key(alias)] = category
    return value_map


def _format_value_aliases(spec: ExplicitFeatureSpec) -> str:
    aliases = getattr(spec, "value_aliases", None) or {}
    if not isinstance(aliases, dict):
        return ""
    chunks = []
    for category, raw_aliases in aliases.items():
        alias_values = raw_aliases if isinstance(raw_aliases, list) else [raw_aliases]
        clean_aliases = [str(alias).strip() for alias in alias_values if str(alias).strip()]
        if clean_aliases:
            chunks.append(f'{category}: {", ".join(clean_aliases)}')
    return "; ".join(chunks)


@dataclass
class ExplicitFeatureValue:
    """Extracted value for a single feature."""

    name: str
    type: str  # "categorical" or "continuous"
    value: Optional[Union[str, float]]  # Extracted value (None if missing)
    is_missing: bool  # True if extraction failed after retries


@dataclass
class _ExtractionParseResult:
    values: Dict[str, ExplicitFeatureValue]
    issues: List[str]


def build_extraction_prompt(
    clinical_text: str,
    specs: List[ExplicitFeatureSpec],
    max_text_length: Optional[int] = None,
    context_strategy: str = "tail",
    source_text_temporally_valid_by_design: bool = False,
    colbert: ColBERTConfig | None = None,
) -> str:
    """Build prompt for feature extraction.

    Args:
        clinical_text: Clinical text to extract from
        specs: List of feature specifications
        max_text_length: Maximum context characters to include
        context_strategy: Historical ``tail`` truncation or deterministic
            ``contract_lexical_rag`` retrieval
        source_text_temporally_valid_by_design: Trust source-text timing and do
            not impose a treatment-time eligibility boundary

    Returns:
        Formatted prompt string for the LLM
    """
    instructions = []
    json_fields = []

    for i, spec in enumerate(specs, 1):
        name = spec.name
        conf_type = spec.type
        description = spec.description or name.replace("_", " ").title()

        if conf_type == "categorical":
            categories = spec.categories or []
            cat_list = ", ".join(f'"{c}"' for c in categories)
            alias_text = _format_value_aliases(spec)
            alias_instruction = (
                f"\n   Value aliases to canonicalize: {alias_text}" if alias_text else ""
            )
            missing_scope = (
                "does not state this value in the supplied source text."
                if source_text_temporally_valid_by_design
                else "does not state this value before treatment."
            )
            missing_instruction = (
                '\n   Use "unknown" only when the note explicitly says the value is '
                'unknown or indeterminate. Use "not_documented" when the note '
                f"{missing_scope}"
                if {"unknown", "not_documented"}.issubset(set(categories))
                else ""
            )
            instructions.append(
                f"{i}. {name} (categorical): {description}\n"
                f"   Valid values: {cat_list}"
                f"{missing_instruction}"
                f"{alias_instruction}"
            )
            json_fields.append(f'"{name}": "<category>"')
        else:  # continuous
            instructions.append(
                f"{i}. {name} (continuous): {description}\n" f"   Respond with a numeric value."
            )
            json_fields.append(f'"{name}": <number>')

    instructions_text = "\n".join(instructions)
    json_example = "{" + ", ".join(json_fields) + "}"

    text = str(clinical_text)
    strategy = str(context_strategy).strip().lower().replace("-", "_")
    if strategy == "colbert":
        from .colbert import get_retriever
        retrieval_config = colbert or ColBERTConfig()
        text = get_retriever(retrieval_config).retrieve(text, specs, top_k=retrieval_config.top_k)["context"]
        if max_text_length is not None and len(text) > int(max_text_length):
            raise ValueError("ColBERT retrieved context exceeds extraction_max_text_length; increase the budget or reduce top_k")
    elif strategy == "tail":
        # Preserve the historical behavior exactly for backward compatibility.
        if max_text_length is not None and len(text) > int(max_text_length):
            text = text[-int(max_text_length) :]
    elif strategy == "contract_lexical_rag":
        if max_text_length is None:
            raise ValueError("contract_lexical_rag requires a finite max_text_length")
        text = compact_contract_lexical_context(
            text,
            specs,
            max_chars=int(max_text_length),
        ).text
    elif strategy == "complete_paged_v1":
        # Page construction and reconciliation are owned by the extraction
        # provider. Each call here receives one already bounded complete page.
        if max_text_length is not None and len(text) > int(max_text_length):
            raise ValueError("complete_paged_v1 received an oversized unpaged input")
    else:
        raise ValueError(
            "context_strategy must be 'colbert', 'tail', 'contract_lexical_rag', or "
            "'complete_paged_v1'"
        )

    if strategy == "colbert" or text.startswith("[oci_colbert_v1]"):
        document_instruction = (
            "Read the retrieved excerpts and extract the declared characteristics. "
            "These excerpts are a subset of the patient's record; missing values do not "
            "establish absence from the complete record. Follow temporal and aggregation rules."
        )
        document_label = "ColBERT-retrieved excerpts"
    elif strategy == "contract_lexical_rag":
        document_instruction = (
            "Read every contract-guided verbatim excerpt below and extract the "
            "following characteristics. Excerpts retain their original source order."
        )
        document_label = "Contract-guided retrieved excerpts"
    elif "[neural_query_rag_v1]" in text:
        document_instruction = (
            "Read every retrieved excerpt below and extract the following patient "
            "characteristics according to each contract."
        )
        document_label = "Query-retrieved excerpts"
    else:
        document_instruction = (
            "Read this complete clinical note and extract the following patient " "characteristics."
        )
        document_label = "Clinical Note"

    prompt = f"""{document_instruction}
Follow each extraction contract exactly. Do not guess.
For categorical fields, use a listed category when supported and follow each field's unknown/not_documented policy. Every categorical field may be JSON null when its value is not documented.
For continuous fields that are not explicitly stated or cannot be deterministically converted, return null.

{instructions_text}

{document_label}:
{text}

Respond with JSON only, no other text:
{json_example}"""

    return prompt


def build_extraction_repair_prompt(
    issues: List[str],
    specs: List[ExplicitFeatureSpec],
    *,
    source_text_temporally_valid_by_design: bool = False,
) -> str:
    """Build a follow-up prompt for malformed extraction JSON."""
    issue_text = "\n".join(f"- {issue}" for issue in issues[:10])
    fields = []
    instructions = []
    for spec in specs:
        if spec.type == "categorical":
            categories = ", ".join(f'"{category}"' for category in (spec.categories or []))
            fields.append(f'"{spec.name}": "<category-or-null>"')
            instructions.append(f'- "{spec.name}" must be one of [{categories}] or null.')
        else:
            fields.append(f'"{spec.name}": <number-or-null>')
            instructions.append(f'- "{spec.name}" must be a number or null.')
    shape = "{" + ", ".join(fields) + "}"
    instruction_text = "\n".join(instructions)
    missing_rule = (
        "Use null when a value is unknown, not stated, or cannot be inferred from "
        "the supplied source text."
    )
    return f"""The previous extraction response could not be used.

Problems:
{issue_text}

Repair the response using the original clinical note and extraction instructions already in this conversation.
Return JSON only. Do not include markdown, prose, comments, or reasoning text.
Return exactly one JSON object with exactly these keys:
{shape}

Field rules:
{instruction_text}
{missing_rule}"""


def parse_extraction_response(
    response: str, specs: List[ExplicitFeatureSpec]
) -> Dict[str, ExplicitFeatureValue]:
    """Parse LLM JSON response to extract feature values.

    Args:
        response: Raw LLM response text (expected to be JSON)
        specs: List of feature specifications

    Returns:
        Dictionary mapping feature names to ExplicitFeatureValue objects.
        Categorical values are validated; invalid ones are marked as missing.
        Continuous values that fail parsing are marked as missing.
    """
    return _parse_extraction_response_with_issues(response, specs).values


def _parse_extraction_response_with_issues(
    response: str,
    specs: List[ExplicitFeatureSpec],
) -> _ExtractionParseResult:
    response = strip_reasoning_trace(response or "")

    # Try to extract JSON from response (handle markdown code blocks)
    json_str = _extract_json_object_text(response)
    try:
        parsed = json.loads(json_str)
    except (TypeError, json.JSONDecodeError):
        logger.debug(f"Could not parse JSON response: {response[:200]}")
        return _ExtractionParseResult(
            values=_missing_values_for_specs(specs),
            issues=["malformed JSON response; expected one JSON object"],
        )

    if not isinstance(parsed, dict):
        return _ExtractionParseResult(
            values=_missing_values_for_specs(specs),
            issues=[f"top-level JSON must be an object, got {type(parsed).__name__}"],
        )

    # Validate and extract each feature
    result = {}
    issues: List[str] = []
    for spec in specs:
        name = spec.name
        conf_type = spec.type
        if name not in parsed:
            issues.append(f'missing required key "{name}"')
            result[name] = ExplicitFeatureValue(
                name=name, type=conf_type, value=None, is_missing=True
            )
            continue
        value = parsed.get(name)

        if conf_type == "categorical":
            categories = spec.categories or []
            if value is None:
                result[name] = ExplicitFeatureValue(
                    name=name, type=conf_type, value=None, is_missing=True
                )
            else:
                value_map = _categorical_value_map(spec)
                match_key = _category_match_key(value)
                matched_cat = value_map.get(match_key)
                if matched_cat:
                    result[name] = ExplicitFeatureValue(
                        name=name, type=conf_type, value=matched_cat, is_missing=False
                    )
                elif (
                    _is_null_like(value)
                    or match_key in _UNDECLARED_CATEGORICAL_MISSING_SENTINEL_KEYS
                ):
                    # Fresh models sometimes emit a conventional missing label
                    # even when that label is not part of the contract.  Exact
                    # declared categories and aliases win above; only the two
                    # generic undeclared sentinels fall back to JSON-null
                    # semantics.  Using the category match key intentionally
                    # accepts deterministic whitespace/underscore/hyphen forms.
                    result[name] = ExplicitFeatureValue(
                        name=name, type=conf_type, value=None, is_missing=True
                    )
                else:
                    logger.debug(f"Invalid category '{value}' for {name}, valid: {categories}")
                    issues.append(
                        f'"{name}" has invalid category {value!r}; '
                        f"expected one of {categories} or null"
                    )
                    result[name] = ExplicitFeatureValue(
                        name=name, type=conf_type, value=None, is_missing=True
                    )
        else:  # continuous
            if _is_null_like(value):
                result[name] = ExplicitFeatureValue(
                    name=name, type=conf_type, value=None, is_missing=True
                )
            else:
                try:
                    float_value = float(value)
                    result[name] = ExplicitFeatureValue(
                        name=name, type=conf_type, value=float_value, is_missing=False
                    )
                except (ValueError, TypeError):
                    logger.debug(f"Could not parse continuous value '{value}' for {name}")
                    issues.append(
                        f'"{name}" has non-numeric value {value!r}; expected number or null'
                    )
                    result[name] = ExplicitFeatureValue(
                        name=name, type=conf_type, value=None, is_missing=True
                    )

    extra_keys = sorted(
        str(key) for key in parsed.keys() if key not in {spec.name for spec in specs}
    )
    if extra_keys:
        issues.append(f"unexpected extra key(s): {extra_keys}")

    return _ExtractionParseResult(values=result, issues=issues)


def _missing_values_for_specs(
    specs: List[ExplicitFeatureSpec],
) -> Dict[str, ExplicitFeatureValue]:
    return {
        spec.name: ExplicitFeatureValue(
            name=spec.name,
            type=spec.type,
            value=None,
            is_missing=True,
        )
        for spec in specs
    }


def _extract_json_object_text(response: str) -> str:
    text = str(response or "").strip()
    code_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if code_match:
        candidate = code_match.group(1).strip()
        if candidate:
            return candidate

    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]


class VLLMFeatureExtractor:
    """Extractor for explicit features using vLLM.

    Supports three modes:
    - "server": Connect to running vLLM OpenAI-compatible server
    - "start_server": Start vLLM server subprocess, then connect
    - "python_api": Use vLLM Python API directly (in-process)
    """

    def __init__(
        self,
        specs: List[ExplicitFeatureSpec],
        mode: str = "server",
        server_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        model_names_by_url: Optional[Mapping[str, str]] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        download_dir: Optional[str] = None,
        max_model_len: Optional[int] = None,
        vllm_reasoning_parser: Optional[str] = "auto",
        vllm_enable_thinking: Optional[bool] = None,
        api_key: str = "EMPTY",
        max_retries: int = 3,
        retry_initial_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        retry_backoff_factor: float = 2.0,
        request_timeout: Optional[float] = 900.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_text_length: Optional[int] = None,
        context_strategy: str = "colbert",
        source_text_temporally_valid_by_design: bool = False,
        schema_repair_attempts: Optional[int] = None,
        fail_closed: bool = False,
        colbert: ColBERTConfig | None = None,
    ):
        """Initialize extractor.

        Args:
            specs: List of feature specifications
            mode: "server", "start_server", or "python_api"
            server_url: URL for vLLM server (used in server modes)
            model_name: Model name/path for vLLM
            model_names_by_url: Optional per-endpoint model ids for heterogeneous pools
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory fraction to use
            download_dir: Model download directory
            max_model_len: Maximum model context length (for start_server/python_api)
            vllm_reasoning_parser: vLLM reasoning parser name, "auto", or disabled with None/"none"
            vllm_enable_thinking: Optional chat-template reasoning switch for server requests
            api_key: API key (use "EMPTY" for local vLLM)
            max_retries: Maximum retries per patient before marking as missing
            retry_initial_delay: Initial exponential backoff delay after request failures
            retry_max_delay: Maximum exponential backoff delay after request failures
            retry_backoff_factor: Exponential backoff multiplier
            request_timeout: OpenAI-compatible client request timeout in seconds
            temperature: LLM temperature (0 for deterministic)
            max_tokens: Maximum tokens in response
            max_text_length: Maximum clinical text characters included in prompt
            context_strategy: ``colbert`` (default), ``tail``, or ``contract_lexical_rag``
            source_text_temporally_valid_by_design: Trust source-text timing
                instead of imposing a treatment-time eligibility boundary
        """
        if mode not in ("server", "start_server", "python_api"):
            raise ValueError(
                f"mode must be 'server', 'start_server', or 'python_api', got '{mode}'"
            )

        self.specs = specs
        self.mode = mode
        self.server_urls = parse_server_urls(server_url)
        self.server_url = self.server_urls[0]
        self.model_name = model_name
        supplied_inventory = {
            str(url): str(endpoint_model)
            for url, endpoint_model in (model_names_by_url or {}).items()
        }
        missing_inventory = [
            url for url in self.server_urls if supplied_inventory and url not in supplied_inventory
        ]
        if missing_inventory:
            raise ValueError(
                "model_names_by_url is missing configured endpoint(s): " f"{missing_inventory}"
            )
        self.model_names_by_url = {
            url: supplied_inventory.get(url, str(model_name)) for url in self.server_urls
        }
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.download_dir = download_dir
        self.max_model_len = max_model_len
        self.vllm_reasoning_parser = resolve_vllm_reasoning_parser(
            vllm_reasoning_parser,
            next(iter(self.model_names_by_url.values())),
        )
        self.vllm_enable_thinking = vllm_enable_thinking
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_initial_delay = retry_initial_delay
        self.retry_max_delay = retry_max_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_text_length = max_text_length
        self.context_strategy = str(context_strategy)
        self.colbert = colbert_config_from_mapping(colbert)
        self.source_text_temporally_valid_by_design = bool(source_text_temporally_valid_by_design)
        self.schema_repair_attempts = (
            None if schema_repair_attempts is None else int(schema_repair_attempts)
        )
        if self.schema_repair_attempts is not None and self.schema_repair_attempts < 0:
            raise ValueError("schema_repair_attempts cannot be negative")
        self.fail_closed = bool(fail_closed)

        # These are set lazily
        self._client = None
        self._client_pool: Optional[OpenAIClientPool] = None
        self._llm = None
        self._server_process = None

        logger.info(
            "VLLMFeatureExtractor initialized: mode=%s, model=%s, "
            "endpoint_models=%s, reasoning_parser=%s",
            mode,
            model_name,
            self.model_names_by_url,
            self.vllm_reasoning_parser,
        )
        logger.info(f"Extracting {len(specs)} features: {[s.name for s in specs]}")

    def _init_server_client(self):
        """Initialize OpenAI client for server mode."""
        self._client_pool = OpenAIClientPool(
            server_urls=self.server_urls,
            api_key=self.api_key,
            timeout=self.request_timeout,
            max_retries=0,  # No internal retries (we have our own outer retry loop)
        )
        self._client = self._client_pool.client_for_url(self.server_url)
        logger.info("Configured %s vLLM server endpoint(s)", len(self.server_urls))

    def _start_server(self):
        """Start vLLM server subprocess."""
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_name,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--trust-remote-code",
        ]
        if self.download_dir:
            cmd.extend(["--download-dir", self.download_dir])
        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        if self.vllm_reasoning_parser:
            cmd.extend(["--reasoning-parser", self.vllm_reasoning_parser])

        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        self._server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to be ready
        logger.info("Waiting for vLLM server to start...")
        time.sleep(30)  # Initial wait

        import requests

        for i in range(60):  # Wait up to 5 minutes
            try:
                resp = requests.get(f"{self.server_url.rstrip('/v1')}/health")
                if resp.status_code == 200:
                    logger.info("vLLM server is ready")
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(5)
        else:
            raise RuntimeError("vLLM server failed to start within 5 minutes")

        self._init_server_client()

    def _init_python_api(self):
        """Initialize vLLM Python API."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vllm package required. Install with: pip install vllm")

        logger.info(f"Loading vLLM model: {self.model_name} with TP={self.tensor_parallel_size}")

        kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": True,
        }
        if self.download_dir:
            kwargs["download_dir"] = self.download_dir
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len

        self._llm = LLM(**kwargs)
        logger.info("vLLM model loaded successfully")

    def _ensure_initialized(self):
        """Ensure backend is initialized."""
        if self.mode == "server":
            if self._client is None:
                self._init_server_client()
        elif self.mode == "start_server":
            if self._server_process is None:
                self._start_server()
        elif self.mode == "python_api":
            if self._llm is None:
                self._init_python_api()

    def _extract_single_server(self, text: str) -> Dict[str, ExplicitFeatureValue]:
        """Extract features from single text using server API."""
        prompt = build_extraction_prompt(
            text,
            self.specs,
            max_text_length=self.max_text_length,
            context_strategy=self.context_strategy,
            colbert=self.colbert,
            source_text_temporally_valid_by_design=(self.source_text_temporally_valid_by_design),
        )
        messages = [{"role": "user", "content": prompt}]
        best_result = None
        max_attempts = (
            1 + self.schema_repair_attempts
            if self.schema_repair_attempts is not None
            else max(1, int(self.max_retries))
        )
        start_index = (
            self._client_pool.reserve_start_index() if self._client_pool is not None else 0
        )

        for attempt in range(max_attempts):
            try:
                if self._client_pool is not None:
                    server_url, client = self._client_pool.client_for_attempt(
                        start_index,
                        attempt,
                    )
                    logger.debug("Sending explicit feature extraction request to %s", server_url)
                else:
                    server_url = self.server_url
                    client = self._client
                request_model_name = self.model_names_by_url.get(
                    server_url,
                    self.model_name,
                )
                response_kwargs = {
                    "model": request_model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.vllm_enable_thinking is not None:
                    response_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": bool(self.vllm_enable_thinking)}
                    }
                response_kwargs.update(
                    google_json_response_format_kwargs(
                        api_key=self.api_key,
                        server_url=server_url,
                        model_name=request_model_name,
                    )
                )
                response = client.chat.completions.create(**response_kwargs)
                choice = response.choices[0]
                content = choice.message.content
                if content:
                    parsed = _parse_extraction_response_with_issues(
                        content,
                        self.specs,
                    )
                    result = parsed.values
                    # Track best partial result (fewest missing values)
                    if best_result is None or sum(
                        1 for v in result.values() if not v.is_missing
                    ) > sum(1 for v in best_result.values() if not v.is_missing):
                        best_result = result
                    if parsed.issues and attempt < max_attempts - 1:
                        logger.warning(
                            "Explicit feature extraction response had schema issues "
                            "on attempt %s/%s: finish_reason=%s content_chars=%s "
                            "max_tokens=%s issues=%s. Asking model to repair JSON.",
                            attempt + 1,
                            max_attempts,
                            getattr(choice, "finish_reason", None),
                            len(content),
                            self.max_tokens,
                            "; ".join(parsed.issues[:3]),
                        )
                        messages.extend(
                            [
                                {"role": "assistant", "content": content},
                                {
                                    "role": "user",
                                    "content": build_extraction_repair_prompt(
                                        parsed.issues,
                                        self.specs,
                                        source_text_temporally_valid_by_design=(
                                            self.source_text_temporally_valid_by_design
                                        ),
                                    ),
                                },
                            ]
                        )
                        continue
                    # A schema-complete null is a valid conservative extraction,
                    # not a request failure.  Retrying it pressures the model to
                    # guess undocumented values and can multiply server work by
                    # ``max_retries`` for sparse clinical variables.  Retry only
                    # malformed/incomplete schemas; once parsing has no issues,
                    # accept the result exactly as returned.
                    if not parsed.issues:
                        return result
            except Exception as e:
                if self.fail_closed:
                    raise RuntimeError("production extraction transport failed") from e
                logger.debug(f"Extraction attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1 and is_retryable_llm_exception(e):
                    delay = retry_delay(
                        attempt,
                        initial_delay=self.retry_initial_delay,
                        max_delay=self.retry_max_delay,
                        backoff_factor=self.retry_backoff_factor,
                    )
                    logger.warning(
                        "Explicit feature extraction request failed on attempt "
                        "%s/%s with %s: %s. Retrying in %.2fs.",
                        attempt + 1,
                        max_attempts,
                        e.__class__.__name__,
                        e,
                        delay,
                    )
                    time.sleep(delay)

        # Return best partial result, or all-missing if no successful parse
        if self.fail_closed:
            raise ValueError("production extraction exhausted its single schema repair")
        if best_result is not None:
            return best_result
        return self._missing_result()

    def _complete_page_completion(
        self,
        *,
        prompt: str,
        repair_instruction: str,
        validator: Any,
    ) -> Any:
        """Execute one closed-schema complete-page request plus one repair.

        Complete-note paging uses a response contract distinct from the
        grouped explicit-feature schema.  Keeping this transport on the normal
        extractor means the lossless path works with the same configured
        OpenAI-compatible endpoint pool, without the deleted production/auth
        wrapper.
        """

        if self.mode == "python_api":
            raise ValueError(
                "complete_paged_v1 currently requires server or start_server mode"
            )
        self._ensure_initialized()
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        start_index = (
            self._client_pool.reserve_start_index() if self._client_pool is not None else 0
        )
        first_error: Optional[Exception] = None
        for attempt in range(2):
            if self._client_pool is not None:
                server_url, client = self._client_pool.client_for_attempt(start_index, attempt)
            else:
                server_url, client = self.server_url, self._client
            model_name = self.model_names_by_url.get(server_url, self.model_name)
            request: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.vllm_enable_thinking is not None:
                request["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": bool(self.vllm_enable_thinking)
                    }
                }
            request.update(
                google_json_response_format_kwargs(
                    api_key=self.api_key,
                    server_url=server_url,
                    model_name=model_name,
                )
            )
            response = client.chat.completions.create(**request)
            choice = response.choices[0]
            content = strip_reasoning_trace(str(choice.message.content or ""))
            if not content:
                raise ValueError("complete-page extraction returned an empty response")
            try:
                return validator(_extract_json_object_text(content))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                if attempt:
                    raise ValueError(
                        "complete-page response remained invalid after one schema repair"
                    ) from exc
                first_error = exc
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": repair_instruction},
                    ]
                )
        raise RuntimeError(f"unreachable complete-page response state: {first_error}")

    def extract_complete_page(
        self,
        *,
        text: str,
        page: CompleteNotePage,
        feature: CompleteFeatureContract,
        geometry: CompletePagingGeometry,
    ) -> CompletePageResponse:
        prompt = build_complete_page_prompt(
            text,
            page=page,
            feature=feature,
            geometry=geometry,
        )
        return self._complete_page_completion(
            prompt=prompt,
            repair_instruction=(
                "SCHEMA REPAIR ONLY. Return exactly the closed JSON object specified "
                "above. Preserve exact absolute citations and add no prose or keys."
            ),
            validator=lambda content: CompletePageResponse.validate(
                json.loads(content),
                text=text,
                page=page,
            ),
        )

    def reconcile_complete_pages(
        self,
        *,
        text: str,
        feature: CompleteFeatureContract,
        children: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        child_ids = [str(child["node_id"]) for child in children]
        child_payloads = [
            {
                "child_id": child["node_id"],
                "response": {
                    **dict(child["response"]),
                    "citations": [
                        {
                            "start": citation["start"],
                            "end": citation["end"],
                            "text": citation["text"],
                        }
                        for citation in child["response"].get("citations", ())
                    ],
                },
            }
            for child in children
        ]
        contract = {
            "name": feature.name,
            "value_type": feature.value_type,
            "description": feature.description,
            "categories": list(feature.categories),
            "temporal_rule": feature.temporal_rule,
            "aggregation_rule": feature.aggregation_rule,
        }
        prompt = (
            "Reconcile every child result exactly once under the declared temporal "
            "and aggregation rules. Use only citations already present in children. "
            '{"child_ids":["every required child ID in order"],"schema_version":'
            f'"{COMPLETE_PAGED_RESPONSE_SCHEMA}","status":'
            '"positive|negative|missing|ambiguous","normalized_value":null,'
            '"reason":null,"citations":[{"start":0,"end":1,'
            '"text":"exact prepared-text substring"}]}. Return that closed JSON '
            "object only; citation objects contain start, end, and text only.\n"
            f"feature_contract={json.dumps(contract, sort_keys=True)}\n"
            f"required_child_ids={json.dumps(child_ids)}\n"
            f"children={json.dumps(child_payloads, sort_keys=True)}"
        )
        allowed_citations = {
            (int(citation["start"]), int(citation["end"]), str(citation["text"]))
            for child in children
            for citation in child["response"].get("citations", ())
        }

        def validate(content: str) -> Mapping[str, Any]:
            parsed = json.loads(content)
            expected = {
                "child_ids",
                "schema_version",
                "status",
                "normalized_value",
                "reason",
                "citations",
            }
            if not isinstance(parsed, Mapping) or set(parsed) != expected:
                raise ValueError("complete-page reconciliation schema is not closed")
            if list(map(str, parsed["child_ids"])) != child_ids:
                raise ValueError(
                    "complete-page reconciliation omitted, reordered, or duplicated children"
                )
            response = CompletePageResponse.validate(
                {key: parsed[key] for key in expected if key != "child_ids"},
                text=text,
                page=None,
            )
            observed = {
                (int(citation["start"]), int(citation["end"]), str(citation["text"]))
                for citation in response.citations
            }
            if not observed <= allowed_citations:
                raise ValueError("complete-page reconciliation invented a citation")
            return {"child_ids": child_ids, **response.as_dict()}

        return self._complete_page_completion(
            prompt=prompt,
            repair_instruction=(
                "SCHEMA REPAIR ONLY. Return required child_ids in their exact order "
                "and only the closed JSON object. Citations contain start, end, text."
            ),
            validator=validate,
        )

    def _missing_result(self) -> Dict[str, ExplicitFeatureValue]:
        """Return a missing-value result for every requested feature."""
        return _missing_values_for_specs(self.specs)

    def _extract_batch_python_api(self, texts: List[str]) -> List[Dict[str, ExplicitFeatureValue]]:
        """Extract features from batch using vLLM Python API."""
        from vllm import SamplingParams

        def prepare(text):
            return build_extraction_prompt(
                text, self.specs, max_text_length=self.max_text_length,
                context_strategy=self.context_strategy, colbert=self.colbert,
                source_text_temporally_valid_by_design=self.source_text_temporally_valid_by_design,
            )

        if self.context_strategy == "colbert" and len(texts) > 1:
            with ThreadPoolExecutor(max_workers=min(32, len(texts))) as pool:
                user_contents = list(pool.map(prepare, texts))
        else:
            user_contents = [prepare(text) for text in texts]
        prompts = []
        for user_content in user_contents:
            tokenizer = self._llm.get_tokenizer()

            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_content}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    prompt = f"User: {user_content}\n\nAssistant:"
            else:
                prompt = f"User: {user_content}\n\nAssistant:"
            prompts.append(prompt)

        # Sample params
        sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens)

        # Generate
        logger.info(f"Running vLLM batch inference on {len(prompts)} texts...")
        outputs = self._llm.generate(prompts, sampling_params)

        # Parse results
        results = []
        for output in outputs:
            if output.outputs and len(output.outputs) > 0:
                content = output.outputs[0].text.strip()
                result = parse_extraction_response(content, self.specs)
            else:
                result = self._missing_result()
            results.append(result)

        return results

    def extract(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> List[Dict[str, ExplicitFeatureValue]]:
        """Extract features from a list of clinical texts.

        Args:
            texts: List of clinical text strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of dictionaries mapping feature names to ExplicitFeatureValue
        """
        self._ensure_initialized()

        if self.mode == "python_api":
            # Process all at once (vLLM handles batching internally)
            return self._extract_batch_python_api(texts)

        max_workers = max(1, min(len(texts), int(batch_size or 1)))
        if max_workers == 1:
            results = []
            iterator = tqdm(texts, desc="Extracting features") if show_progress else texts
            for text in iterator:
                results.append(self._extract_single_server(text))
            return results

        logger.info(
            "Running vLLM server extraction with %s concurrent request(s)",
            max_workers,
        )
        results: List[Optional[Dict[str, ExplicitFeatureValue]]] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._extract_single_server, text): idx
                for idx, text in enumerate(texts)
            }
            futures = as_completed(future_to_idx)
            iterator = (
                tqdm(futures, total=len(future_to_idx), desc="Extracting features")
                if show_progress
                else futures
            )
            for future in iterator:
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [result if result is not None else self._missing_result() for result in results]

    def extract_to_dataframe(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> pd.DataFrame:
        """Extract features and return as DataFrame.

        Args:
            texts: List of clinical text strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with columns: explicit_feat_{name}, explicit_feat_{name}_missing
        """
        results = self.extract(texts, batch_size, show_progress)

        # Convert to DataFrame format
        data = {}
        for spec in self.specs:
            values = []
            missing_flags = []
            for result in results:
                val = result.get(spec.name)
                if val:
                    values.append(val.value)
                    missing_flags.append(val.is_missing)
                else:
                    values.append(None)
                    missing_flags.append(True)

            data[f"explicit_feat_{spec.name}"] = values
            data[f"explicit_feat_{spec.name}_missing"] = missing_flags

        return pd.DataFrame(data)

    def cleanup(self):
        """Clean up resources."""
        if self._client_pool is not None:
            self._client_pool.close()
            self._client_pool = None
        elif self._client is not None:
            close_client = getattr(self._client, "close", None)
            if callable(close_client):
                try:
                    close_client()
                except Exception:
                    logger.warning("Error closing OpenAI-compatible client", exc_info=True)
        self._client = None

        if self._server_process is not None:
            logger.info("Stopping vLLM server...")
            self._server_process.terminate()
            self._server_process.wait()
            self._server_process = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def extract_explicit_features(
    texts: List[str],
    specs: List[ExplicitFeatureSpec],
    mode: str = "server",
    server_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,
    max_model_len: Optional[int] = None,
    vllm_reasoning_parser: Optional[str] = "auto",
    vllm_enable_thinking: Optional[bool] = None,
    max_retries: int = 3,
    retry_initial_delay: float = 1.0,
    retry_max_delay: float = 30.0,
    retry_backoff_factor: float = 2.0,
    request_timeout: Optional[float] = 900.0,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_text_length: Optional[int] = None,
    context_strategy: str = "colbert",
    batch_size: int = 32,
    colbert: ColBERTConfig | None = None,
) -> pd.DataFrame:
    """Convenience function to extract features from texts.

    Args:
        texts: List of clinical text strings
        specs: List of feature specifications
        mode: vLLM mode ("server", "start_server", or "python_api")
        server_url: URL for vLLM server
        model_name: Model name/path
        tensor_parallel_size: Number of GPUs
        gpu_memory_utilization: GPU memory fraction
        download_dir: Model download directory
        max_model_len: Maximum model context length (for start_server/python_api)
        vllm_reasoning_parser: vLLM reasoning parser name, "auto", or disabled with None/"none"
        vllm_enable_thinking: Optional chat-template reasoning switch for server requests
        max_retries: Retries per patient before marking as missing
        retry_initial_delay: Initial exponential backoff delay after request failures
        retry_max_delay: Maximum exponential backoff delay after request failures
        retry_backoff_factor: Exponential backoff multiplier
        request_timeout: OpenAI-compatible client request timeout in seconds
        temperature: LLM temperature
        max_tokens: Max response tokens
        max_text_length: Maximum clinical text characters included in prompt
        context_strategy: ``colbert`` (default), ``tail``, or ``contract_lexical_rag``
        batch_size: Batch size for processing

    Returns:
        DataFrame with columns: explicit_feat_{name}, explicit_feat_{name}_missing
    """
    extractor = VLLMFeatureExtractor(
        specs=specs,
        mode=mode,
        server_url=server_url,
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        download_dir=download_dir,
        max_model_len=max_model_len,
        vllm_reasoning_parser=vllm_reasoning_parser,
        vllm_enable_thinking=vllm_enable_thinking,
        max_retries=max_retries,
        retry_initial_delay=retry_initial_delay,
        retry_max_delay=retry_max_delay,
        retry_backoff_factor=retry_backoff_factor,
        request_timeout=request_timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        max_text_length=max_text_length,
        context_strategy=context_strategy,
        colbert=colbert,
    )

    try:
        return extractor.extract_to_dataframe(texts, batch_size=batch_size)
    finally:
        extractor.cleanup()
