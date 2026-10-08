import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from oci.config import ExperimentConfig, ExplicitFeatureSpec
from oci.extraction import (
    ExplicitFeatureValue,
    VLLMFeatureExtractor,
    build_extraction_prompt,
    infer_vllm_reasoning_parser,
    parse_extraction_response,
    parse_server_urls,
    resolve_vllm_reasoning_parser,
)
from oci.extraction.explicit_features import (
    _parse_extraction_response_with_issues,
    build_extraction_repair_prompt,
)
from oci.models.explicit_feature_featurizer import (
    ExplicitFeatureFeaturizer,
    filter_specs_by_role,
    get_raw_explicit_features,
)


def test_clinical_text_examples_keep_selected_long_note_complete():
    from oci.inference.agentic_explicit_feature_forest import (
        _clinical_text_examples,
    )

    note = "complete clinical note " + ("x" * 5_000)
    examples = _clinical_text_examples(
        pd.DataFrame({"clinical_text": [note]}),
        "clinical_text",
        n_examples=1,
        max_chars=1_600,
    )

    assert examples == [note]


def test_evidence_digest_keeps_complete_clinical_examples():
    from oci.inference.multi_model_agentic_forest import (
        _build_evidence_digest_agent_context,
        _evidence_digest_role_context,
    )

    note = "complete note " + ("z" * 8_000)
    context = _build_evidence_digest_agent_context(
        outer_fold=1,
        feature_discovery_methods=["bow"],
        max_proposals=4,
        clinical_question="Identify baseline variables.",
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        current_features=[],
        metrics={},
        importance={},
        clinical_text_examples=[note],
    )

    assert context["clinical_text_examples"] == [note]
    assert _evidence_digest_role_context(
        context,
        role="confounder",
        max_proposals=2,
    )["clinical_text_examples"] == [note]


def test_complete_paged_provider_processes_and_reconciles_every_page(tmp_path: Path):
    from oci.extraction import COMPLETE_PAGED_RESPONSE_SCHEMA, CompletePageResponse
    from oci.inference.agentic_explicit_feature_forest import (
        VLLMExplicitFeatureExtractionProvider,
    )

    note = ("baseline text " * 12) + "ECOG 2" + (" later text" * 12)
    spec = ExplicitFeatureSpec(
        name="performance_status",
        type="categorical",
        categories=["ECOG 0", "ECOG 1", "ECOG 2"],
        description="Baseline ECOG score",
        roles=["confounder"],
    )
    provider = object.__new__(VLLMExplicitFeatureExtractionProvider)
    provider.output_dir = tmp_path
    provider.config = SimpleNamespace(text_column="clinical_text")
    provider.feature_config = SimpleNamespace(
        complete_page_core_chars=80,
        complete_page_context_chars=10,
        complete_page_max_chars=100,
        complete_reconciliation_fan_in=3,
        extraction_batch_size=4,
    )

    class FakeExtractor:
        def __init__(self):
            self.pages = []

        def extract_complete_page(self, *, text, page, feature, geometry):
            self.pages.append(page)
            marker_start = text.find("ECOG 2", page.context_start, page.context_end)
            if marker_start >= 0:
                return CompletePageResponse.validate(
                    {
                        "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
                        "status": "positive",
                        "normalized_value": "ECOG 2",
                        "reason": None,
                        "citations": [
                            {
                                "start": marker_start,
                                "end": marker_start + len("ECOG 2"),
                                "text": "ECOG 2",
                            }
                        ],
                    },
                    text=text,
                    page=page,
                )
            return CompletePageResponse.validate(
                {
                    "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
                    "status": "negative",
                    "normalized_value": None,
                    "reason": None,
                    "citations": [],
                },
                text=text,
                page=page,
            )

        def reconcile_complete_pages(self, *, text, feature, children):
            positive = next(
                (
                    child["response"]
                    for child in children
                    if child["response"]["status"] == "positive"
                ),
                None,
            )
            response = positive or {
                "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
                "status": "negative",
                "normalized_value": None,
                "reason": None,
                "citations": [],
            }
            return {
                "child_ids": [child["node_id"] for child in children],
                **response,
            }

    extractor = FakeExtractor()
    frame = provider._extract_complete_paged_spec(
        dataset=pd.DataFrame({"clinical_text": [note]}),
        specs=[spec],
        extractor=extractor,
    )

    cores = sorted(extractor.pages, key=lambda page: page.core_start)
    assert [(page.core_start, page.core_end) for page in cores] == [
        (start, min(len(note), start + 80)) for start in range(0, len(note), 80)
    ]
    assert frame.loc[0, "explicit_feat_performance_status"] == "ECOG 2"
    assert not bool(frame.loc[0, "explicit_feat_performance_status_missing"])
    assert list((tmp_path / "complete_paged_ledgers").glob("*.json"))


def test_explicit_feature_roles_are_valid_and_deduped():
    spec = ExplicitFeatureSpec(
        name="ecog_status",
        type="categorical",
        categories=["0", "1", "2"],
        roles=["confounder", "effect_modifier", "confounder"],
    )

    assert spec.roles == ["confounder", "effect_modifier"]

    with pytest.raises(ValueError, match="roles required"):
        ExplicitFeatureSpec(name="age", type="continuous")

    with pytest.raises(ValueError, match="invalid roles"):
        ExplicitFeatureSpec(name="age", type="continuous", roles=["instrument"])


def test_raw_explicit_features_split_by_role_with_overlap():
    specs = [
        ExplicitFeatureSpec(
            name="ecog_status",
            type="categorical",
            categories=["0", "1", "2"],
            roles=["confounder", "effect_modifier"],
        ),
        ExplicitFeatureSpec(
            name="age",
            type="continuous",
            roles=["confounder"],
        ),
        ExplicitFeatureSpec(
            name="marker",
            type="continuous",
            roles=["effect_modifier"],
        ),
    ]
    values = [
        {
            "ecog_status": "1",
            "ecog_status_missing": False,
            "age": 60.0,
            "age_missing": False,
            "marker": 2.0,
            "marker_missing": False,
        },
        {
            "ecog_status": "2",
            "ecog_status_missing": False,
            "age": 70.0,
            "age_missing": False,
            "marker": 4.0,
            "marker_missing": False,
        },
    ]

    confounder_specs = filter_specs_by_role(specs, "confounder")
    effect_specs = filter_specs_by_role(specs, "effect_modifier")
    assert [s.name for s in confounder_specs] == ["ecog_status", "age"]
    assert [s.name for s in effect_specs] == ["ecog_status", "marker"]

    w_features, w_names = get_raw_explicit_features(values, specs, role="confounder")
    x_features, x_names = get_raw_explicit_features(values, specs, role="effect_modifier")

    assert w_names == [
        "ecog_status_1",
        "ecog_status_2",
        "ecog_status_missing",
        "age_normalized",
        "age_missing",
    ]
    assert x_names == [
        "ecog_status_1",
        "ecog_status_2",
        "ecog_status_missing",
        "marker_normalized",
        "marker_missing",
    ]
    assert len(w_features[0]) == 5
    assert len(x_features[0]) == 5
    assert w_features[0][0] == 1.0
    assert x_features[0][0] == 1.0


def test_raw_explicit_features_populates_provided_normalization_dicts():
    specs = [
        ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
    ]
    values = [
        {"age": 60.0, "age_missing": False},
        {"age": 70.0, "age_missing": False},
    ]
    means = {}
    stds = {}

    get_raw_explicit_features(values, specs, continuous_means=means, continuous_stds=stds)

    assert means["age"] == 65.0
    assert stds["age"] == 5.0


def test_vllm_reasoning_parser_inference_and_resolution():
    assert infer_vllm_reasoning_parser("nvidia/Qwen3.6-35B-A3B-NVFP4") == "qwen3"
    assert infer_vllm_reasoning_parser("nvidia/Gemma-4-31B-IT-NVFP4") == "gemma4"
    assert infer_vllm_reasoning_parser("openai/gpt-oss-120b") == "openai_gptoss"
    assert infer_vllm_reasoning_parser("meta-llama/Llama-3.1-8B-Instruct") is None

    assert (
        resolve_vllm_reasoning_parser(
            "auto",
            "nvidia/Qwen3.6-35B-A3B-NVFP4",
        )
        == "qwen3"
    )
    assert (
        resolve_vllm_reasoning_parser(
            "deepseek_r1",
            "unknown/model",
        )
        == "deepseek_r1"
    )
    assert resolve_vllm_reasoning_parser("none", "nvidia/Gemma-4-31B-IT-NVFP4") is None


def test_parse_extraction_response_strips_inline_reasoning_trace():
    specs = [
        ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
    ]

    parsed = parse_extraction_response(
        '<think>{"age": "not the answer"}</think>\n{"age": 71}',
        specs,
    )

    assert parsed["age"].value == 71.0
    assert parsed["age"].is_missing is False


@pytest.mark.parametrize("mode", ["server", "python_api"])
@pytest.mark.parametrize("maximum", [10, 4])
def test_standalone_extraction_batches_variables_and_preserves_patient_values(monkeypatch, mode, maximum):
    specs = [ExplicitFeatureSpec(name=f"f_{i}", type="continuous", roles=["confounder"]) for i in range(23)]
    extractor = VLLMFeatureExtractor(specs, mode=mode, max_variables_per_extraction_request=maximum)
    monkeypatch.setattr(extractor, "_ensure_initialized", lambda: None)
    calls = []

    def group(text, group_specs):
        calls.append((text, [s.name for s in group_specs]))
        return {s.name: ExplicitFeatureValue(s.name, s.type, int(text) + int(s.name[2:]), False) for s in group_specs}

    monkeypatch.setattr(extractor, "_extract_single_server_group", group)
    monkeypatch.setattr(extractor, "_extract_batch_python_api_group", lambda texts, specs: [group(t, specs) for t in texts])
    results = extractor.extract(["100", "200"], batch_size=2, show_progress=False)
    assert len(calls) == 2 * ((23 + maximum - 1) // maximum)
    assert all(len(names) <= maximum for _, names in calls)
    assert [len(names) for text, names in calls if text == "100"] == [maximum] * (23 // maximum) + [23 % maximum]
    for text, result in zip([100, 200], results):
        assert list(result) == [s.name for s in specs]
        assert [result[s.name].value for s in specs] == [text + i for i in range(23)]


def test_build_extraction_prompt_truncates_to_note_tail():
    specs = [
        ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
    ]

    prompt = build_extraction_prompt(
        "beginning age 44. " + ("middle " * 20) + "end age 71.",
        specs,
        max_text_length=30,
        context_strategy="tail",
    )

    assert "end age 71" in prompt
    assert "beginning age 44" not in prompt


def test_extraction_prompt_distinguishes_unknown_from_not_documented():
    specs = [
        ExplicitFeatureSpec(
            name="marker_status",
            type="categorical",
            categories=["negative", "positive", "unknown", "not_documented"],
            roles=["confounder"],
        ),
    ]

    prompt = build_extraction_prompt("Marker testing was indeterminate.", specs)

    assert 'Use "unknown" only when the note explicitly says' in prompt
    assert 'Use "not_documented" when the note does not state' in prompt
    assert "Every categorical field may be JSON null when its value is not documented" in prompt
    assert "For continuous fields" in prompt


def test_neural_query_rag_prompt_uses_retrieved_excerpt_instruction_without_policy():
    specs = [
        ExplicitFeatureSpec(
            name="prior_platinum_response",
            type="categorical",
            categories=["response", "no_response", "not_documented"],
            roles=["confounder"],
        )
    ]
    document = (
        "[neural_query_rag_v1]\n"
        "<retrieved_excerpt>Prior platinum produced a partial response.</retrieved_excerpt>"
    )

    prompt = build_extraction_prompt(document, specs)

    assert "Read every retrieved excerpt" in prompt
    assert "prior therapies, responses, or outcomes" not in prompt
    assert "current treatment decision" not in prompt
    assert "complete clinical note" not in prompt


def test_extraction_prompts_contain_no_temporal_policy_text():
    specs = [
        ExplicitFeatureSpec(
            name="response_status",
            type="categorical",
            categories=["absent", "present", "unknown", "not_documented"],
            roles=["effect_modifier"],
        )
    ]
    prompt = build_extraction_prompt(
        "After therapy, response status was present.",
        specs,
        source_text_temporally_valid_by_design=True,
    )
    repair = build_extraction_repair_prompt(
        ["malformed JSON"],
        specs,
        source_text_temporally_valid_by_design=True,
    )

    for text in (prompt, repair):
        assert "temporally valid by design" not in text
        assert "treatment-time boundary" not in text
        assert "source_text_temporal_policy" not in text
        assert "current treatment" not in text


def test_default_extraction_prompt_also_contains_no_temporal_policy_text():
    specs = [ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"])]
    prompt = build_extraction_prompt("Age 70.", specs)
    repair = build_extraction_repair_prompt(["malformed JSON"], specs)

    assert "treatment" not in prompt.casefold()
    assert "treatment" not in repair.casefold()
    assert "temporally" not in prompt.casefold()
    assert "temporally" not in repair.casefold()


def test_parse_extraction_response_accepts_quoted_null_as_missing():
    specs = [
        ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ExplicitFeatureSpec(
            name="marker_status",
            type="categorical",
            categories=["negative", "positive", "unknown", "not_documented"],
            roles=["confounder"],
        ),
    ]

    parsed = parse_extraction_response('{"age": "null", "marker_status": "null"}', specs)

    assert parsed["age"].value is None and parsed["age"].is_missing
    assert parsed["marker_status"].value is None and parsed["marker_status"].is_missing


def test_parse_extraction_response_maps_categorical_value_aliases():
    specs = [
        ExplicitFeatureSpec(
            name="pd_l1_expression",
            type="categorical",
            categories=["<1%", "1-49%", ">=50%"],
            value_aliases={">=50%": ["high", "50% or greater"]},
            roles=["effect_modifier"],
        ),
    ]

    parsed = parse_extraction_response('{"pd_l1_expression": "high"}', specs)

    assert parsed["pd_l1_expression"].value == ">=50%"
    assert parsed["pd_l1_expression"].is_missing is False


@pytest.mark.parametrize(
    "sentinel",
    [
        "unknown",
        " UNKNOWN ",
        "not_documented",
        "not documented",
        "not-documented",
    ],
)
def test_undeclared_categorical_missing_sentinel_is_null_without_schema_issue(
    sentinel,
):
    specs = [
        ExplicitFeatureSpec(
            name="surface_state",
            type="categorical",
            categories=["matte", "gloss"],
            roles=["confounder"],
        )
    ]

    parsed = _parse_extraction_response_with_issues(
        json.dumps({"surface_state": sentinel}),
        specs,
    )

    assert parsed.issues == []
    assert parsed.values["surface_state"].value is None
    assert parsed.values["surface_state"].is_missing is True


def test_declared_categorical_missing_labels_and_aliases_win_before_null_fallback():
    specs = [
        ExplicitFeatureSpec(
            name="declared_state",
            type="categorical",
            categories=["matte", "unknown"],
            roles=["confounder"],
        ),
        ExplicitFeatureSpec(
            name="aliased_state",
            type="categorical",
            categories=["matte", "gloss"],
            value_aliases={"gloss": ["not documented"]},
            roles=["confounder"],
        ),
    ]

    parsed = _parse_extraction_response_with_issues(
        '{"declared_state": "unknown", "aliased_state": "not_documented"}',
        specs,
    )

    assert parsed.issues == []
    assert parsed.values["declared_state"].value == "unknown"
    assert parsed.values["declared_state"].is_missing is False
    assert parsed.values["aliased_state"].value == "gloss"
    assert parsed.values["aliased_state"].is_missing is False


def test_nonexact_undeclared_missing_label_remains_a_schema_issue():
    spec = ExplicitFeatureSpec(
        name="surface_state",
        type="categorical",
        categories=["matte", "gloss"],
        roles=["confounder"],
    )

    parsed = _parse_extraction_response_with_issues(
        '{"surface_state": "unknown_value"}',
        [spec],
    )

    assert len(parsed.issues) == 1
    assert "invalid category" in parsed.issues[0]


def test_raw_explicit_features_maps_categorical_value_aliases():
    specs = [
        ExplicitFeatureSpec(
            name="pd_l1_expression",
            type="categorical",
            categories=["<1%", "1-49%", ">=50%"],
            value_aliases={">=50%": ["high"]},
            roles=["effect_modifier"],
        ),
    ]
    features, names = get_raw_explicit_features(
        [
            {
                "pd_l1_expression": "high",
                "pd_l1_expression_missing": False,
            }
        ],
        specs,
        role="effect_modifier",
    )

    assert names == [
        "pd_l1_expression_1-49%",
        "pd_l1_expression_>=50%",
        "pd_l1_expression_missing",
    ]
    assert features == [[0.0, 1.0, 0.0]]


def test_vllm_feature_extractor_server_client_uses_request_timeout(monkeypatch):
    calls = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
        request_timeout=123.0,
    )

    extractor._init_server_client()

    assert calls["timeout"] == 123.0
    assert calls["max_retries"] == 0


def test_vllm_feature_extractor_cleanup_closes_server_client_pool(monkeypatch):
    closed = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def close(self):
            closed.append(self.kwargs["base_url"])

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
        server_url="http://server/v1",
    )
    extractor._init_server_client()

    extractor.cleanup()

    assert closed == ["http://server/v1"]
    assert extractor._client is None
    assert extractor._client_pool is None


def test_vllm_feature_extractor_request_does_not_set_timeout():
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.update(kwargs)

            class Message:
                content = '{"age": 41}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class Chat:
            completions = FakeCompletions()

        chat = Chat()

    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
        max_retries=1,
    )
    extractor._client = FakeClient()

    result = extractor._extract_single_server("Age: 41")

    assert "timeout" not in calls
    assert "response_format" not in calls
    assert result["age"].value == 41.0
    assert result["age"].is_missing is False


def test_vllm_feature_extractor_can_disable_chat_template_thinking():
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.update(kwargs)

            class Message:
                content = '{"age": 41}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class Chat:
            completions = FakeCompletions()

        chat = Chat()

    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
        vllm_enable_thinking=False,
        max_retries=1,
    )
    extractor._client = FakeClient()

    result = extractor._extract_single_server("Age: 41")

    assert calls["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert result["age"].value == 41.0


def test_vllm_feature_extractor_requests_json_response_format_for_google_agent_platform():
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.update(kwargs)

            class Message:
                content = '{"age": 41}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class Chat:
            completions = FakeCompletions()

        chat = Chat()

    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
        server_url=(
            "https://aiplatform.googleapis.com/v1/projects/p/" "locations/global/endpoints/openapi"
        ),
        model_name="google/gemma-4-26b-a4b-it-maas",
        api_key="GOOGLE_ADC",
        max_retries=1,
    )
    extractor._client = FakeClient()

    result = extractor._extract_single_server("Age: 41")

    assert calls["response_format"] == {"type": "json_object"}
    assert result["age"].value == 41.0
    assert result["age"].is_missing is False


def test_vllm_feature_extractor_repairs_malformed_extraction_json():
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)

            class Message:
                content = "age: 41"

            if len(calls) == 2:
                Message.content = '{"age": 41}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class Chat:
            completions = FakeCompletions()

        chat = Chat()

    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
        max_retries=2,
    )
    extractor._client = FakeClient()

    result = extractor._extract_single_server("Age: 41")

    assert result["age"].value == 41.0
    assert result["age"].is_missing is False
    assert len(calls) == 2
    repair_messages = calls[1]["messages"]
    assert repair_messages[1]["role"] == "assistant"
    assert repair_messages[1]["content"] == "age: 41"
    assert repair_messages[2]["role"] == "user"
    assert "could not be used" in repair_messages[2]["content"]
    assert "malformed JSON" in repair_messages[2]["content"]
    assert '"age": <number-or-null>' in repair_messages[2]["content"]


def test_vllm_feature_extractor_accepts_schema_complete_null_without_retry():
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)

            class Message:
                content = '{"age": null}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class Chat:
            completions = FakeCompletions()

        chat = Chat()

    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
        max_retries=3,
    )
    extractor._client = FakeClient()

    result = extractor._extract_single_server("Age is not documented")

    assert len(calls) == 1
    assert result["age"].value is None
    assert result["age"].is_missing is True


def test_vllm_feature_extractor_accepts_undeclared_missing_sentinel_without_retry():
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)

            class Message:
                content = '{"surface_state": "not_documented"}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class Chat:
            completions = FakeCompletions()

        chat = Chat()

    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(
                name="surface_state",
                type="categorical",
                categories=["matte", "gloss"],
                roles=["confounder"],
            ),
        ],
        mode="server",
        max_retries=3,
    )
    extractor._client = FakeClient()

    result = extractor._extract_single_server("No surface state is documented.")

    assert len(calls) == 1
    assert result["surface_state"].value is None
    assert result["surface_state"].is_missing is True


def test_vllm_feature_extractor_retries_next_server(monkeypatch):
    calls = []

    class FakeCompletions:
        def __init__(self, base_url):
            self.base_url = base_url

        def create(self, **kwargs):
            calls.append((self.base_url, kwargs["model"]))
            if self.base_url == "http://server-a/v1":
                raise TimeoutError("server overloaded")

            class Message:
                content = '{"age": 41}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type(
                "Chat",
                (),
                {"completions": FakeCompletions(kwargs["base_url"])},
            )()

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
        server_url="http://server-a/v1,http://server-b/v1",
        model_name="heterogeneous-pool",
        model_names_by_url={
            "http://server-a/v1": "served-model-a",
            "http://server-b/v1": "served-model-b",
        },
        max_retries=2,
        retry_initial_delay=0.0,
    )
    extractor._init_server_client()
    extractor._client_pool._next_index = 0

    result = extractor._extract_single_server("Age: 41")

    assert parse_server_urls("http://server-a/v1, http://server-b/v1") == [
        "http://server-a/v1",
        "http://server-b/v1",
    ]
    assert calls == [
        ("http://server-a/v1", "served-model-a"),
        ("http://server-b/v1", "served-model-b"),
    ]
    assert result["age"].value == 41.0
    assert result["age"].is_missing is False


def test_vllm_feature_extractor_server_uses_batch_size_for_concurrency(monkeypatch):
    calls = {"submitted": []}

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers):
            calls["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, text):
            calls["submitted"].append(text)
            return FakeFuture(fn(text))

    monkeypatch.setattr("oci.extraction.explicit_features.ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        "oci.extraction.explicit_features.as_completed",
        lambda futures: list(futures)[::-1],
    )

    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
    )
    monkeypatch.setattr(extractor, "_ensure_initialized", lambda: None)

    def fake_extract(text):
        return {
            "age": ExplicitFeatureValue(
                name="age",
                type="continuous",
                value=float(text),
                is_missing=False,
            )
        }

    monkeypatch.setattr(extractor, "_extract_single_server", fake_extract)

    results = extractor.extract(["1", "2", "3"], batch_size=2, show_progress=False)

    assert calls["max_workers"] == 2
    assert calls["submitted"] == ["1", "2", "3"]
    assert [row["age"].value for row in results] == [1.0, 2.0, 3.0]


def test_experiment_config_rejects_old_explicit_confounder_keys():
    with pytest.raises(ValueError, match="explicit_confounders"):
        ExperimentConfig.from_dict(
            {
                "applied_inference": {
                    "dataset_path": "dataset.parquet",
                    "explicit_confounders": {"enabled": True, "confounders": []},
                }
            }
        )

    with pytest.raises(ValueError, match="confounder_forest"):
        ExperimentConfig.from_dict(
            {
                "applied_inference": {
                    "dataset_path": "dataset.parquet",
                    "architecture": {"model_type": "confounder_forest"},
                }
            }
        )


def test_explicit_confounder_imports_are_role_aware_feature_aliases():
    from oci.extraction.explicit_confounders import (
        ExplicitConfounderValue,
        VLLMConfounderExtractor,
    )
    from oci.models.explicit_confounder_featurizer import (
        ExplicitConfounderFeaturizer,
        get_raw_confounder_features,
    )

    assert ExplicitConfounderValue is ExplicitFeatureValue
    assert VLLMConfounderExtractor is VLLMFeatureExtractor
    assert ExplicitConfounderFeaturizer is ExplicitFeatureFeaturizer
    assert get_raw_confounder_features is get_raw_explicit_features


def test_legacy_htr_and_training_keys_migrate_without_reactivating_old_models():
    config = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": "dataset.parquet",
                "architecture": {
                    "model_type": "multi_model_forest",
                    "causal_head_hidden_outcome_dim": 73,
                },
                "training": {
                    "learning_rate": 0.002,
                    "gamma_rlearner": 99,
                    "beta_targreg": 88,
                    "stop_grad_propensity": True,
                },
            }
        }
    )

    assert config.applied_inference.architecture.htr_prediction_head_hidden_dim == 73
    assert config.applied_inference.training.learning_rate == pytest.approx(0.002)
    assert not hasattr(config.applied_inference.training, "gamma_rlearner")
