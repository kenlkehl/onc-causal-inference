from dataclasses import replace
from types import SimpleNamespace

import pytest

import oci.inference.plain_handoff_stage2 as stage2
from oci.inference.stage2_sampling import SAMPLING_FIELDS


def config(model="gemma4-31b", **kwargs):
    return stage2.PlainHandoffStage2Config(
        endpoint="http://test/v1", model=model, **kwargs
    )


@pytest.mark.parametrize(
    "model,effort,expected",
    [
        ("nvidia/Gemma-4-31B-IT-NVFP4", "high", (1.0, 0.95, 64, 0.0, 1.0)),
        ("google/gemma-4-E4B-it", "none", (1.0, 0.95, 64, 0.0, 1.0)),
        ("Qwen/Qwen3-8B", "high", (0.6, 0.95, 20, 0.0, 1.0)),
        ("Qwen/Qwen3-32B", "high", (0.6, 0.95, 20, 0.0, 1.0)),
        ("Qwen/Qwen3-32B", "none", (0.7, 0.8, 20, 0.0, 1.0)),
        ("Qwen/Qwen3.5-27B", "high", (1.0, 0.95, 20, 1.5, 1.0)),
        ("Qwen/Qwen3.6-27B", "high", (1.0, 0.95, 20, 0.0, 1.0)),
        ("Qwen/Qwen3.8-27B", "none", (0.7, 0.8, 20, 1.5, 1.0)),
        ("LiquidAI/LFM2.5-2.6B", "none", (0.1, 1.0, 50, 0.0, 1.1)),
        ("LiquidAI/LFM2.5-1.2B-Instruct", "none", (0.1, 1.0, 50, 0.0, 1.05)),
    ],
)
def test_publisher_profiles(model, effort, expected):
    policy = stage2._stage2_request_policy(
        config(model, interpretation_reasoning_effort=effort)
    )
    assert (
        tuple(
            policy[key]
            for key in (
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
                "repetition_penalty",
            )
        )
        == expected
    )


def test_unknown_family_defers_to_server_and_explicit_zero_survives_roundtrip():
    assert (
        not set(SAMPLING_FIELDS)
        & stage2._stage2_request_policy(config("unknown")).keys()
    )
    original = config(
        temperature=0.0,
        top_p=0.8,
        top_k=10,
        presence_penalty=0.5,
        min_p=0.1,
        frequency_penalty=0.2,
        repetition_penalty=1.02,
    )
    restored = stage2.plain_stage2_config_from_mapping(
        original.public_dict(), default_workers=1
    )
    policy = stage2._stage2_request_policy(restored)
    assert {k: policy[k] for k in SAMPLING_FIELDS} == {
        k: getattr(original, k) for k in SAMPLING_FIELDS
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("temperature", True),
        ("temperature", float("nan")),
        ("top_p", 0),
        ("top_k", 1.5),
        ("top_k", -2),
        ("min_p", 2),
        ("presence_penalty", 3),
        ("frequency_penalty", float("inf")),
        ("repetition_penalty", 0),
    ],
)
def test_invalid_overrides_rejected_from_mapping(key, value):
    with pytest.raises(ValueError, match=key):
        stage2.plain_stage2_config_from_mapping(
            {"endpoint": "http://test/v1", key: value},
            default_workers=1,
        )


def test_gemma_sampling_on_wire(monkeypatch):
    calls = []
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"ok":true}'),
                finish_reason="stop",
            )
        ]
    )

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            return response

        def close(self):
            pass

    monkeypatch.setattr("openai.OpenAI", Client)
    stage2._openai_completion([{"role": "user", "content": "Test"}], config())
    assert calls[0]["temperature"] == 1.0
    assert calls[0]["top_p"] == 0.95
    assert calls[0]["presence_penalty"] == 0.0
    assert calls[0]["extra_body"] == {
        "top_k": 64,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_extractor_and_repair_resolve_their_own_profiles():
    primary = config()
    extractor = replace(
        primary, model="Qwen/Qwen3.8-27B", runtime_request_kind="extraction"
    )
    before = stage2._stage2_request_policy(extractor)
    after = stage2._stage2_request_policy(
        replace(extractor, runtime_reasoning_effort="high")
    )
    assert (before["temperature"], before["presence_penalty"]) == (0.7, 1.5)
    assert (after["temperature"], after["presence_penalty"]) == (1.0, 0.0)
    assert stage2._stage2_request_policy(primary)["top_k"] == 64


def test_alias_uses_backing_model_version_and_sampling_affects_fingerprint():
    aliased = config(
        "my-server",
        runtime_model_family="qwen3",
        runtime_sampling_model="Qwen/Qwen3.8-27B",
    )
    assert stage2._stage2_request_policy(aliased)["temperature"] == 1.0

    def fingerprint(c):
        return stage2._feature_definition_input_value(
            config=c,
            clinical_question="Test",
            outer_fold=1,
            discovery_packets=[],
            seed=42,
        )

    assert fingerprint(aliased) != fingerprint(replace(aliased, temperature=0.0))
    assert "runtime_sampling_model" not in aliased.public_dict()


def test_rejected_top_k_does_not_exhaust_budget_or_drop_other_sampling(monkeypatch):
    calls = []

    class Unsupported(ValueError):
        status_code = 400

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            if "top_k" in kwargs.get("extra_body", {}):
                raise Unsupported("unsupported parameter: top_k")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"ok":true}'),
                        finish_reason="stop",
                    )
                ]
            )

        def close(self):
            pass

    monkeypatch.setattr("openai.OpenAI", Client)
    stage2._openai_completion(
        [{"role": "user", "content": "Test"}], config(transport_max_attempts=2)
    )
    assert len(calls) == 2
    assert calls[1]["temperature"] == 1.0
    assert calls[1]["extra_body"]["repetition_penalty"] == 1.0
    assert "top_k" not in calls[1]["extra_body"]


def test_live_alias_resolution_routes_primary_and_extraction_profiles(monkeypatch):
    class Models(list):
        def __init__(self, alias, root):
            super().__init__([alias])
            self.records = [{"id": alias, "root": root}]

    def models(c):
        return Models(
            c.model,
            "Qwen/Qwen3.8-27B" if c.model == "primary" else "google/gemma-4-E4B-it",
        )

    monkeypatch.setattr(stage2, "_served_model_ids", models)
    runner = stage2.PlainHandoffStage2(
        config=config(
            "primary",
            extraction_llm=stage2.Stage2ExtractionLLMConfig(
                endpoint="http://extractor/v1",
                model="extractor",
            ),
        ),
        clinical_question="Test",
        extraction_tokenizer=object(),
    )
    assert stage2._stage2_request_policy(runner.config)["temperature"] == 1.0
    assert stage2._stage2_request_policy(runner.config)["top_k"] == 20
    policy = stage2._stage2_request_policy(
        runner.extraction_request_config, "extraction"
    )
    assert policy["top_k"] == 64
    assert policy["temperature"] == 1.0


@pytest.mark.parametrize("request_kind", ["interpretation", "extraction"])
def test_flash_next_auto_uses_xhigh_and_publisher_sampling_for_both_roles(request_kind):
    cfg = config("local-alias", runtime_model_family="qwen3",
                 runtime_sampling_model="Inferact/Qwen3.8-Flash-Next-NVFP4")
    policy = stage2._stage2_request_policy(cfg, request_kind)
    assert policy["reasoning_effort"] == "xhigh"
    assert {k: policy[k] for k in SAMPLING_FIELDS} == {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
    }
    assert policy["sampling_provenance"]["profile"] == "qwen3.8_flash_next"
    variants = stage2._openai_request_variants(base_kwargs={}, request_policy=policy, model_family="qwen3")
    for variant in variants:
        assert variant["reasoning_effort"] == "xhigh"
        assert variant["extra_body"]["top_k"] == 20
        assert variant["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}


def test_flash_next_rejected_controls_fail_without_silent_downgrade(monkeypatch):
    calls = []

    class Unsupported(ValueError):
        status_code = 400

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            raise Unsupported("unsupported parameter: reasoning_effort")

        def close(self):
            pass

    monkeypatch.setattr("openai.OpenAI", Client)
    with pytest.raises(stage2._Stage2TransportFailure):
        stage2._openai_completion([{"role": "user", "content": "Test"}],
                                 config("Inferact/Qwen3.8-Flash-Next-NVFP4"))
    assert len(calls) == 1
    assert calls[0]["reasoning_effort"] == "xhigh"
    assert calls[0]["extra_body"]["top_k"] == 20


def test_live_flash_next_backing_model_controls_both_roles_and_manifest(monkeypatch, tmp_path):
    class Models(list):
        records = [{"id": "clinical", "root": "Inferact/Qwen3.8-Flash-Next-NVFP4"}]
    monkeypatch.setattr(stage2, "_served_model_ids", lambda cfg: Models(["clinical"]))
    runner = stage2.PlainHandoffStage2(config=config("clinical", extraction_llm=stage2.Stage2ExtractionLLMConfig(
        endpoint="http://test/v1", model="clinical")), clinical_question="Treatment effect", extraction_tokenizer=object())
    runner._check_and_record_model_identity(tmp_path)
    import json
    manifest = json.loads((tmp_path / "model_identity.json").read_text())
    assert all(p["reasoning_effort"] == "xhigh" for p in manifest["effective_request_policies"].values())
    explicit = replace(runner.config, extraction_reasoning_effort="none")
    policy = stage2._stage2_request_policy(explicit, "extraction")
    assert (policy["reasoning_effort"], policy["temperature"], policy["top_p"], policy["presence_penalty"]) == ("none", 0.7, 0.8, 1.5)
