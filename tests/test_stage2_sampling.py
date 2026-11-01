from types import SimpleNamespace

import pytest

import oci.inference.plain_handoff_stage2 as stage2
from oci.inference.stage2_sampling import is_qwen_3_6_35b_a3b, recommended_sampling


def config(model="gemma4-31b", **kwargs):
    return stage2.PlainHandoffStage2Config(
        endpoint="http://test/v1", model=model, **kwargs
    )


@pytest.mark.parametrize(
    "key,value",
    [("temperature", float("nan"))],
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


@pytest.mark.parametrize(
    "model,thinking,temperature,top_p,presence_penalty",
    [
        ("Qwen/Qwen3.6-35B-A3B", True, 1.0, 0.95, 1.5),
        ("Qwen/Qwen3.6-35B-A3B-FP8", True, 1.0, 0.95, 1.5),
        ("Qwen/Qwen3.6-35B-A3B-FP8", False, 0.7, 0.8, 1.5),
        ("Qwen/Qwen3.6-27B", True, 1.0, 0.95, 0.0),
        ("Qwen/Qwen3.6-27B", False, 0.7, 0.8, 1.5),
        ("Qwen/Qwen3.5-27B", True, 1.0, 0.95, 1.5),
        ("Qwen/Qwen3.8-27B", True, 1.0, 0.95, 0.0),
        ("Qwen/Qwen3-32B", True, 0.6, 0.95, 0.0),
    ],
)
def test_qwen_sampling_profiles(model, thinking, temperature, top_p, presence_penalty):
    sampling = recommended_sampling(model, "qwen3", thinking)
    assert sampling["temperature"] == temperature
    assert sampling["top_p"] == top_p
    assert sampling["top_k"] == 20
    assert sampling["min_p"] == 0.0
    assert sampling["presence_penalty"] == presence_penalty
    assert sampling["repetition_penalty"] == 1.0


def test_qwen_3_6_35b_a3b_request_policy():
    policy = stage2._stage2_request_policy(config(model="Qwen/Qwen3.6-35B-A3B-FP8"), "interpretation")
    assert policy["reasoning_effort"] == "high"
    assert (policy["temperature"], policy["presence_penalty"]) == (1.0, 1.5)
    assert policy["sampling_provenance"] == {
        "profile": "qwen3.6_35b_a3b",
        "verified_on": "2026-10-31",
        "source": "https://huggingface.co/Qwen/Qwen3.6-35B-A3B#best-practices",
    }
    extraction = stage2._stage2_request_policy(config(model="Qwen/Qwen3.6-35B-A3B-FP8"), "extraction")
    assert extraction["reasoning_effort"] == "none"
    assert (extraction["temperature"], extraction["presence_penalty"]) == (0.7, 1.5)
    assert stage2._stage2_request_policy(
        config(model="Qwen/Qwen3.6-27B"), "interpretation"
    )["presence_penalty"] == 0.0


@pytest.mark.parametrize(
    "model,expected",
    [
        ("Qwen/Qwen3.6-35B-A3B", True),
        ("Qwen/Qwen3.6-35B-A3B-FP8", True),
        ("qwen3.6_35b_a3b", True),
        ("Qwen/Qwen3.6-27B", False),
        ("Qwen/Qwen3.5-35B-A3B", False),
        ("Qwen/Qwen3.6-35B-A30B", False),
    ],
)
def test_is_qwen_3_6_35b_a3b(model, expected):
    assert is_qwen_3_6_35b_a3b(model) is expected
