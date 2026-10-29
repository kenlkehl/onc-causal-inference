from types import SimpleNamespace

import pytest

import oci.inference.plain_handoff_stage2 as stage2


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
