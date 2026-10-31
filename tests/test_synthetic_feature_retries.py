"""Reject invalid LLM feature designs and repair them with precise feedback."""

from unittest.mock import Mock

import pytest

from synthetic_data import generator as g


def _feature(name, roles):
    return {"name": name, "type": "continuous", "roles": roles}


def _request(client, **kwargs):
    return g._generate_confounders(
        client, "Compare treatment A with treatment B.",
        num_confounders=5, num_effect_modifiers=5, **kwargs,
    )


def test_exhaustion_is_bounded_and_reports_the_latest_error():
    client = Mock()
    client.generate_json.side_effect = [
        {"features": [_feature("first_bad", ["unknown_role"])]},
        {"features": [_feature("second_bad", [])]},
        {"features": [_feature("last_bad", ["confounder"])]},
    ]

    with pytest.raises(ValueError, match="after 3 attempts") as raised:
        _request(client)

    assert client.generate_json.call_count == 3
    assert "got {'total': 1, 'confounder': 1, 'effect_modifier': 0}" in str(raised.value)
    assert isinstance(raised.value.__cause__, ValueError)
    final_prompt = client.generate_json.call_args.kwargs["prompt"]
    assert "second_bad" in final_prompt
    assert "missing roles" in final_prompt
    assert "first_bad" not in final_prompt
