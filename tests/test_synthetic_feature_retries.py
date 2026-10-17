"""Reject invalid LLM feature designs and repair them with precise feedback."""

import copy
import json
from unittest.mock import Mock

import pytest

from synthetic_data import generator as g


def _feature(name, roles):
    return {"name": name, "type": "continuous", "roles": roles}


def _disjoint_features():
    return [
        _feature(f"confounder_{i}", ["confounder"]) for i in range(5)
    ] + [
        _feature(f"modifier_{i}", ["effect_modifier"]) for i in range(5)
    ]


def _request(client, **kwargs):
    return g._generate_confounders(
        client, "Compare treatment A with treatment B.",
        num_confounders=5, num_effect_modifiers=5, **kwargs,
    )


def test_s5_bad_role_counts_are_repaired_with_the_response_and_error(caplog):
    valid = _disjoint_features()
    invalid = copy.deepcopy(valid)
    for feature in invalid[-2:]:
        feature["roles"].append("confounder")
    rejected = {"features": invalid}
    client = Mock()
    client.generate_json.side_effect = [rejected, {"features": valid}]

    assert _request(client) == valid
    assert client.generate_json.call_count == 2
    first, second = [call.kwargs["prompt"] for call in client.generate_json.call_args_list]
    assert "exactly 10 total features" in first
    assert "exactly 5 confounder-only features" in first
    assert "exactly 5 effect_modifier-only features" in first
    assert "do not tag any feature with both roles" in first
    assert second.startswith(first)
    assert json.dumps(rejected, ensure_ascii=False) in second
    assert "Requested total=10, confounders=5, effect_modifiers=5" in second
    assert "got {'total': 10, 'confounder': 7, 'effect_modifier': 5}" in second
    assert "Keep the original requested counts" in second
    assert "attempt 1/3" in caplog.text
    assert g._count_features_by_role(invalid)["confounder"] == 7


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


@pytest.mark.parametrize("response, error", [
    ([], "must be a JSON object"),
    ({}, "non-empty 'features' array"),
    ({"features": None}, "must be a JSON array"),
    ({"features": {}}, "must be a JSON array"),
    ({"features": [None]}, "Invalid feature structure"),
    ({"features": [{"name": "age"}]}, "Invalid feature structure"),
    ({"features": [_feature("", ["confounder"])]}, "non-empty string"),
    ({"features": [_feature("age", ["confounder"])] * 2}, "Duplicate feature name"),
    ({"features": [{"name": "age", "type": "integer", "roles": ["confounder"]}]},
     "type must be 'continuous' or 'categorical'"),
    ({"features": [{"name": "ecog", "type": "categorical", "roles": ["confounder"]}]},
     "missing categories"),
    ({"features": [_feature("age", "confounder")]}, "roles must be a JSON array"),
    ({"features": [_feature("age", 1)]}, "roles must be a JSON array"),
    ({"features": [_feature("age", [])]}, "missing roles"),
    ({"features": [_feature("age", ["instrument"])]}, "invalid roles"),
])
def test_invalid_feature_structure_is_repaired(response, error):
    client = Mock()
    client.generate_json.side_effect = [response, {"features": _disjoint_features()}]

    assert _request(client) == _disjoint_features()
    assert client.generate_json.call_count == 2
    assert error in client.generate_json.call_args.kwargs["prompt"]


@pytest.mark.parametrize("key", ["features", "confounders"])
def test_valid_response_needs_only_one_call_including_legacy_key(key):
    client = Mock()
    client.generate_json.return_value = {key: _disjoint_features()}

    assert _request(client) == _disjoint_features()
    assert client.generate_json.call_count == 1


def test_requested_overlap_is_preserved():
    features = [
        _feature("c1", ["confounder"]),
        _feature("c2", ["confounder"]),
        _feature("c3", ["confounder"]),
        _feature("m1", ["effect_modifier"]),
        _feature("m2", ["effect_modifier"]),
        _feature("both", ["confounder", "effect_modifier"]),
    ]
    client = Mock()
    client.generate_json.return_value = {"features": features}

    actual = g._generate_confounders(
        client, "Question", num_features=6, num_confounders=4, num_effect_modifiers=3,
    )

    assert actual == features
    assert client.generate_json.call_count == 1
    prompt = client.generate_json.call_args.kwargs["prompt"]
    assert "exactly 3 confounder-only features" in prompt
    assert "exactly 2 effect_modifier-only features" in prompt
    assert "exactly 1 features tagged with both roles" in prompt
    assert "groups must be disjoint" not in prompt


def test_existing_ordered_subset_recovery_does_not_need_a_retry():
    features = _disjoint_features()
    client = Mock()
    client.generate_json.return_value = {"features": [
        _feature("extra_dual", ["confounder", "effect_modifier"]), *features,
    ]}

    assert _request(client) == features
    assert client.generate_json.call_count == 1


@pytest.mark.parametrize("error", [RuntimeError("server unavailable"), ValueError("invalid JSON")])
def test_client_failures_are_not_retried_by_the_feature_validation_loop(error):
    client = Mock()
    client.generate_json.side_effect = error

    with pytest.raises(type(error), match=str(error)) as raised:
        _request(client)

    assert raised.value is error
    assert client.generate_json.call_count == 1


def test_nonpositive_attempt_budget_fails_before_calling_the_llm():
    client = Mock()
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        _request(client, max_attempts=0)
    client.generate_json.assert_not_called()
