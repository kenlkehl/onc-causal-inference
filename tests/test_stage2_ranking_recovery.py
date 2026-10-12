"""Constrained clinical ranking, bounded recovery, and exact cache reuse."""
import json

import jsonschema
import pytest

from oci.inference import stage2_clinical_prompts as clinical
from oci.inference import stage2_modifier_count as count
from oci.inference import stage2_modifier_ranking as ranking
from oci.inference.plain_handoff_stage2_analysis import (
    Stage2InfrastructureError,
    Stage2ResponseValidationError,
)
from oci.inference.stage2_prompt_io import fingerprint
from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig
from tests.test_stage2_multi_model import role_fixture


def cards():
    return [{"feature_id": f"f{i}", "definition": {"name": name},
             "modeling_evidence": [{"role": "effect", "evidence_id": f"e{i}"}]}
            for i, name in enumerate(("alk status", "alk status: expression", "alkaline phosphatase"))]


def group(names, reason="Compare the supplied validation evidence."):
    return {"features": names, "rationale": reason}


def schema_names(validate):
    properties = validate.response_format["json_schema"]["schema"]["properties"]
    if "preferred_feature" in properties:
        return [x for x in properties["preferred_feature"]["enum"] if x is not None]
    return properties["ordered_groups"]["items"]["properties"]["features"]["items"]["enum"]


def response(messages, validate, **kwargs):
    names = sorted(schema_names(validate), reverse=True)
    if "preferred_feature" in validate.response_format["json_schema"]["schema"]["properties"]:
        return validate({"preferred_feature": names[0], "rationale": "Compare supplied effect evidence."})
    return validate({"ordered_groups": [group([name]) for name in names]})


def run_review(path, request=response, **overrides):
    definitions, report = role_fixture()
    arguments = dict(definitions=definitions, statistical_report=report,
        role_policy=Stage2RoleAdjudicationConfig(max_candidates_per_request=4),
        maximum=6, maximum_chars=100_000, model_identity="controlled-response", output_dir=path)
    arguments.update(overrides)
    return ranking.rank_modifier_candidates(**arguments, request_json=request)


def test_ranking_schema_requires_arrays_and_exact_names_and_repairs_explain_shape():
    validate = ranking._ranking_validator(cards())
    schema = validate.response_format["json_schema"]["schema"]
    jsonschema.Draft202012Validator.check_schema(schema)
    names = schema_names(validate)
    good = {"ordered_groups": [group([name]) for name in names]}
    jsonschema.validate(good, schema)
    normalized = validate(good)
    assert validate(normalized) == normalized
    for bad in ({"ordered_groups": [group(names[0])]},
                {"ordered_groups": [group(["alk alkaline phosphatase"])]},
                {"ordered_groups": [{"members": names, "rationale": "Evidence."}]}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
        with pytest.raises(ValueError):
            validate(bad)
    with pytest.raises(ValueError, match="ordered group 1.*nonempty array"):
        validate({"ordered_groups": [group(names[0])]})
    assert "array, even for one name" in validate.repair_instructions


def test_ranking_repair_lists_missing_repeated_and_unknown_variables():
    validate = ranking._ranking_validator(cards())
    with pytest.raises(ValueError) as exc:
        validate({"ordered_groups": [group(["alk status", "alk status", "alkaline phosphatase"])]})
    assert 'missing variables: ["alk status: expression"]' in str(exc.value)
    assert 'repeated variables: ["alk status"]' in str(exc.value)
    with pytest.raises(ValueError, match=r'unknown variables: \["foreign"\]'):
        validate({"ranking": [{"feature_id": "foreign", "rationale": "Evidence."}]})


@pytest.mark.parametrize("bad", [None, [], {"ordered_groups": [None]},
    {"ordered_groups": [group([{}])]}, {"ordered_groups": [group(["alk status"], False)]},
    {"ranking": [{"feature_id": []}]}])
def test_malformed_ranking_is_repairable_value_error(bad):
    with pytest.raises(ValueError):
        ranking._ranking_validator(cards())(bad)


def test_pair_schema_accepts_only_supplied_names_or_explicit_tie():
    validate = ranking._pair_validator(cards()[:2])
    schema = validate.response_format["json_schema"]["schema"]
    for name in [*schema_names(validate), None]:
        value = {"preferred_feature": name, "rationale": "Evidence cannot distinguish them."}
        jsonschema.validate(value, schema)
        assert validate(validate(value)) == validate(value)
    for value in ("unknown", False, 1, []):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"preferred_feature": value, "rationale": "Evidence."}, schema)
        with pytest.raises(ValueError):
            validate({"preferred_feature": value, "rationale": "Evidence."})


def test_exhausted_batches_split_preserve_every_candidate_and_resume(tmp_path):
    expected = run_review(tmp_path / "reference")
    def request(messages, validate, **kwargs):
        if messages[0]["content"] == clinical.SYSTEM_PROMPTS["18_rank_modifiers"]:
            raise Stage2ResponseValidationError("invalid groups after bounded repairs")
        return response(messages, validate, **kwargs)
    actual = run_review(tmp_path / "recover", request)
    assert [r["feature_id"] for r in actual["ranking"]] == [r["feature_id"] for r in expected["ranking"]]
    assert actual["locked_feature_ids"] == ["f0"]
    assert {r["strategy"] for r in actual["validation_recoveries"].values()} == {"split_batch", "single_candidate"}
    def no_request(*args, **kwargs):
        pytest.fail("completed recovery must resume without another LLM call")
    assert run_review(tmp_path / "recover", no_request) == actual


def test_split_resume_keeps_finished_children_and_does_not_retry_failed_parent(tmp_path):
    calls, interrupted = [], False
    def request(messages, validate, **kwargs):
        nonlocal interrupted
        if messages[0]["content"] == clinical.SYSTEM_PROMPTS["18_rank_modifiers"]:
            names = tuple(schema_names(validate)); calls.append(names)
            if len(names) > 2:
                raise Stage2ResponseValidationError("missing variable")
            if names == ("f3", "f4") and not interrupted:
                interrupted = True
                raise Stage2InfrastructureError("connection failed")
        return response(messages, validate, **kwargs)
    with pytest.raises(Stage2InfrastructureError):
        run_review(tmp_path, request)
    actual = run_review(tmp_path, request)
    assert [r["feature_id"] for r in actual["ranking"]] == [f"f{i}" for i in range(6, 0, -1)]
    assert calls.count(("f1", "f2", "f3", "f4")) == 1
    assert calls.count(("f1", "f2")) == 1
    assert calls.count(("f3", "f4")) == 2


def test_failed_pair_uses_alternate_format_without_inventing_tie(tmp_path):
    def request(messages, validate, **kwargs):
        if messages[0]["content"] == clinical.SYSTEM_PROMPTS["19_merge_rankings"]:
            raise Stage2ResponseValidationError("invalid preference")
        return response(messages, validate, **kwargs)
    actual = run_review(tmp_path, request)
    assert [r["feature_id"] for r in actual["ranking"]] == [f"f{i}" for i in range(6, 0, -1)]
    assert {r["strategy"] for r in actual["validation_recoveries"].values()} == {"alternate_pair_format"}


def test_both_comparison_formats_failing_is_bounded_and_never_becomes_a_tie(tmp_path):
    definitions, report = role_fixture()
    definitions = definitions[:3]
    report["multi_model_evidence"] = {k: v for k, v in report["multi_model_evidence"].items() if k in {"f0", "f1", "f2"}}
    calls = []
    def request(*args, **kwargs):
        calls.append(1)
        assert len(calls) <= 5, "recovery recursively re-entered its failed pair"
        raise Stage2ResponseValidationError("still invalid")
    with pytest.raises(Stage2ResponseValidationError, match="still invalid"):
        run_review(tmp_path, request, definitions=definitions, statistical_report=report)
    assert len(calls) == 5  # batch, two singletons, pair, alternative pair format
    assert not (tmp_path / "ranking.json").exists()


@pytest.mark.parametrize("error", [Stage2InfrastructureError("offline"), ValueError("bug")])
def test_ranking_recovery_does_not_mask_transport_or_programming_errors(tmp_path, error):
    def request(*args, **kwargs):
        raise error
    with pytest.raises(type(error), match=str(error)):
        run_review(tmp_path, request)
    assert not list(tmp_path.rglob("recovery.json"))


def test_predecessor_requests_reused_across_parent_directories_with_integrity_check(tmp_path):
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    original = run_review(old_root)
    previous = {**original["identity"], "source_sha256": ranking.COMPATIBLE_SOURCE_HASHES[0]}
    protected = {}
    for prompt in list(old_root.glob("requests/*/prompt.json")):
        obj = json.loads(prompt.read_text()); messages = obj["messages"]
        key = ranking._fingerprint({"identity": previous, "messages": messages})
        new_dir = prompt.parent.with_name(key[:20]); prompt.parent.rename(new_dir)
        obj["input_fingerprint"] = fingerprint({"messages": messages, "identity": previous, "version": clinical.PROMPT_VERSION})
        (new_dir / "prompt.json").write_text(json.dumps(obj))
        marker = new_dir / "complete.json"; complete = json.loads(marker.read_text())
        complete["input_fingerprint"] = obj["input_fingerprint"]; marker.write_text(json.dumps(complete))
        for p in new_dir.iterdir():
            protected[p] = p.read_bytes()
    def no_request(*args, **kwargs):
        pytest.fail("compatible saved clinical comparison requested again")
    actual = run_review(new_root, no_request, compatible_output_dirs=[old_root])
    assert actual == original
    assert all(p.read_bytes() == contents for p, contents in protected.items())
    damaged = next(old_root.glob("requests/*/response.json"))
    damaged.write_text(damaged.read_text().replace("Compare", "Changed"))
    with pytest.raises(ValueError, match="hash does not match"):
        run_review(new_root, no_request, compatible_output_dirs=[old_root])


def test_count_directory_compatibility_requires_identical_science_and_other_sources(tmp_path):
    current = {"numerical_input_fingerprint": "training-a", "model": "model-a",
        "sources": {"stage2_modifier_count.py": "new", "stage2_modifier_ranking.py": "new",
                    "dina.py": "unchanged"}}
    previous = {**current, "sources": {**current["sources"],
        "stage2_modifier_count.py": count._RANKING_REPAIR_PREDECESSOR,
        "stage2_modifier_ranking.py": ranking.COMPATIBLE_SOURCE_HASHES[0]}}
    key = ranking._fingerprint(previous); root = tmp_path / key[:20]; root.mkdir()
    path = root / "input.json"; path.write_text(json.dumps({**previous, "input_fingerprint": key}))
    assert count._compatible_ranking_directories(tmp_path, current) == [root]
    assert not count._compatible_ranking_directories(tmp_path, {**current, "model": "model-b"})
    assert not count._compatible_ranking_directories(tmp_path, {**current,
        "sources": {**current["sources"], "dina.py": "changed"}})
    path.write_text(json.dumps({**previous, "input_fingerprint": "wrong"}))
    with pytest.raises(ValueError, match="fingerprint"):
        count._compatible_ranking_directories(tmp_path, current)
