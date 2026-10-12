"""Theme coverage, bounded recovery, and preservation of completed reviews."""
import json

import jsonschema
import pytest

from oci.inference import stage2_clinical_prompts as clinical
from oci.inference import stage2_multi_model_adjudication as adjudication
from oci.inference.plain_handoff_stage2_analysis import (
    Stage2InfrastructureError,
    Stage2ResponseValidationError,
)
from oci.inference.stage2_prompt_io import fingerprint, request_review
from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig
from tests.stage2_prompt_spy import role_response
from tests.test_stage2_multi_model import role_fixture


def theme(members):
    return {"name": "Related clinical measurements", "members": members,
            "interpretation": "Assess the supplied evidence for each distinct measurement.",
            "disagreements": ""}


def cards():
    # Similar nested names reproduced from the failed theme batches.
    names = ["cough", "cough frequency", "cough characteristics: cough frequency"]
    return [{"feature_id": f"f{i}", "definition": {"name": name},
             "modeling_evidence": [{"evidence_id": f"e{i}"}]}
            for i, name in enumerate(names)]


def validator():
    return adjudication._themes_validator(["f0", "f1", "f2"], ["e0", "e1", "e2"],
                                          maximum=3, cards=cards())


def test_theme_repair_names_missing_and_repeated_measurements():
    validate = validator()
    with pytest.raises(ValueError) as error:
        validate({"themes": [theme(["cough", "cough frequency"]), theme(["cough"])]})
    assert 'missing variables: ["cough characteristics: cough frequency"]' in str(error.value)
    assert 'repeated variables: ["cough"]' in str(error.value)
    assert "cough characteristics: cough frequency" in validate.repair_instructions
    response = validate({"themes": [theme(["cough", "cough frequency",
                                            "cough characteristics: cough frequency"])]})
    assert response["themes"][0]["member_feature_ids"] == ["f0", "f1", "f2"]
    assert validate(response) == response


def test_theme_schema_rejects_invented_names_and_validator_rejects_duplicates():
    validate = validator()
    schema = validate.response_format["json_schema"]["schema"]
    jsonschema.Draft202012Validator.check_schema(schema)
    good = {"themes": [theme(["cough", "cough frequency",
                              "cough characteristics: cough frequency"])]}
    jsonschema.validate(good, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"themes": [theme(["cough type"])]}, schema)
    with pytest.raises(ValueError, match="Required variable names"):
        validate({"themes": [theme(["cough type"])]})
    with pytest.raises(ValueError, match="repeated variables"):
        validate({"themes": [theme(["cough", "cough", "cough frequency",
                                     "cough characteristics: cough frequency"])]})


@pytest.mark.parametrize("members", [None, "cough", {}, [None], [["cough"]]])
def test_malformed_theme_members_are_repairable_validation_errors(members):
    with pytest.raises(ValueError):
        validator()({"themes": [theme(members)]})


def run_review(path, request):
    definitions, report = role_fixture()
    return adjudication.adjudicate_multi_model_roles(
        definitions=definitions, statistical_report=report, request_json=request,
        output_dir=path, policy=Stage2RoleAdjudicationConfig(max_candidates_per_request=4),
        workers=2,
    )


def successful_response(messages, validate, **kwargs):
    slug = next(k for k, v in clinical.SYSTEM_PROMPTS.items() if v == messages[0]["content"])
    if slug == "16_merge_themes":
        return validate({"merges": []})
    if slug == "15_model_themes":
        names = validate.response_format["json_schema"]["schema"]["properties"]["themes"]["items"]["properties"]["members"]["items"]["enum"]
        return validate({"themes": [theme(names)]})
    return validate(role_response(confounder=True))


def test_exhausted_batches_split_singletons_stay_neutral_and_merges_preserve_all(tmp_path):
    calls = []

    def request(messages, validate, **kwargs):
        slug = next(k for k, v in clinical.SYSTEM_PROMPTS.items() if v == messages[0]["content"])
        calls.append(slug)
        if slug in {"15_model_themes", "16_merge_themes"}:
            raise Stage2ResponseValidationError("invalid membership after bounded repairs")
        return successful_response(messages, validate, **kwargs)

    result = run_review(tmp_path, request)
    _, report, _ = result
    members = [key for t in report["themes"] for key in t["member_feature_ids"]]
    assert sorted(members) == [f"f{i}" for i in range(7)]
    assert len(report["decisions"]) == 7
    assert calls.count("17_model_roles") == 7
    assert calls.count("15_model_themes") == 12  # Two finite binary split trees.
    strategies = {row["strategy"] for row in report["theme_recoveries"].values()}
    assert strategies == {"split_batch", "neutral_singleton", "retain_unmerged_themes"}
    assert all("unavailable" in t["interpretation"] for t in report["themes"])

    def no_request(*args, **kwargs):
        pytest.fail("completed recovered reviews must resume without more LLM calls")

    assert run_review(tmp_path, no_request) == result


def test_interrupted_split_resumes_children_without_retrying_exhausted_parent(tmp_path):
    calls = []
    interrupted = False

    def request(messages, validate, **kwargs):
        nonlocal interrupted
        if messages[0]["content"] == clinical.SYSTEM_PROMPTS["15_model_themes"]:
            ids = tuple(validate.response_format["json_schema"]["schema"]["properties"]["themes"]["items"]["properties"]["members"]["items"]["enum"])
            calls.append(ids)
            if len(ids) > 2:
                raise Stage2ResponseValidationError("missing variables")
            if ids == ("f2", "f3") and not interrupted:
                interrupted = True
                raise Stage2InfrastructureError("connection interrupted")
        return successful_response(messages, validate, **kwargs)

    with pytest.raises(Stage2InfrastructureError):
        run_review(tmp_path, request)
    result = run_review(tmp_path, request)
    assert len(result[1]["decisions"]) == 7
    assert calls.count(("f0", "f1", "f2", "f3")) == 1
    assert calls.count(("f0", "f1")) == 1
    assert calls.count(("f2", "f3")) == 2


@pytest.mark.parametrize("error", [Stage2InfrastructureError("offline"), ValueError("bug")])
def test_theme_recovery_does_not_mask_infrastructure_or_programming_errors(tmp_path, error):
    def request(*args, **kwargs):
        raise error
    with pytest.raises(type(error), match=str(error)):
        run_review(tmp_path, request)
    assert not list(tmp_path.rglob("recovery.json"))


def test_predecessor_checkpoints_are_revalidated_and_reused_without_rewriting(tmp_path):
    result = run_review(tmp_path, successful_response)
    identity = json.loads((tmp_path / "complete.json").read_text())["identity"]
    previous = {**identity, "source_sha256": adjudication._COMPATIBLE_SOURCE_HASHES[0]}
    protected = {}
    for prompt in tmp_path.rglob("prompt.json"):
        payload = json.loads(prompt.read_text())
        key = fingerprint({"messages": payload["messages"], "identity": previous,
                           "version": clinical.PROMPT_VERSION})
        payload["input_fingerprint"] = key
        prompt.write_text(json.dumps(payload))
        complete = prompt.with_name("complete.json")
        value = json.loads(complete.read_text())
        value["input_fingerprint"] = key
        complete.write_text(json.dumps(value))
        for p in (prompt, complete, prompt.with_name("response.json")):
            protected[p] = p.read_bytes()

    def no_request(*args, **kwargs):
        pytest.fail("valid predecessor checkpoint should be reused")

    assert run_review(tmp_path, no_request) == result
    assert all(p.read_bytes() == value for p, value in protected.items())
    damaged = tmp_path / "roles/batch_000/response.json"
    damaged.write_text(damaged.read_text().replace("Exploratory support.", "Changed."))
    with pytest.raises(ValueError, match="hash does not match"):
        run_review(tmp_path, no_request)


def test_checkpoint_compatibility_does_not_accept_other_inputs_or_bypass_validator(tmp_path):
    messages = [{"role": "user", "content": "clinical input"}]
    old = {"source": "previous", "evidence": "a"}
    new = {"source": "current", "evidence": "a"}
    calls = []
    def request(*args, **kwargs):
        calls.append(1)
        return {"value": len(calls)}
    request_review(tmp_path, messages, lambda x: x, request_json=request, identity=old)
    assert request_review(tmp_path, messages, lambda x: x, request_json=request,
                          identity=new, compatible_identities=[old]) == {"value": 1}
    def invalid(response):
        raise ValueError("new coverage check")
    with pytest.raises(ValueError, match="new coverage check"):
        request_review(tmp_path, messages, invalid, request_json=request, identity=new,
                       compatible_identities=[old])
    request_review(tmp_path, messages, lambda x: x, request_json=request,
                   identity={**new, "evidence": "b"},
                   compatible_identities=[{**old, "evidence": "b"}])
    assert len(calls) == 2
