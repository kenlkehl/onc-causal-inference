import copy
from dataclasses import replace
import json
import math

import pandas as pd
import pytest

from oci.colbert_config import ColBERTConfig
from oci.extraction import colbert
from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_decision as decision
from oci.inference.stage2_decision_client import VLLMDecisionClient, decision_messages, probabilities_from_response
from oci.inference.stage2_decision_config import DecisionExtractionConfig, classifier_vllm_config, decision_config_from_mapping
from oci.inference.stage2_decision_ontology import prepare_ontologies, triggered_patterns
from oci.inference.stage2_endpoint_pool import ExtractionEndpoint


def feature(name="weight", *, categorical=False):
    return {"feature_id": name, "name": name, "description": name,
            "value_type": "categorical" if categorical else "continuous",
            "categories_or_unit": ["red", "blue"] if categorical else ["kg"],
            "measurement_definition": f"Documented pretreatment {name}.",
            "missing_value_rule": "Null if not documented.",
            **({} if categorical else {"decision_ontology": {"minimum": 0, "maximum": 216}})}


class FakeClient:
    def __init__(self, answers=(), *, policy=None, choose=None):
        self.policy = policy or DecisionExtractionConfig(enabled=True)
        self.answers = iter(answers)
        self.choose = choose
        self.calls = []

    def encode(self, messages):
        return list(range(len(json.dumps(messages)) // 4))

    def decide(self, evidence, criterion, options):
        answer = self.choose(evidence, criterion, options) if self.choose else next(self.answers)
        assert answer in dict(options)
        messages = decision_messages(evidence, criterion, options)
        result = {"selected": answer, "probabilities": {k: .9 if k == answer else .1/(len(options)-1) for k, _ in options},
                  "options": options, "messages": messages, "prompt_tokens": len(self.encode(messages))}
        assert result["prompt_tokens"] <= self.policy.max_prompt_tokens
        self.calls.append(result)
        return result


@pytest.fixture
def retrieval(monkeypatch):
    calls = []

    class Retriever:
        def retrieve(self, text, features, *, top_k):
            assert len(features) == 1
            calls.append((text, features[0]["name"]))
            return evidence(text)

    monkeypatch.setattr(colbert, "get_retriever", lambda config: Retriever())
    monkeypatch.setattr(colbert, "retrieval_identity", lambda config: {"test": "colbert", "top_k": config.top_k})
    return calls


def evidence(text):
    return {"hits": [[{"start": 0, "end": len(text), "chunk_index": 0, "rank": 1, "score": 1}]],
            "source_sha256": "test", "context": text}


@pytest.mark.parametrize("estimate_bounds,bin_answers,expected", [
    ((0, 216), ["bin_3", "bin_1", "bin_1"], 72.5),
    ((-216, 0), ["bin_4", "bin_6", "bin_6"], -72.5),
    ((-108, 108), ["bin_4", "bin_1", "bin_1"], .5),
])
def test_numeric_narrowing_and_separate_verification(estimate_bounds, bin_answers, expected):
    f = feature()
    f["decision_ontology"] = dict(zip(("minimum", "maximum"), estimate_bounds))
    client = FakeClient([*bin_answers, "true"])
    result = decision.measure_feature(feature=f, source="Documented measurement", evidence=evidence("Documented measurement"), client=client)
    assert result["value"] == expected
    assert len(result["calls"]) == 4
    assert all(len(c["options"]) == 8 for c in result["calls"][:3])
    assert len(result["calls"][-1]["options"]) == 2
    left, right = result["verification_documented_range"]
    assert left <= right
    for actual in (left, right):
        assert math.isclose(abs(expected-actual), .05*abs(actual))


def test_verification_rejection_and_missing_do_not_become_values():
    text = "Weight was measured."
    for answers, status in ((["bin_3"]*3+["false"], "numeric_verification_failed"),
                            ([decision.MISSING], "not_documented"),
                            ([decision.NONE], "none_of_above")):
        result = decision.measure_feature(feature=feature(), source=text, evidence=evidence(text), client=FakeClient(answers))
        assert result["value"] is None and result["status"] == status
    no_text = decision.measure_feature(feature=feature(), source="", evidence=evidence(""), client=FakeClient())
    assert no_text["status"] == "not_documented" and no_text["calls"] == []


def test_bin_edges_partition_domain_without_overlap_and_preserve_excluded_upper_bound():
    edges, options = decision.numeric_options(-12, 12, 6, upper_inclusive=True)
    assert edges == [-12, -8, -4, 0, 4, 8, 12]
    assert options[2][1].endswith("-4 <= value < 0")
    assert options[5][1].endswith("8 <= value <= 12")
    _, refined = decision.numeric_options(8, 12, 6, upper_inclusive=False)
    assert refined[5][1].endswith("value < 12")


def test_budget_keeps_whole_ranked_chunks_and_accounts_for_chat_template():
    text = "a "*4000
    hits = [[{"start": i, "end": i+1000, "chunk_index": i//1000} for i in range(0, len(text), 1000)]]
    client = FakeClient(["yes"], policy=replace(DecisionExtractionConfig(), max_prompt_tokens=800))
    result = decision.packed_decision(client, source=text, evidence={"hits": hits}, criterion="criterion",
                                      options=[("yes", "yes"), ("no", "no")])
    assert result["prompt_tokens"] <= 800
    assert result["retrieval_budget"]["omitted_chunk_indices"]
    context = json.loads(result["messages"][1]["content"])["evidence"]
    selected = len(result["retrieval_budget"]["selected_chunk_indices"])
    assert text[:1000*selected] in context
    with pytest.raises(ValueError, match="ontology alone"):
        decision.packed_decision(client, source=text, evidence={"hits": hits}, criterion="x"*10000,
                                 options=[("yes", "yes"), ("no", "no")])


@pytest.mark.parametrize("budget", [50, 300, 800, 2000])
def test_batched_packing_preserves_first_failure_even_when_later_prefix_shrinks(budget):
    text = "é漢 " * 400
    hits = [[{"start": left, "end": right, "chunk_index": i}
             for i, (left, right) in enumerate([(0, 400), (800, 1200), (400, 800)])]]

    class PrefixClient(FakeClient):
        def encode(self, messages):
            context = json.loads(messages[1]["content"])["evidence"]
            # The third chunk bridges two source spans, removing their headers.
            count = 1000 if context.count("[Source characters") == 2 else 500 if context else 100
            return list(range(count))

    class BatchClient(PrefixClient):
        def encode_batch(self, conversations):
            return [self.encode(messages) for messages in conversations]

        def decide(self, evidence, criterion, options, *, prompt_token_ids):
            result = super().decide(evidence, criterion, options)
            assert len(prompt_token_ids) == result["prompt_tokens"]
            return result

    kwargs = dict(source=text, evidence={"hits": hits}, criterion="criterion", options=[("yes", "yes"), ("no", "no")])
    policy = replace(DecisionExtractionConfig(), max_prompt_tokens=budget)
    if budget in (50, 300):
        for client in (PrefixClient(["yes"], policy=policy), BatchClient(["yes"], policy=policy)):
            with pytest.raises(ValueError, match="ontology alone" if budget == 50 else "No complete"):
                decision.packed_decision(client, **kwargs)
    else:
        expected = decision.packed_decision(PrefixClient(["yes"], policy=policy), **kwargs)
        actual = decision.packed_decision(BatchClient(["yes"], policy=policy), **kwargs)
        assert actual == expected
        if budget == 800:
            assert actual["retrieval_budget"]["selected_chunk_indices"] == [0]


def test_batched_packing_sends_identical_token_ids_without_final_retokenization():
    class Tokenizer:
        def __init__(self):
            self.batch_calls = self.single_calls = 0

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return json.dumps(messages, ensure_ascii=False) + "ASSISTANT"

        def encode(self, text, **kwargs):
            self.single_calls += 1
            return list(text.encode())

        def __call__(self, texts, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            self.batch_calls += 1
            return {"input_ids": [list(text.encode()) for text in texts]}

    tokenizer = Tokenizer()
    sent = []

    def transport(endpoint, payload, timeout):
        sent.append(payload["input"])
        return {"model": "plumb", "data": [{"num_classes": 16, "probs": [8, 0] + [-1]*14}],
                "usage": {"prompt_tokens": len(payload["input"]), "completion_tokens": 0}}

    client = VLLMDecisionClient(policy=DecisionExtractionConfig(enabled=True), model="plumb",
        endpoints=(ExtractionEndpoint("http://localhost:8134/v1", 1),), tokenizer=tokenizer, transport=transport)
    client.pool._metrics_reader = None
    result = decision.packed_decision(client, source="red patient", evidence=evidence("red patient"),
        criterion="Documented color", options=[("red", "Red"), ("blue", "Blue")])
    expected = list(tokenizer.apply_chat_template(result["messages"], tokenize=False,
        add_generation_prompt=True, enable_thinking=False).encode())
    assert sent == [expected] and result["selected"] == "red"
    assert tokenizer.batch_calls == 1 and tokenizer.single_calls == 0


def test_classifier_uses_only_offered_logits_and_checks_protocol():
    response = {"data": [{"num_classes": 16, "probs": [2.07, 0] + [1000]*14}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 0}}
    probs = probabilities_from_response(response, [("yes", "yes"), ("no", "no")], 2.07, prompt_tokens=123)
    assert probs["yes"] == pytest.approx(math.e/(1+math.e))
    for logits, message in (([.0625]*16, "probabilities"), ([float("nan")]*16, "nonfinite"), ([1, 2], "16-class")):
        bad = copy.deepcopy(response)
        bad["data"][0]["probs"] = logits
        with pytest.raises(ValueError, match=message):
            probabilities_from_response(bad, [("yes", "yes"), ("no", "no")], 2.07, prompt_tokens=123)
    response["usage"]["prompt_tokens"] = 122
    with pytest.raises(ValueError, match="altered the prompt"):
        probabilities_from_response(response, [("yes", "yes"), ("no", "no")], 2.07, prompt_tokens=123)


def extract_args(tmp_path, client, definitions, *, row_ids=(0, 1)):
    return dict(dataset=pd.DataFrame({"text": ["red", "blue", "HELDOUT_ONLY"], "outcome": [0, 1, 1]}),
                row_ids=list(row_ids), text_column="text", definitions=definitions, output_dir=tmp_path / "extraction",
                request_json=lambda *args, **kwargs: pytest.fail("LLM patient extraction attempted"),
                workers=1, max_prompt_chars=10000, feature_batch_size=10, context_strategy="colbert",
                colbert=ColBERTConfig(devices=("cpu",)), decision_extraction=client.policy, decision_client=client)


def test_extraction_dispatch_resume_and_definition_source_invalidation(tmp_path, retrieval):
    client = FakeClient(choose=lambda text, criterion, options: "category_0")
    args = extract_args(tmp_path, client, [feature("color", categorical=True), feature("shade", categorical=True)])
    first = analysis.extract_rows(**args)
    assert first["_oci_row_id"].tolist() == [0, 1]
    assert len(client.calls) == 4
    analysis.extract_rows(**args)
    assert len(client.calls) == 4
    args["dataset"].loc[1, "text"] = "changed patient"
    analysis.extract_rows(**args)
    assert len(client.calls) == 6
    args["definitions"][0]["categories_or_unit"] = ["green", "blue"]
    analysis.extract_rows(**args)
    assert len(client.calls) == 8
    assert all("HELDOUT_ONLY" not in text for text, name in retrieval)
    assert all("outcome" not in str(call["messages"]) for call in client.calls)
    with pytest.raises(ValueError, match="policy changed"):
        analysis.extract_rows(**{**args, "decision_extraction": replace(client.policy, numeric_passes=2),
                                 "decision_client": FakeClient(policy=replace(client.policy, numeric_passes=2))})


def test_training_revision_reextracts_changed_feature_and_heldout_is_frozen(tmp_path, retrieval):
    def choose(text, criterion, options):
        return decision.NONE if '"name":"color"' in criterion and not any("green" in v for _, v in options) else "category_0"

    client = FakeClient(choose=choose, policy=replace(DecisionExtractionConfig(enabled=True), none_above_min_patients=2))
    args = extract_args(tmp_path, client, [feature("color", categorical=True), feature("stable", categorical=True)])
    revisions = []

    def reviewer(messages, validate, **kwargs):
        revisions.append(messages)
        payload = json.loads(messages[1]["content"])
        assert payload["failure_summary"]["patient_fraction"] == 1
        assert "patient_row_ids" not in payload["failure_summary"]
        assert "HELDOUT_ONLY" not in str(messages) and "outcome\":" not in str(messages)
        return validate({"action": "revise", "rationale": "Missing documented green category", "categories_or_unit": ["red", "blue", "green"]})

    args["request_json"] = reviewer
    frame, definitions, rounds = analysis._extract_training_with_ontology_feedback(
        **args, feedback_dir=tmp_path / "feedback", minimum_failure_patients=3, max_refinement_rounds=2)
    assert rounds == 1 and len(revisions) == 1
    assert len(client.calls) == 6  # Four initial measurements, only color's two rows repeated.
    assert frame["color"].tolist() == ["red", "red"]
    assert definitions[0]["categories_or_unit"] == ["red", "blue", "green"]
    heldout = {**args, "definitions": definitions, "row_ids": [2], "output_dir": tmp_path / "heldout"}
    analysis.extract_rows(**heldout)
    assert len(revisions) == 1
    assert retrieval[-1][0] == "HELDOUT_ONLY"


def test_numeric_bounds_prepared_without_patient_data_then_reused(tmp_path):
    f = feature()
    del f["decision_ontology"]
    calls = []

    def reviewer(messages, validate, **kwargs):
        payload = json.loads(messages[1]["content"])
        assert set(payload) == {"feature"}
        calls.append(messages)
        return validate({"value_type": "continuous", "categories_or_unit": ["kg"],
                         "decision_ontology": {"minimum": 0, "maximum": 250}, "rationale": "Plausible domain"})

    a = prepare_ontologies([f], output_dir=tmp_path, request_json=reviewer)
    b = prepare_ontologies([f], output_dir=tmp_path, request_json=reviewer)
    assert a == b and len(calls) == 1
    changed = {**a[0], "decision_ontology": {"minimum": 0, "maximum": 500}}
    assert analysis._feature_extraction_fingerprint(a[0]) != analysis._feature_extraction_fingerprint(changed)


def test_parallel_ontology_preparation_preserves_order_and_checkpoints(tmp_path):
    import threading
    from oci.inference.stage2_decision_ontology import preparation_complete

    definitions = [feature(f"weight_{i}") for i in range(8)]
    for f in definitions:
        del f["decision_ontology"]
    original = copy.deepcopy(definitions)
    barrier = threading.Barrier(4)

    def reviewer(messages, validate, **kwargs):
        barrier.wait(timeout=5)  # Four independent feature requests must overlap.
        assert kwargs["request_kind"] == "interpretation"
        assert set(json.loads(messages[1]["content"])) == {"feature"}
        return validate({"value_type": "continuous", "categories_or_unit": ["kg"],
                         "decision_ontology": {"minimum": 0, "maximum": 250}, "rationale": "Plausible domain"})

    prepared = prepare_ontologies(definitions, output_dir=tmp_path, request_json=reviewer, workers=4)
    assert [f["name"] for f in prepared] == [f["name"] for f in definitions]
    assert definitions == original
    assert preparation_complete(definitions, output_dir=tmp_path)
    assert len(list((tmp_path / "preparation").glob("*/response.json"))) == 8
    cached = prepare_ontologies(definitions, output_dir=tmp_path, workers=4,
        request_json=lambda *a, **kw: pytest.fail("completed preparation was repeated"))
    assert cached == prepared
    definitions[0]["measurement_definition"] = "A different measurement contract"
    assert not preparation_complete(definitions, output_dir=tmp_path)


@pytest.mark.parametrize("count,target,call_count", [(14, 13, 1), (15, 14, 2), (33, 32, 2), (197, 14, 3)])
def test_large_categorical_ontology_preserves_values_with_bounded_choice_groups(count, target, call_count):
    f = feature(categorical=True)
    f["categories_or_unit"] = [f"value_{i}" for i in range(count)]
    original = copy.deepcopy(f)

    def choose(evidence, criterion, options):
        assert 2 <= len(options) <= 16
        for key, description in options:
            if key == f"category_{target}":
                return key
            if key.startswith("group_") and f'"value_{target}"' in description:
                return key
        pytest.fail("target category missing from offered choices")

    client = FakeClient(choose=choose)
    result = decision.measure_feature(feature=f, source="Documented measurement",
                                      evidence=evidence("Documented measurement"), client=client)
    assert result["value"] == f"value_{target}" and result["status"] == "accepted"
    assert len(result["calls"]) == call_count
    assert f == original
    if count > 14:
        initial_groups = [json.loads(description.split(": ", 1)[1])
                          for key, description in result["calls"][0]["options"] if key.startswith("group_")]
        assert [value for group in initial_groups for value in group] == f["categories_or_unit"]


@pytest.mark.parametrize("answers,status", [
    ([decision.MISSING], "not_documented"),
    ([decision.NONE], "none_of_above"),
    (["group_0", decision.NONE], "category_selection_failed"),
])
def test_grouped_category_exits_preserve_missingness_and_do_not_expand_wrong_branches(answers, status):
    f = feature(categorical=True)
    f["categories_or_unit"] = [f"value_{i}" for i in range(15)]
    result = decision.measure_feature(feature=f, source="Documented measurement",
                                      evidence=evidence("Documented measurement"), client=FakeClient(answers))
    assert result["value"] is None and result["status"] == status
    assert result["outside_initial_domain"] == (status == "none_of_above")


@pytest.mark.parametrize("unit,measurement", [
    ("unitless", "A dimensionless symptom score."),
    ("mg", "The explicitly documented medication dose in mg."),
])
def test_missing_numeric_unit_prepared_from_contract_without_mutating_frozen_definition(tmp_path, unit, measurement):
    f = feature()
    f["categories_or_unit"] = []
    f["measurement_definition"] = measurement
    del f["decision_ontology"]
    original = copy.deepcopy(f)

    def reviewer(messages, validate, **kwargs):
        payload = json.loads(messages[1]["content"])
        assert set(payload) == {"feature"}
        assert payload["feature"]["categories_or_unit"] == []
        return validate({"value_type": "continuous", "categories_or_unit": [unit],
                         "decision_ontology": {"minimum": 0, "maximum": 100}, "rationale": measurement})

    prepared = prepare_ontologies([f], output_dir=tmp_path, request_json=reviewer)
    assert prepared[0]["categories_or_unit"] == [unit]
    assert prepared[0]["decision_ontology"] == {"minimum": 0, "maximum": 100}
    assert f == original


def test_threshold_excludes_missing_and_verification_failure_and_small_counts():
    policy = DecisionExtractionConfig(enabled=True)
    base = {"feature_name": "weight", "failure_kind": "none_of_above", "patient_count": 3, "patient_fraction": .2}
    assert triggered_patterns({"feature_failure_patterns": [base]}, policy)
    for changes in ({"failure_kind": "not_documented"}, {"failure_kind": "numeric_verification_failed"},
                    {"failure_kind": "category_selection_failed"},
                    {"patient_count": 2}, {"patient_fraction": .19}):
        assert not triggered_patterns({"feature_failure_patterns": [{**base, **changes}]}, policy)


def test_pipeline_config_and_managed_readout():
    from oci.inference.plain_handoff_stage2 import plain_stage2_config_from_mapping
    from oci.inference.vllm_server_pool import ManagedVLLMConfig

    raw = {"endpoint": "http://localhost:8000/v1", "model": "primary", "decision_extraction": {"enabled": True},
           "extraction_llm": {"endpoint": "http://localhost:8134/v1", "model": "plumb-4b"}}
    config = plain_stage2_config_from_mapping(raw, default_workers=2)
    assert config.decision_extraction.enabled and config.public_dict()["decision_extraction"]["numeric_bins"] == 6
    assert analysis._configured_extraction_feature_batch_size(config) == 1
    for changes in ({"max_prompt_tokens": 3001}, {"numeric_bins": 7}, {"temperature": 0}, {"enabled": "true"}):
        with pytest.raises(ValueError):
            decision_config_from_mapping(changes)
    managed = classifier_vllm_config(ManagedVLLMConfig(server_count=1, gpus=("1",)))
    args = list(managed.extra_args)
    override = json.loads(args[args.index("--hf-overrides")+1])
    assert override["classifier_from_token"] == list("ABCDEFGHIJKLMNOP")
    assert override["method"] == "no_post_processing"
    assert json.loads(args[args.index("--pooler-config")+1]) == {"pooling_type": "LAST", "use_activation": False}
    with pytest.raises(ValueError, match="manages"):
        classifier_vllm_config(managed)


def test_client_sends_pretokenized_classify_request_and_releases_capacity_on_error():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return "rendered"

        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return [1, 2, 3]

    requests = []

    def transport(endpoint, payload, timeout):
        requests.append(payload)
        assert payload == {"model": "plumb", "input": [1, 2, 3], "use_activation": False, "add_special_tokens": False}
        return {"model": "plumb", "data": [{"num_classes": 16, "probs": [8, 0]+[-1]*14}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 0}}

    client = VLLMDecisionClient(policy=DecisionExtractionConfig(enabled=True), model="plumb",
        endpoints=(ExtractionEndpoint("http://localhost:8134/v1", 1),), tokenizer=Tokenizer(), transport=transport)
    client.pool._metrics_reader = None
    assert client.decide("Evidence", "Criterion", [("true", "True"), ("false", "False")])["selected"] == "true"

    def bad(*args):
        raise ValueError("Malformed classifier response")

    client._transport = bad
    with pytest.raises(ValueError, match="Malformed"):
        client.decide("Evidence", "Criterion", [("true", "True"), ("false", "False")])
    assert client.pool.snapshot()[0]["in_flight"] == 0
    assert client.capacity.acquire(blocking=False)
    client.capacity.release()


def test_classifier_dispatch_uses_all_replicas_despite_slow_initial_timings():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "rendered"

        def encode(self, text, **kwargs):
            return [1, 2, 3]

    endpoints = tuple(ExtractionEndpoint(f"http://replica-{i}.test/v1", 128) for i in range(7))
    called = []

    def transport(endpoint, payload, timeout):
        called.append(endpoint)
        return {"model": "plumb", "data": [{"num_classes": 16, "probs": [8, 0]+[-1]*14}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 0}}

    client = VLLMDecisionClient(policy=DecisionExtractionConfig(enabled=True), model="plumb",
        endpoints=endpoints, api_key="test-secret", workers=128, tokenizer=Tokenizer(), transport=transport)
    client.pool._metrics_reader = None
    assert client.pool._api_key == "test-secret"
    for i, state in enumerate(client.pool._states):
        state.latency_seconds = .01 if i in (0, 2, 5) else 10
    for _ in range(70):
        assert client.decide("Evidence", "Criterion", [("true", "True"), ("false", "False")])["selected"] == "true"
    assert [called.count(e.endpoint) for e in endpoints] == [10] * 7


def test_explicit_ontologies_are_immutable_and_missing_never_triggers_review(tmp_path, retrieval):
    client = FakeClient(choose=lambda *args: decision.NONE,
                        policy=replace(DecisionExtractionConfig(enabled=True), none_above_min_patients=1))
    protected = {**feature("color", categorical=True), "configured_explicit_feature": True}
    args = extract_args(tmp_path, client, [protected])
    frame, definitions, rounds = analysis._extract_training_with_ontology_feedback(
        **args, feedback_dir=tmp_path / "feedback", minimum_failure_patients=1, max_refinement_rounds=2)
    assert frame["color"].isna().all() and definitions == [protected]
    report = json.loads((tmp_path / "feedback" / "round_001" / "result.json").read_text())
    assert report["decisions"][0]["action"] == "immutable_explicit_feature"


def test_numeric_verification_respects_confidence_threshold():
    client = FakeClient(["bin_3"]*3+["true"],
                        policy=replace(DecisionExtractionConfig(enabled=True), verification_min_probability=.95))
    result = decision.measure_feature(feature=feature(), source="Documented 89 kg", evidence=evidence("Documented 89 kg"), client=client)
    assert result["status"] == "numeric_verification_failed" and result["value"] is None


@pytest.mark.parametrize("categorical", [True, False])
def test_whole_fold_uses_decisions_and_freezes_measurements(tmp_path, monkeypatch, retrieval, categorical):
    from oci.inference.plain_handoff_stage2 import PlainHandoffStage2Config, Stage2ExtractionLLMConfig
    from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig

    dataset = pd.DataFrame({"patient_id": ["p0", "p1", "p2", "p3"],
        "text": ["red", "blue", "red", "HELDOUT_ONLY"], "treatment": [0, 1, 0, 1], "outcome": [0, 1, 1, 0]})
    client = FakeClient(choose=lambda text, criterion, options: (
        "category_0" if categorical else "true" if len(options) == 2 else "bin_3"))
    config = PlainHandoffStage2Config(endpoint="http://primary.test/v1", model="primary",
        extraction_llm=Stage2ExtractionLLMConfig(endpoint="http://plumb.test/v1", model="plumb", workers=1),
        decision_extraction=client.policy, colbert=ColBERTConfig(devices=("cpu",)),
        extraction_context_strategy="colbert",
        role_adjudication=Stage2RoleAdjudicationConfig(enabled=False), estimation_trees=10)
    monkeypatch.setattr(analysis, "_request_aggregate_ontology_supervisor",
                        lambda **kw: pytest.fail("unconditional ontology LLM review attempted"))
    monkeypatch.setattr(analysis, "select_stage2_features_elastic_net", lambda **kw: (
        [{**f, "roles": ["confounder"], "nuisance_model_roles": ["treatment", "outcome"]} for f in kw["definitions"]],
        {"schema_version": analysis.STAGE2_ROLE_SELECTION_SCHEMA_VERSION, "status": "complete"}, kw["definitions"], []))
    monkeypatch.setattr(analysis, "estimate_outer_fold", lambda **kw: {"status": "tested", "estimator": kw["estimator"]})
    args = dict(dataset=dataset, definitions=[feature("color", categorical=categorical)],
        split={"outer_fold": 1, "fit_row_ids": [0, 1, 2], "heldout_row_ids": [3],
               "inner_splits": [{"inner_fold": 1, "fit_row_ids": [0, 1], "heldout_row_ids": [2]},
                                {"inner_fold": 2, "fit_row_ids": [1, 2], "heldout_row_ids": [0]}]},
        clinical_question="Effect?", unit_id_column="patient_id", text_column="text", treatment_column="treatment",
        outcome_column="outcome", outcome_type="binary", inner_folds=2, seed=3, output_dir=tmp_path,
        request_json=lambda *a, **kw: pytest.fail("unexpected generative request"), config=config, decision_client=client)
    # Managed serving prepares ontologies before the extractor starts, including
    # older checkpoints that predate the fold's measurement-method marker.
    round_dir = tmp_path / "ontology_supervision" / "round_001"
    analysis._write_json(round_dir / "definitions_before_extraction.json",
                         {"features": args["definitions"]})
    prepare_ontologies(args["definitions"],
        output_dir=round_dir / "failure_ontology_refinement", request_json=args["request_json"])
    result = analysis.run_fold_analysis(**args)
    assert result["estimation"]["status"] == "tested"
    assert len(client.calls) == (4 if categorical else 16)
    assert retrieval[-1][0] == "HELDOUT_ONLY"
    analysis.run_fold_analysis(**args)
    assert len(client.calls) == (4 if categorical else 16)
    assert json.loads((tmp_path / "measurement_method.json").read_text())["method"] == "plumb_decision"


@pytest.mark.parametrize("artifact", [
    "ontology_supervision/round_001/extraction/extracted.csv",
    "ontology_supervision/round_001/extraction/decisions/row_00000000/result.json",
    "ontology_supervision/round_001/harmonization/result.json",
    "ontology_supervision/round_001/failure_ontology_refinement/round_001/extraction/extracted.csv",
    "extraction/extracted.csv",
    "preselection/complete.json",
])
def test_unmarked_measurements_still_block_prepared_fold(tmp_path, retrieval, artifact):
    prepare_ontologies([feature()],
        output_dir=tmp_path / "ontology_supervision/round_001/failure_ontology_refinement",
        request_json=lambda *a, **kw: pytest.fail("unexpected generative request"))
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="Existing measurements"):
        decision.claim_output_method(tmp_path, DecisionExtractionConfig(enabled=True),
                                     ColBERTConfig(devices=("cpu",)), fold_root=True)
    assert not (tmp_path / "measurement_method.json").exists()


def test_prepared_fold_rejects_changed_measurement_policy(tmp_path, retrieval):
    f = feature()
    del f["decision_ontology"]
    prepare_ontologies([f],
        output_dir=tmp_path / "ontology_supervision/round_001/failure_ontology_refinement",
        request_json=lambda messages, validate, **kw: validate({
            "value_type": "continuous", "categories_or_unit": ["kg"],
            "decision_ontology": {"minimum": 0, "maximum": 250}, "rationale": "Plausible domain"}))
    policy, retrieval_config = DecisionExtractionConfig(enabled=True), ColBERTConfig(devices=("cpu",))
    original = decision.claim_output_method(tmp_path, policy, retrieval_config, fold_root=True)
    assert decision.claim_output_method(tmp_path, policy, retrieval_config, fold_root=True) == original
    with pytest.raises(ValueError, match="Decision measurement policy changed"):
        decision.claim_output_method(tmp_path, policy, replace(retrieval_config, top_k=10), fold_root=True)
    assert json.loads((tmp_path / "measurement_method.json").read_text()) == original
