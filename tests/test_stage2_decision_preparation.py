"""Exercise real spawned workers, exact prompt parity, and checkpoint identity."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import time

import pytest

from oci.inference.stage2_decision import packed_decision, policy_identity
from oci.inference.stage2_decision_client import VLLMDecisionClient
from oci.inference.stage2_decision_config import DecisionExtractionConfig
from oci.inference.stage2_endpoint_pool import ExtractionEndpoint


def _worker_barrier(directory):
    directory = Path(directory)
    (directory / str(os.getpid())).touch()
    deadline = time.monotonic() + 60
    while not (directory / "release").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("test worker barrier expired")
        time.sleep(.02)
    return os.getpid(), os.environ["CUDA_VISIBLE_DEVICES"], os.environ["TOKENIZERS_PARALLELISM"]


@pytest.fixture
def tokenizer_policy(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab["<unk>"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}{% endfor %}"
        "{% if enable_thinking %}THINK{% endif %}{% if add_generation_prompt %}ASSISTANT{% endif %}"
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    return DecisionExtractionConfig(enabled=True, tokenizer_name=str(tmp_path / "tokenizer"),
                                    tokenizer_revision="", max_prompt_tokens=1800)


def make_client(policy, workers):
    def transport(endpoint, payload, timeout):
        return {"model": "plumb", "data": [{"num_classes": 16, "probs": [8, 0] + [-1]*14}],
                "usage": {"prompt_tokens": len(payload["input"]), "completion_tokens": 0}}

    client = VLLMDecisionClient(policy=policy, model="plumb", preparation_workers=workers,
        endpoints=(ExtractionEndpoint("http://localhost:1/v1", 8),), workers=8, transport=transport)
    client.pool._metrics_reader = None
    return client


def test_spawned_cpu_workers_preserve_prompt_ids_and_budget_and_close(tokenizer_policy, tmp_path):
    source = "é漢 red blue " * 200
    hits = [{"start": start, "end": start + 200, "chunk_index": i, "score": 1.0}
            for i, start in enumerate([800, 0, 400, 200, 600, 1000, 1200, 1400, 1600])]
    kwargs = dict(source=source, evidence={"hits": [hits]}, criterion="Documented color",
                  options=[("red", "Red"), ("blue", "Blue")])
    local, parallel = make_client(tokenizer_policy, 0), make_client(tokenizer_policy, 2)
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    pool = None
    try:
        expected = packed_decision(local, **kwargs)
        actual = packed_decision(parallel, **kwargs)
        assert actual == expected
        assert actual["retrieval_budget"]["omitted_chunk_indices"]
        pool = parallel._preparation_pool
        # Occupy each worker until both independent interpreters are ready.
        waiting = [pool.submit(_worker_barrier, str(barrier)) for _ in range(2)]
        deadline = time.monotonic() + 60
        while len(list(barrier.iterdir())) < 2 and time.monotonic() < deadline:
            time.sleep(.02)
        assert len(list(barrier.iterdir())) == 2
        (barrier / "release").touch()
        workers = [job.result(timeout=10) for job in waiting]
        assert len({pid for pid, _, _ in workers}) == 2
        assert all(cuda == "" and native == "false" for _, cuda, native in workers)
        with ThreadPoolExecutor(max_workers=8) as threads:
            results = list(threads.map(lambda _: packed_decision(parallel, **kwargs), range(32)))
        assert all(result == expected for result in results)
        ids = parallel.prepare_prompt(source=source, ranked=hits,
            criterion=kwargs["criterion"], options=kwargs["options"])["prompt_token_ids"]
        assert ids == local.encode(expected["messages"])
        assert "THINK" not in local.tokenizer.decode(ids)
        for error_kwargs, message in (({**kwargs, "criterion": "x" * 4000}, "ontology alone"),
                ({**kwargs, "evidence": {"hits": [[{"start": 0, "end": len(source), "chunk_index": 0}]]}}, "No complete")):
            with pytest.raises(ValueError, match=message):
                packed_decision(parallel, **error_kwargs)
        empty = {**kwargs, "evidence": {"hits": [[]]}}
        assert packed_decision(parallel, **empty) == packed_decision(local, **empty)
        processes = list(pool._processes.values())
    finally:
        (barrier / "release").touch()
        parallel.close()
        local.close()
    assert all(not process.is_alive() for process in processes)
    with pytest.raises(RuntimeError, match="closed"):
        packed_decision(parallel, **kwargs)


@pytest.mark.parametrize("workers", [-1, True, 1.5, "2", None])
def test_invalid_cpu_worker_counts_rejected(workers):
    from oci.inference.plain_handoff_stage2 import plain_stage2_config_from_mapping

    with pytest.raises(ValueError, match="decision_preparation_workers"):
        plain_stage2_config_from_mapping({"endpoint": "http://localhost:1/v1",
            "decision_preparation_workers": workers}, default_workers=2)
    with pytest.raises(ValueError, match="decision_preparation_workers"):
        make_client(DecisionExtractionConfig(), workers)


def test_cpu_parallelism_preserves_frozen_definition_and_measurement_identities():
    from oci.inference.plain_handoff_stage2 import plain_stage2_config_from_mapping, _feature_definition_input_value

    before = plain_stage2_config_from_mapping({"endpoint": "http://localhost:1/v1", "model": "primary",
        "decision_extraction": {"enabled": True},
        "extraction_llm": {"model": "plumb", "endpoint": "http://localhost:1/v1"}}, default_workers=2)
    after = replace(before, decision_preparation_workers=16)
    kwargs = dict(clinical_question="Question", outer_fold=1, discovery_packets=[], seed=42)
    assert _feature_definition_input_value(config=before, **kwargs) == _feature_definition_input_value(config=after, **kwargs)
    assert policy_identity(before.decision_extraction, before.colbert) == policy_identity(after.decision_extraction, after.colbert)
