"""Plumb's decision readout over vLLM's pooling/classify API.

The response field called ``probs`` contains logits when use_activation=false.
Normalize only the offered letters, after applying Plumb's temperature.
"""

import json
import logging
import math
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .stage2_decision_config import LETTERS
from .stage2_endpoint_pool import EndpointPool, ExtractionEndpoint

LOGGER = logging.getLogger(__name__)

SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def decision_messages(evidence, criterion, options):
    if not 2 <= len(options) <= len(LETTERS) or len({k for k, _ in options}) != len(options):
        raise ValueError("decision options require 2–16 distinct keys")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps({
        "evidence": evidence, "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": description}
                    for i, (_, description) in enumerate(options)],
    }, ensure_ascii=False)}]


def probabilities_from_response(response, options, temperature, *, prompt_tokens):
    data = response.get("data")
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError("vLLM classifier must return exactly one result")
    row = data[0]
    logits = row.get("probs")
    if not isinstance(logits, list) or len(logits) != 16 or row.get("num_classes") != 16:
        raise ValueError("vLLM must use the 16-class A–P Plumb readout")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in logits):
        raise ValueError("vLLM returned nonfinite/invalid decision logits")
    if all(0 <= x <= 1 for x in logits) and math.isclose(sum(logits), 1, abs_tol=1e-4):
        raise ValueError("vLLM returned probabilities; configure use_activation=false for raw logits")
    usage = response.get("usage") or {}
    if usage.get("prompt_tokens") != prompt_tokens or usage.get("completion_tokens", 0) != 0:
        raise ValueError("vLLM altered the prompt or generated tokens; refusing this measurement")
    active = logits[:len(options)]
    weights = [math.exp((x - max(active)) / temperature) for x in active]
    return {key: weight / sum(weights) for (key, _), weight in zip(options, weights, strict=True)}


class VLLMDecisionClient:
    def __init__(self, *, policy, model, endpoints, api_key="EMPTY", timeout=120,
                 workers=4, tokenizer=None, transport=None, preparation_workers=0):
        policy.validate()
        if type(preparation_workers) is not int or preparation_workers < 0:
            raise ValueError("decision_preparation_workers must be a nonnegative integer")
        self.preparation_workers = preparation_workers
        self._preparation_pool = None
        self._preparation_lock = threading.Lock()
        self._preparation_worker_pids = set()
        self._closed = False
        self.policy, self.model, self.api_key = policy, model, api_key
        self.timeout = float(timeout)
        if self.timeout <= 0 or not math.isfinite(self.timeout):
            raise ValueError("decision request timeout must be positive and finite")
        self.pool = EndpointPool(endpoints, api_key=api_key, latency_weighted=False)
        self.capacity = threading.BoundedSemaphore(max(1, min(workers, self.pool.capacity)))
        self._tokenizer = tokenizer
        self._tokenizer_lock = threading.Lock()
        self._transport = transport or self._http

    @classmethod
    def from_config(cls, config):
        route = config.extraction_llm
        if route is None:
            raise ValueError("decision extraction requires stage2.extraction_llm as its vLLM route")
        if config.runtime_disable_extraction:
            raise ValueError("decision extraction cannot run with runtime_disable_extraction")
        if route.runtime_endpoint:
            raise ValueError("decision extraction requires its configured classifier, not runtime_endpoint continuation")
        endpoints = route.endpoints or tuple(ExtractionEndpoint(url, route.workers)
            for url in (route.runtime_endpoints or (route.endpoint,)))
        return cls(policy=config.decision_extraction, model=route.model, endpoints=endpoints,
                   api_key=route.api_key, workers=route.workers, timeout=config.request_timeout,
                   preparation_workers=config.decision_preparation_workers)

    @property
    def tokenizer(self):
        with self._tokenizer_lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.policy.tokenizer_name, revision=self.policy.tokenizer_revision or None,
                    trust_remote_code=False,
                )
                if any(len(self._tokenizer.encode(letter, add_special_tokens=False)) != 1 for letter in LETTERS):
                    raise ValueError("decision letters must each be one tokenizer token")
        return self._tokenizer

    def encode(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_batch(self, conversations):
        from .stage2_decision_preparation import encode_conversations

        return encode_conversations(self.tokenizer, conversations)

    def prepare_prompt(self, *, source, ranked, criterion, options):
        from .stage2_decision_preparation import initialize_worker, prepare_in_worker

        with self._preparation_lock:
            if self._closed:
                raise RuntimeError("decision client is closed")
            if self._preparation_pool is None:
                if not self.preparation_workers:
                    raise ValueError("CPU preparation requires decision_preparation_workers > 0")
                # Validate and populate the tokenizer cache before children load
                # the same revision offline. No models or tensors cross IPC.
                self.tokenizer
                self._preparation_pool = ProcessPoolExecutor(
                    max_workers=self.preparation_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_worker,
                    initargs=(self.policy.tokenizer_name, self.policy.tokenizer_revision),
                )
                LOGGER.info("Stage 2 decision CPU preparation processes=%s start_method=spawn",
                            self.preparation_workers)
            ranges = [{key: hit[key] for key in ("start", "end", "chunk_index")} for hit in ranked]
            future = self._preparation_pool.submit(prepare_in_worker,
                source, ranges, criterion, options, self.policy.max_prompt_tokens)
        pid, prepared = future.result(timeout=self.timeout)
        with self._preparation_lock:
            if pid not in self._preparation_worker_pids:
                self._preparation_worker_pids.add(pid)
                LOGGER.info("Stage 2 decision CPU preparation worker ready pid=%s", pid)
        return prepared

    def close(self):
        with self._preparation_lock:
            self._closed = True
            pool, self._preparation_pool = self._preparation_pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def _http(self, endpoint, payload, timeout):
        url = endpoint.rstrip("/").removesuffix("/v1") + "/classify"
        request = Request(url, data=json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def decide(self, evidence, criterion, options, *, prompt_token_ids=None):
        messages = decision_messages(evidence, criterion, options)
        ids = self.encode(messages) if prompt_token_ids is None else prompt_token_ids
        if len(ids) > self.policy.max_prompt_tokens:
            raise ValueError(f"decision prompt has {len(ids)} tokens; limit {self.policy.max_prompt_tokens}")
        payload = {"model": self.model, "input": ids, "use_activation": False,
                   "add_special_tokens": False}
        deadline = time.monotonic() + self.timeout
        if not self.capacity.acquire(timeout=self.timeout):
            raise TimeoutError("decision request expired waiting for global capacity")
        try:
            for attempt in range(3):
                index = self.pool.reserve(deadline=deadline)
                started = time.monotonic()
                failed, succeeded = False, False
                try:
                    remaining = deadline - started
                    if remaining <= 0:
                        raise TimeoutError("decision request deadline expired")
                    response = self._transport(self.pool.server(index).endpoint, payload, remaining)
                    if response.get("model") != self.model:
                        raise ValueError("vLLM classifier response model does not match requested model")
                    probabilities = probabilities_from_response(response, options,
                        self.policy.temperature, prompt_tokens=len(ids))
                    succeeded = True
                    return {"selected": max(probabilities, key=probabilities.get),
                            "probabilities": probabilities, "prompt_tokens": len(ids),
                            "options": [{"key": k, "description": v} for k, v in options],
                            "messages": messages, "raw_logits": response["data"][0]["probs"],
                            "model": self.model, "temperature": self.policy.temperature}
                except (HTTPError, URLError, TimeoutError, OSError) as error:
                    failed = True
                    if isinstance(error, HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}:
                        raise
                    if attempt == 2 or time.monotonic() >= deadline:
                        raise
                finally:
                    self.pool.release(index, duration_seconds=time.monotonic()-started,
                                      transport_failed=failed, succeeded=succeeded)
        finally:
            self.capacity.release()
