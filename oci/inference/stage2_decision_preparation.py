"""CPU-only prompt preparation, shared by local and spawned-process execution."""

import os

_TOKENIZER = None


def encode_conversations(tokenizer, conversations):
    texts = [tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False) for messages in conversations]
    return tokenizer(texts, add_special_tokens=False)["input_ids"]


def prepare_prompt(source, ranked, criterion, options, max_prompt_tokens, encode_batch):
    from ..extraction.colbert import render_context
    from .stage2_decision_client import decision_messages

    contexts = [""] + [render_context(source, ranked[:i]) for i in range(1, len(ranked) + 1)]
    encoded = encode_batch([decision_messages(context, criterion, options) for context in contexts])
    if len(encoded[0]) > max_prompt_tokens:
        raise ValueError("Feature ontology alone exceeds the decision prompt token budget")
    selected_count = 0
    for ids in encoded[1:]:
        if len(ids) > max_prompt_tokens:
            break
        selected_count += 1
    if ranked and not selected_count:
        raise ValueError("No complete retrieved chunk fits the decision prompt; shorten the feature contract")
    return {"context": contexts[selected_count], "prompt_token_ids": encoded[selected_count],
        "retrieval_budget": {
            "selected_chunk_indices": [h["chunk_index"] for h in ranked[:selected_count]],
            "omitted_chunk_indices": [h["chunk_index"] for h in ranked[selected_count:]],
            "max_prompt_tokens": max_prompt_tokens,
        }}


def initialize_worker(tokenizer_name, tokenizer_revision):
    # Spawn rather than fork an interpreter that owns CUDA models. These workers
    # only load a tokenizer, and do not create nested native thread pools.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    from transformers import AutoTokenizer

    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name,
        revision=tokenizer_revision or None, local_files_only=True, trust_remote_code=False)


def prepare_in_worker(source, ranked, criterion, options, max_prompt_tokens):
    prepared = prepare_prompt(source, ranked, criterion, options, max_prompt_tokens,
        lambda conversations: encode_conversations(_TOKENIZER, conversations))
    return os.getpid(), prepared
