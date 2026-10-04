#!/usr/bin/env python3
"""Replay audited decision prompts against one managed Plumb replica.

Measures serving throughput, excluding retrieval and ontology generation. Prefix
caching is disabled to avoid overstating throughput from repeated benchmark inputs.
Only this script's server is stopped on completion. Use a fresh output directory.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", type=Path, default=ROOT / "artifacts/plumb_feasibility/five_conf_five_mod_probe/extraction/decisions")
    parser.add_argument("--model", default=str(ROOT / "artifacts/plumb_feasibility/model"))
    parser.add_argument("--gpu", default="1", help="Logical GPU index under CUDA_VISIBLE_DEVICES")
    parser.add_argument("--port", type=int, default=8134)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.28)
    parser.add_argument("--concurrency", default="8,16,32,64,128")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    levels = [int(v) for v in args.concurrency.split(",")]
    if not levels or min(levels) < 1 or args.repeats < 1:
        parser.error("concurrency and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    from oci.inference.stage2_decision_client import VLLMDecisionClient, probabilities_from_response
    from oci.inference.stage2_decision_config import DecisionExtractionConfig, classifier_vllm_config
    from oci.inference.stage2_endpoint_pool import ExtractionEndpoint
    from oci.inference.vllm_server_pool import ManagedVLLMConfig, launch_managed_vllm_servers

    policy = replace(DecisionExtractionConfig(), enabled=True,
                     tokenizer_name=args.model, tokenizer_revision="")
    encoder = VLLMDecisionClient(policy=policy, model=args.model,
        endpoints=(ExtractionEndpoint("http://127.0.0.1:1/v1", 1),))
    prompts = []
    for path in sorted(args.audits.rglob("result.json")):
        for call in json.loads(path.read_text()).get("calls", []):
            ids = encoder.encode(call["messages"])
            if len(ids) > policy.max_prompt_tokens:
                raise ValueError("Audit prompt exceeds the decision token budget")
            prompts.append({"input": ids, "options": [(v["key"], v["description"]) for v in call["options"]],
                            "prior_selected": call["selected"], "source": str(path.relative_to(args.audits)),
                            "phase": call["phase"]})
    if not prompts:
        raise ValueError("No audited decision prompts found")
    manifest = {"scope": "serving_only_replayed_audits", "prefix_caching": False,
        "model": args.model, "gpu": args.gpu, "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization, "concurrency": levels,
        "prompts": [{k: v for k, v in p.items() if k not in {"input", "options"}} | {
            "tokens": len(p["input"]), "sha256": hashlib.sha256(json.dumps(p["input"]).encode()).hexdigest()}
            for p in prompts]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Loaded {len(prompts)} prompts: {min(len(p['input']) for p in prompts)}–"
          f"{max(len(p['input']) for p in prompts)} tokens", flush=True)
    config = classifier_vllm_config(ManagedVLLMConfig(
        server_count=1, gpus=(f"cuda:{args.gpu.removeprefix('cuda:')}",),
        base_port=args.port, internal_port_base=42000, startup_timeout=600,
        extra_args=("--gpu-memory-utilization", str(args.gpu_memory_utilization),
                    "--max-model-len", "3072", "--max-num-seqs", str(args.max_num_seqs),
                    "--max-num-batched-tokens", str(args.max_num_batched_tokens),
                    "--enforce-eager", "--dtype", "bfloat16", "--no-enable-prefix-caching")))
    summaries = []
    with launch_managed_vllm_servers(config=config, model=args.model, api_key="EMPTY",
                                    output_dir=args.output / "server") as endpoints:
        client = VLLMDecisionClient(policy=policy, model=args.model, tokenizer=encoder.tokenizer,
            endpoints=(ExtractionEndpoint(endpoints[0], max(levels)),), workers=max(levels))

        def request(index):
            prompt = prompts[index % len(prompts)]
            started = time.perf_counter()
            result = client._http(endpoints[0], {"model": args.model, "input": prompt["input"],
                "use_activation": False, "add_special_tokens": False}, timeout=120)
            latency = time.perf_counter() - started
            if result.get("model") != args.model:
                raise ValueError("Response model mismatch")
            probabilities = probabilities_from_response(result, prompt["options"], policy.temperature,
                                                        prompt_tokens=len(prompt["input"]))
            selected = max(probabilities, key=probabilities.get)
            return {"prompt": index % len(prompts), "latency_seconds": latency,
                    "selected": selected, "agrees_with_audit": selected == prompt["prior_selected"],
                    "tokens": len(prompt["input"])}

        # Warm kernels without counting startup or first-use compilation.
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(request, range(min(len(prompts), 16))))
        jobs = list(range(len(prompts) * args.repeats))
        random.Random(20261004).shuffle(jobs)
        for trial, concurrency in enumerate(levels):
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                started = time.perf_counter()
                results = list(pool.map(request, jobs))
                elapsed = time.perf_counter() - started
            latencies = sorted(r["latency_seconds"] for r in results)
            summary = {"trial": trial, "concurrency": concurrency, "requests": len(results),
                "elapsed_seconds": elapsed, "requests_per_second": len(results) / elapsed,
                "input_tokens_per_second": sum(r["tokens"] for r in results) / elapsed,
                "latency_median_seconds": statistics.median(latencies),
                "latency_p95_seconds": latencies[min(len(latencies)-1, int(.95 * len(latencies)))],
                "agreement_with_audit": sum(r["agrees_with_audit"] for r in results) / len(results)}
            summaries.append(summary)
            (args.output / f"trial_{trial:02d}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")
            (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
