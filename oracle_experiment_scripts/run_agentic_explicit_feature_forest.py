#!/usr/bin/env python3
"""Run agentic explicit-feature causal forest search on a Parquet dataset."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from oci.config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    ModelArchitectureConfig,
)
from oci.inference.agentic_explicit_feature_forest import run_agentic_explicit_feature_forest


def _load_specs(path: str | None):
    if not path:
        return []
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        feature_data = data
    elif "features" in data:
        feature_data = data["features"]
    else:
        feature_data = data.get("confounders", []) + data.get("effect_modifiers", [])
    return [
        ExplicitFeatureSpec(**item) if isinstance(item, dict) else item
        for item in feature_data
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Agentic LLM explicit-feature search + causal forest"
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--features-json",
        default=None,
        help="JSON list or metadata.json with initial features; omit to start from no variables",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--outcome-type", default="binary", choices=["binary", "continuous"])

    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-additions-per-iter", type=int, default=6)
    parser.add_argument("--max-removals-per-iter", type=int, default=3)

    parser.add_argument("--cf-n-estimators", type=int, default=200)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    parser.add_argument("--cf-no-inference", action="store_true")

    parser.add_argument("--vllm-mode", default="python_api", choices=["server", "start_server", "python_api"])
    parser.add_argument("--vllm-server-url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model-name", default="nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--extraction-batch-size", type=int, default=64)
    parser.add_argument("--extraction-max-retries", type=int, default=5)
    parser.add_argument("--extraction-max-tokens", type=int, default=10000)
    parser.add_argument("--extraction-max-text-length", type=int, default=80000)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")

    parser.add_argument("--agent-server-url", default="http://localhost:8000/v1")
    parser.add_argument("--agent-model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--agent-api-key", default="EMPTY")
    parser.add_argument("--agent-max-tokens", type=int, default=2048)
    parser.add_argument("--agent-context-chars", type=int, default=4800)
    parser.add_argument("--agent-context-examples", type=int, default=3)
    parser.add_argument("--save-agent-context", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    specs = _load_specs(args.features_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_context_examples = max(0, args.agent_context_examples)
    agent_context_chars = max(0, args.agent_context_chars)
    agent_example_chars = (
        0
        if agent_context_examples == 0
        else max(1, -(-agent_context_chars // agent_context_examples))
    )

    config = AppliedInferenceConfig(
        outcome_type=args.outcome_type,
        dataset_path=args.dataset_path,
        text_column=args.text_column,
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
        cv_folds=args.outer_folds,
        architecture=ModelArchitectureConfig(
            model_type="agentic_explicit_feature_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=args.cf_n_estimators,
                min_samples_leaf=args.cf_min_samples_leaf,
                inference=not args.cf_no_inference,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                max_iterations=args.max_iterations,
                max_additions_per_iter=args.max_additions_per_iter,
                max_removals_per_iter=args.max_removals_per_iter,
                agent_server_url=args.agent_server_url,
                agent_model_name=args.agent_model_name,
                agent_api_key=args.agent_api_key,
                agent_max_tokens=args.agent_max_tokens,
                clinical_text_examples_per_prompt=agent_context_examples,
                clinical_text_example_chars=agent_example_chars,
                save_agent_context=args.save_agent_context,
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=specs,
            vllm_mode=args.vllm_mode,
            vllm_server_url=args.vllm_server_url,
            vllm_model_name=args.vllm_model_name,
            vllm_max_model_len=args.vllm_max_model_len,
            extraction_batch_size=args.extraction_batch_size,
            extraction_max_retries=args.extraction_max_retries,
            extraction_max_tokens=args.extraction_max_tokens,
            extraction_max_text_length=args.extraction_max_text_length,
            cache_enabled=not args.no_cache,
            cache_dir=args.cache_dir,
        ),
    )

    dataset = pd.read_parquet(args.dataset_path)
    output_path = output_dir / "predictions.parquet"
    run_agentic_explicit_feature_forest(dataset, config, output_path)
    print(f"Predictions: {output_path}")
    print(f"Artifacts: {output_path.parent / 'agentic_feature_search'}")


if __name__ == "__main__":
    main()
