# synthetic_data/cli.py
"""Command-line interface for synthetic data generation."""

import argparse
import logging
import os
import sys
from pathlib import Path

from .config import SyntheticDataConfig, LLMConfig, DEFAULT_CLINICAL_QUESTION
from .generator import generate_synthetic_dataset, generate_synthetic_dataset_batch


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinical datasets with known causal structure using LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with local vLLM server
  python -m synthetic_data.cli --api-url http://localhost:8000/v1 --dataset-size 100

  # Generate with OpenAI API
  python -m synthetic_data.cli --api-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY --model gpt-4

  # Load from config file (CLI args override config file values)
  python -m synthetic_data.cli --config my_config.json --dataset-size 1000

  # Custom clinical question with positivity enforcement
  python -m synthetic_data.cli --clinical-question "Compare pembrolizumab with nivolumab for NSCLC" --enforce-positivity

    # Single GPU (original behavior)                                                                                                                           
  python -m synthetic_data.cli --use-vllm-batch --tensor-parallel-size 2                                                                                     
                                                                                                                                                             
  # Multi-GPU with 2 parallel workers (4 GPUs, 2 per worker)                                                                                                 
  python -m synthetic_data.cli --use-vllm-batch \                                                                                                            
    --gpu-devices 0,1,2,3 --tensor-parallel-size 2 \                                                                                                         
    --dataset-size 100 --output-dir ./test_multi_gpu                                                                                                         
                                                                                                                                                             
  # Multi-GPU with 4 parallel workers (8 GPUs, 2 per worker)                                                                                                 
  python -m synthetic_data.cli --use-vllm-batch \                                                                                                            
    --gpu-devices 0,1,2,3,4,5,6,7 --tensor-parallel-size 2   
        """,
    )

    # Config file (loaded first, then CLI args override)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. CLI arguments override config file values.",
    )

    # Clinical question
    parser.add_argument(
        "--clinical-question",
        type=str,
        default=DEFAULT_CLINICAL_QUESTION,
        help="Comparative effectiveness research question",
    )
    
    # Dataset parameters
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=500,
        help="Number of patients to generate (default: 500)",
    )
    parser.add_argument(
        "--treatment-effect",
        type=float,
        default=0.20,
        help="Target treatment effect on probability scale (e.g., 0.10 = 10%% increase in outcome probability, default: 0.20)",
    )
    parser.add_argument(
        "--target-treatment-rate",
        type=float,
        default=0.5,
        help="Target proportion of patients receiving treatment=1 (default: 0.5)",
    )
    parser.add_argument(
        "--target-control-outcome-rate",
        type=float,
        default=0.5,
        help="Target outcome rate in control group (treatment=0) (default: 0.5)",
    )
    parser.add_argument(
        "--num-confounders",
        type=int,
        default=None,
        help="Number of confounders to generate (default: 8-12, determined by LLM)",
    )

    # Positivity enforcement
    parser.add_argument(
        "--enforce-positivity",
        action="store_true",
        help="Enforce minimum treatment/control rates per confounder stratum (avoids positivity violations)",
    )
    parser.add_argument(
        "--min-treatment-rate",
        type=float,
        default=0.1,
        help="Minimum P(T=1|X) per stratum when --enforce-positivity is set (default: 0.1)",
    )
    parser.add_argument(
        "--max-treatment-rate",
        type=float,
        default=0.9,
        help="Maximum P(T=1|X) per stratum when --enforce-positivity is set (default: 0.9)",
    )
    parser.add_argument(
        "--target-logit-std",
        type=float,
        default=2.0,
        help="Target std of logits; lower values compress propensities toward 0.5 (default: 2.0)",
    )

    # LLM parameters
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:8000/v1 for vLLM)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key (can be blank for local models)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model name to use (default: 'openai/gpt-oss-120b')",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50000,
        help="Max tokens per LLM response (default: 20000)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synthetic_output",
        help="Output directory for generated files (default: ./synthetic_output)",
    )
    
    # Execution
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=40,
        help="Number of parallel workers for patient generation (default: 4)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--use-vllm-batch",
        action="store_true",
        help="Use direct vLLM batch inference (faster, no server needed)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM (default: 2)",
    )
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default=None,
        help="Comma-separated GPU device IDs for multi-GPU parallelization (e.g., '0,1,2,3'). "
             "When specified with --use-vllm-batch, spawns multiple parallel workers. "
             "Number of workers = len(gpu_devices) / tensor_parallel_size. "
             "If not specified, uses CUDA_VISIBLE_DEVICES or all available GPUs.",
    )
    parser.add_argument(
        "--reasoning-marker",
        type=str,
        default="assistantfinal",
        help="Marker to strip reasoning prefix from clinical text (default: 'assistantfinal'). Set to empty string to disable.",
    )

    parser.add_argument(
        "--vllm-download-dir",
        type=str,
        default="./",
        help="Download directory for vllm model",
    )
    parser.add_argument(
        "--outcome-type",
        type=str,
        choices=["binary", "continuous"],
        default="binary",
        help="Type of outcome: 'binary' (default) or 'continuous'",
    )
    parser.add_argument(
        "--outcome-noise-std",
        type=float,
        default=1.0,
        help="Noise standard deviation for continuous outcomes (default: 1.0)",
    )
    
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build configuration: load from file first, then override with CLI args
    if args.config:
        logging.info(f"Loading config from: {args.config}")
        config = SyntheticDataConfig.from_json(args.config)
        # Override with any explicitly provided CLI args
        # We check against defaults to see if user provided a value
        if args.clinical_question != DEFAULT_CLINICAL_QUESTION:
            config.clinical_question = args.clinical_question
        if args.dataset_size != 500:
            config.dataset_size = args.dataset_size
        if args.treatment_effect != 0.20:
            config.treatment_effect_prob = args.treatment_effect
        if args.target_treatment_rate != 0.5:
            config.target_treatment_rate = args.target_treatment_rate
        if args.target_control_outcome_rate != 0.5:
            config.target_control_outcome_rate = args.target_control_outcome_rate
        if args.enforce_positivity:
            config.enforce_positivity = True
        if args.min_treatment_rate != 0.1:
            config.min_treatment_rate_per_stratum = args.min_treatment_rate
        if args.max_treatment_rate != 0.9:
            config.max_treatment_rate_per_stratum = args.max_treatment_rate
        if args.target_logit_std != 2.0:
            config.target_logit_std = args.target_logit_std
        if args.num_confounders is not None:
            config.num_confounders = args.num_confounders
        if args.outcome_type != "binary":
            config.outcome_type = args.outcome_type
        if args.outcome_noise_std != 1.0:
            config.outcome_noise_std = args.outcome_noise_std
        if args.output_dir != "./synthetic_output":
            config.output_dir = args.output_dir
        if args.seed != 42:
            config.seed = args.seed
        # LLM overrides
        if args.api_url != "http://localhost:8000/v1":
            config.llm.api_base_url = args.api_url
        if args.api_key != "":
            config.llm.api_key = args.api_key
        if args.model != "openai/gpt-oss-120b":
            config.llm.model_name = args.model
        if args.temperature != 0.7:
            config.llm.temperature = args.temperature
        if args.max_tokens != 50000:
            config.llm.max_tokens = args.max_tokens
    else:
        # Build config from CLI args
        config = SyntheticDataConfig(
            clinical_question=args.clinical_question,
            dataset_size=args.dataset_size,
            treatment_effect_prob=args.treatment_effect,
            target_treatment_rate=args.target_treatment_rate,
            target_control_outcome_rate=args.target_control_outcome_rate,
            enforce_positivity=args.enforce_positivity,
            min_treatment_rate_per_stratum=args.min_treatment_rate,
            max_treatment_rate_per_stratum=args.max_treatment_rate,
            target_logit_std=args.target_logit_std,
            num_confounders=args.num_confounders,
            outcome_type=args.outcome_type,
            outcome_noise_std=args.outcome_noise_std,
            output_dir=args.output_dir,
            seed=args.seed,
            llm=LLMConfig(
                api_base_url=args.api_url,
                api_key=args.api_key,
                model_name=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ),
        )

    # Save config to output directory before generation
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_output_path = output_dir / "generation_config.json"
    config.to_json(str(config_output_path))
    logging.info(f"Config saved to: {config_output_path}")

    # Run generation
    try:
        if args.use_vllm_batch:
            # Use direct vLLM batch inference (faster)
            from .vllm_batch_client import VLLMConfig

            # Parse GPU devices if specified
            gpu_device_ids = None
            if args.gpu_devices:
                try:
                    gpu_device_ids = [int(d.strip()) for d in args.gpu_devices.split(",")]
                except ValueError:
                    logging.error(f"Invalid --gpu-devices format: {args.gpu_devices}. Expected comma-separated integers.")
                    sys.exit(1)

                # Validate: number of GPUs must be divisible by tensor_parallel_size
                if len(gpu_device_ids) % args.tensor_parallel_size != 0:
                    logging.error(
                        f"Number of GPU devices ({len(gpu_device_ids)}) must be divisible by "
                        f"tensor_parallel_size ({args.tensor_parallel_size})"
                    )
                    sys.exit(1)

                num_workers = len(gpu_device_ids) // args.tensor_parallel_size
                logging.info(f"Multi-GPU mode: {len(gpu_device_ids)} GPUs -> {num_workers} parallel workers "
                            f"(tensor_parallel_size={args.tensor_parallel_size})")

            vllm_config = VLLMConfig(
                model_name=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                download_dir=args.vllm_download_dir,
                reasoning_marker=args.reasoning_marker if args.reasoning_marker else None,
            )
            df, metadata = generate_synthetic_dataset_batch(
                config=config,
                vllm_config=vllm_config,
                show_progress=True,
                gpu_device_ids=gpu_device_ids,
            )
        else:
            df, metadata = generate_synthetic_dataset(
                config=config,
                num_workers=args.num_workers,
                show_progress=True,
            )
        
        print(f"\n✓ Generated {len(df)} patients")
        print(f"  - Treatment rate: {df['treatment_indicator'].mean():.1%}")
        if config.outcome_type == "continuous":
            print(f"  - Outcome mean: {df['outcome_indicator'].mean():.2f} (std: {df['outcome_indicator'].std():.2f})")
        else:
            print(f"  - Outcome rate: {df['outcome_indicator'].mean():.1%}")
        if config.enforce_positivity:
            print(f"  - Positivity enforcement: ON (min={config.min_treatment_rate_per_stratum:.0%}, max={config.max_treatment_rate_per_stratum:.0%})")
        print(f"  - Config: {config.output_dir}/generation_config.json")
        print(f"  - Dataset: {config.output_dir}/dataset.parquet")
        print(f"  - Metadata: {config.output_dir}/metadata.json")
        
    except Exception as e:
        logging.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
