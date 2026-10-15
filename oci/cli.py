"""Command-line interface for OCI experiments."""

import argparse
import json
import sys
import logging

from .config import (
    ExperimentConfig,
    create_default_config,
    load_explicit_feature_specs_json,
    normalize_multi_model_feature_discovery_methods,
    parse_explicit_feature_spec_entries,
)
from .experiments.runner import ExperimentRunner
from .utils.system import setup_logging, limit_threads


def main():
    """Main entry point for OCI CLI."""
    limit_threads(n_threads=1)
    
    parser = argparse.ArgumentParser(
        description="Oncology Causal Inference: Causal inference from clinical text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default config
  oci init --output config.json

  # Run experiment with config
  oci run --config config.json

  # Run with custom settings
  oci run --config config.json --device cuda:0 --workers 4
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    init_parser = subparsers.add_parser('init', help='Create default configuration file')
    init_parser.add_argument(
        '--output', '-o',
        default='oci_config.json',
        help='Output path for config file (default: oci_config.json)'
    )
    
    run_parser = subparsers.add_parser('run', help='Run experiment from config')
    run_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to configuration JSON file'
    )
    run_parser.add_argument(
        '--device',
        help='Override device from config (e.g., cuda:0, mps, cpu)'
    )
    run_parser.add_argument(
        '--workers',
        type=int,
        help='Override number of workers from config'
    )
    run_parser.add_argument(
        '--gpu-ids',
        nargs='+',
        type=int,
        help='Override GPU ids from config, e.g. --gpu-ids 0 1 2 3'
    )
    run_parser.add_argument(
        '--output-dir',
        help='Override output directory from config'
    )
    run_parser.add_argument(
        '--multi-model-features-json',
        help=(
            "Add pre-specified feature specs for "
            "model_type='multi_model_agentic_forest' from a JSON file. "
            "Accepted keys: features, confounders, effect_modifiers."
        )
    )
    run_parser.add_argument(
        '--multi-model-confounder',
        action='append',
        default=[],
        help=(
            "Add a pre-specified multi-model confounder as a JSON feature spec. "
            "May be repeated."
        )
    )
    run_parser.add_argument(
        '--multi-model-effect-modifier',
        action='append',
        default=[],
        help=(
            "Add a pre-specified multi-model effect modifier as a JSON feature spec. "
            "May be repeated."
        )
    )
    run_parser.add_argument(
        '--multi-model-feature-discovery-methods',
        nargs='+',
        help=(
            "Override feature discovery methods for "
            "model_type='multi_model_agentic_forest'. Accepted values: "
            "bow, htr, embedding_contrast, or all. Comma-separated values are "
            "also accepted."
        )
    )
    run_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    run_parser.add_argument(
        '--r-stage-min-propensity',
        type=float,
        help=(
            'Override minimum nuisance propensity score eligible for '
            'agentic_attention_variable_forest R-stage training'
        )
    )
    run_parser.add_argument(
        '--r-stage-max-propensity',
        type=float,
        help=(
            'Override maximum nuisance propensity score eligible for '
            'agentic_attention_variable_forest R-stage training'
        )
    )
    run_parser.add_argument(
        '--inner-fold-parallelism',
        help=(
            "Override inner cross-fit fold parallelism for "
            "agentic_attention_variable_forest runs. "
            "Use 'auto' or a positive integer."
        )
    )
    run_parser.add_argument(
        '--outer-fold-parallelism',
        help=(
            "Override outer analysis fold parallelism for "
            "model_type='agentic_attention_variable_forest'. "
            "Use 'auto' or a positive integer."
        )
    )
    run_parser.add_argument(
        '--agent-candidate-parallelism',
        help=(
            "Override per-inner-fold agent candidate proposal parallelism for "
            "model_type='agentic_attention_variable_forest'. "
            "Use 'auto' or a positive integer."
        )
    )
    run_parser.add_argument(
        '--effect-objective',
        choices=['squared_r_loss', 'pseudo_outcome_mse'],
        help=(
            "Override neural effect-stage objective for "
            "model_type='agentic_attention_variable_forest'."
        )
    )
    run_parser.add_argument(
        '--alpha-propensity',
        type=float,
        help="Override applied inference treatment/propensity loss weight."
    )
    
    args = parser.parse_args()
    
    if args.command == 'init':
        create_default_config(args.output)
        print(f"\nEdit {args.output} and then run:")
        print(f"  oci run --config {args.output}")
        return 0
    
    elif args.command == 'run':
        level = logging.DEBUG if args.verbose else logging.INFO
        setup_logging(level=level)
        
        try:
            config = ExperimentConfig.from_json(args.config)
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1
        
        if args.device:
            config.device = args.device
        if args.workers:
            config.num_workers = args.workers
        if args.gpu_ids is not None:
            config.gpu_ids = args.gpu_ids
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.multi_model_feature_discovery_methods is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "multi_model_agentic_forest":
                print(
                    "--multi-model-feature-discovery-methods only applies to "
                    "model_type='multi_model_agentic_forest'"
                )
                return 1
            mm_config = (
                config.applied_inference.architecture
                .multi_model_agentic_forest
            )
            try:
                mm_config.set_feature_discovery_methods(
                    normalize_multi_model_feature_discovery_methods(
                        args.multi_model_feature_discovery_methods,
                        source="--multi-model-feature-discovery-methods",
                    ),
                    source="--multi-model-feature-discovery-methods",
                )
            except ValueError as e:
                print(f"Invalid multi-model feature discovery methods: {e}")
                return 1
        if (
            args.multi_model_features_json
            or args.multi_model_confounder
            or args.multi_model_effect_modifier
        ):
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "multi_model_agentic_forest":
                print(
                    "--multi-model-features-json/--multi-model-confounder/"
                    "--multi-model-effect-modifier only apply to "
                    "model_type='multi_model_agentic_forest'"
                )
                return 1
            mm_config = (
                config.applied_inference.architecture
                .multi_model_agentic_forest
            )
            try:
                if args.multi_model_features_json:
                    mm_config.prespecified_features.extend(
                        load_explicit_feature_specs_json(args.multi_model_features_json)
                    )
                if args.multi_model_confounder:
                    mm_config.prespecified_confounders.extend(
                        parse_explicit_feature_spec_entries(
                            args.multi_model_confounder,
                            default_roles=["confounder"],
                            source="--multi-model-confounder",
                        )
                    )
                if args.multi_model_effect_modifier:
                    mm_config.prespecified_effect_modifiers.extend(
                        parse_explicit_feature_spec_entries(
                            args.multi_model_effect_modifier,
                            default_roles=["effect_modifier"],
                            source="--multi-model-effect-modifier",
                        )
                    )
            except (OSError, json.JSONDecodeError, ValueError) as e:
                print(f"Error loading multi-model pre-specified features: {e}")
                return 1
        if args.r_stage_min_propensity is not None or args.r_stage_max_propensity is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "agentic_attention_variable_forest":
                print(
                    "--r-stage-min-propensity/--r-stage-max-propensity only apply "
                    "to model_type='agentic_attention_variable_forest'"
                )
                return 1
            avf_config = (
                config.applied_inference.architecture
                .agentic_attention_variable_forest
            )
            if args.r_stage_min_propensity is not None:
                avf_config.r_stage_min_propensity = args.r_stage_min_propensity
            if args.r_stage_max_propensity is not None:
                avf_config.r_stage_max_propensity = args.r_stage_max_propensity
            if not (
                0.0
                <= avf_config.r_stage_min_propensity
                < avf_config.r_stage_max_propensity
                <= 1.0
            ):
                print(
                    "R-stage propensity bounds must satisfy 0 <= min < max <= 1"
                )
                return 1
        if args.inner_fold_parallelism is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type == "agentic_attention_variable_forest":
                avf_config = (
                    config.applied_inference.architecture
                    .agentic_attention_variable_forest
                )
                avf_config.fold_parallelism = str(args.inner_fold_parallelism)
            else:
                print(
                    "--inner-fold-parallelism only applies to "
                    "model_type='agentic_attention_variable_forest'"
                )
                return 1
        if args.outer_fold_parallelism is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "agentic_attention_variable_forest":
                print(
                    "--outer-fold-parallelism only applies to "
                    "model_type='agentic_attention_variable_forest'"
                )
                return 1
            (
                config.applied_inference.architecture
                .agentic_attention_variable_forest
                .outer_parallelism
            ) = str(args.outer_fold_parallelism)
        if args.agent_candidate_parallelism is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "agentic_attention_variable_forest":
                print(
                    "--agent-candidate-parallelism only applies to "
                    "model_type='agentic_attention_variable_forest'"
                )
                return 1
            (
                config.applied_inference.architecture
                .agentic_attention_variable_forest
                .candidate_proposal_parallelism
            ) = str(args.agent_candidate_parallelism)
        if args.effect_objective is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "agentic_attention_variable_forest":
                print(
                    "--effect-objective only applies to "
                    "model_type='agentic_attention_variable_forest'"
                )
                return 1
            (
                config.applied_inference.architecture
                .agentic_attention_variable_forest
                .effect_objective
            ) = args.effect_objective
        if args.alpha_propensity is not None:
            config.applied_inference.training.alpha_propensity = args.alpha_propensity
        
        try:
            config.validate()
        except ValueError as e:
            print(f"Configuration error: {e}")
            return 1
        
        runner = ExperimentRunner(config)
        
        try:
            results = runner.run()
            print(f"\n{'='*80}")
            print("EXPERIMENT COMPLETE")
            print(f"{'='*80}")
            print(f"Results saved to: {config.output_dir}")
            
            if results.get('applied_inference'):
                print(f"\nApplied inference results: {results['applied_inference']}")
            
            return 0
            
        except Exception as e:
            logging.error(f"Experiment failed: {e}", exc_info=True)
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
