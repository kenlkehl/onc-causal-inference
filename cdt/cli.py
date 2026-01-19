"""Command-line interface for propensity score matching experiments."""

import argparse
import sys
from pathlib import Path
import logging

from .config import (
    PropensityExperimentConfig,
    create_propensity_config,
    # Legacy imports for backward compatibility
    ExperimentConfig,
    create_default_config
)
from .utils.system import setup_logging, limit_threads


def main():
    """Main entry point for CLI."""
    limit_threads(n_threads=1)

    parser = argparse.ArgumentParser(
        description="Propensity Score Matching for Causal Inference from Clinical Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default propensity matching config
  psm init --output config.json

  # Run propensity score matching experiment
  psm run --config config.json

  # Run with custom settings
  psm run --config config.json --device cuda:0 --cv-folds 10

  # Legacy DragonNet mode (deprecated)
  psm legacy-run --config legacy_config.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # ==========================================================================
    # INIT command - create default config
    # ==========================================================================
    init_parser = subparsers.add_parser('init', help='Create default configuration file')
    init_parser.add_argument(
        '--output', '-o',
        default='psm_config.json',
        help='Output path for config file (default: psm_config.json)'
    )
    init_parser.add_argument(
        '--legacy',
        action='store_true',
        help='Create legacy DragonNet config (deprecated)'
    )

    # ==========================================================================
    # RUN command - run propensity score matching
    # ==========================================================================
    run_parser = subparsers.add_parser('run', help='Run propensity score matching experiment')
    run_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to configuration JSON file'
    )
    run_parser.add_argument(
        '--device',
        help='Override device from config (e.g., cuda:0, cpu)'
    )
    run_parser.add_argument(
        '--cv-folds',
        type=int,
        help='Override number of CV folds (1 = no CV)'
    )
    run_parser.add_argument(
        '--output-dir',
        help='Override output directory from config'
    )
    run_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    run_parser.add_argument(
        '--encoder',
        choices=['cnn', 'transformer', 'gru'],
        help='Override encoder architecture'
    )
    run_parser.add_argument(
        '--joint-outcome',
        action='store_true',
        help='Enable joint outcome prediction (encourages learning true confounders)'
    )
    run_parser.add_argument(
        '--no-joint-outcome',
        action='store_true',
        help='Disable joint outcome prediction'
    )
    run_parser.add_argument(
        '--matching-method',
        choices=['nearest', 'optimal', 'caliper'],
        help='Override matching method'
    )
    run_parser.add_argument(
        '--caliper',
        type=float,
        help='Override caliper value for matching'
    )

    # ==========================================================================
    # LEGACY-RUN command - run old DragonNet experiments (deprecated)
    # ==========================================================================
    legacy_parser = subparsers.add_parser(
        'legacy-run',
        help='Run legacy DragonNet experiment (deprecated)'
    )
    legacy_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to legacy configuration JSON file'
    )
    legacy_parser.add_argument('--device', help='Override device')
    legacy_parser.add_argument('--workers', type=int, help='Override workers')
    legacy_parser.add_argument('--output-dir', help='Override output directory')
    legacy_parser.add_argument('--verbose', '-v', action='store_true')
    legacy_parser.add_argument('--skip-pretraining', action='store_true')
    legacy_parser.add_argument('--skip-plasmode', action='store_true')

    args = parser.parse_args()

    # ==========================================================================
    # Handle commands
    # ==========================================================================
    if args.command == 'init':
        if args.legacy:
            create_default_config(args.output)
            print(f"\nLegacy config created: {args.output}")
            print("Note: Legacy DragonNet mode is deprecated. Consider using propensity matching.")
        else:
            create_propensity_config(args.output)
            print(f"\nPropensity matching config created: {args.output}")

        print(f"\nEdit {args.output} and then run:")
        print(f"  psm run --config {args.output}")
        return 0

    elif args.command == 'run':
        return run_propensity_matching(args)

    elif args.command == 'legacy-run':
        return run_legacy_dragonnet(args)

    else:
        parser.print_help()
        return 1


def run_propensity_matching(args):
    """Run propensity score matching experiment."""
    import torch
    import pandas as pd

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)
    logger = logging.getLogger(__name__)

    # Load config
    try:
        config = PropensityExperimentConfig.from_json(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    # Apply overrides
    if args.device:
        config.device = args.device
    if args.cv_folds:
        config.cv_folds = args.cv_folds
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.encoder:
        config.model.encoder_type = args.encoder
    if args.joint_outcome:
        config.model.joint_outcome_prediction = True
    if args.no_joint_outcome:
        config.model.joint_outcome_prediction = False
    if args.matching_method:
        config.matching.method = args.matching_method
    if args.caliper is not None:
        config.matching.caliper = args.caliper

    # Validate
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    # Set device
    if config.device:
        device = torch.device(config.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")

    # Load dataset
    dataset_path = Path(config.dataset_path)
    if dataset_path.suffix == '.parquet':
        dataset = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == '.csv':
        dataset = pd.read_csv(dataset_path)
    else:
        print(f"Unsupported file format: {dataset_path.suffix}")
        return 1

    logger.info(f"Loaded dataset: {len(dataset)} samples")

    # Import and run pipeline
    from .training.propensity_training import run_propensity_matching_pipeline
    from .data import EmbeddingCache

    # Setup cache if specified
    cache = None
    if config.cache_dir:
        cache = EmbeddingCache(cache_dir=config.cache_dir)

    # Convert matching config to dict
    matching_config = {
        'method': config.matching.method,
        'caliper': config.matching.caliper,
        'caliper_scale': config.matching.caliper_scale,
        'ratio': config.matching.ratio,
        'replacement': config.matching.replacement
    }

    try:
        results = run_propensity_matching_pipeline(
            dataset=dataset,
            config=config.model,
            training_config=config.training,
            matching_config=matching_config,
            output_path=Path(config.output_dir),
            device=device,
            cache=cache,
            cv_folds=config.cv_folds
        )

        print(f"\n{'='*80}")
        print("PROPENSITY SCORE MATCHING COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {config.output_dir}")

        if 'summary' in results:
            summary = results['summary']
            print(f"\nSummary:")
            print(f"  Samples: {summary['n_samples']}")
            print(f"  Treated: {summary['n_treated']}, Control: {summary['n_control']}")
            print(f"  Matched pairs: {summary['n_matched_pairs']}")
            print(f"  Crude difference: {summary['crude_difference']:.4f}")
            print(f"  IPW ATE: {summary['ate_ipw_estimate']:.4f} "
                  f"[{summary['ate_ipw_ci_lower']:.4f}, {summary['ate_ipw_ci_upper']:.4f}]")

            if 'att_matched_estimate' in summary:
                print(f"  Matched ATT: {summary['att_matched_estimate']:.4f} "
                      f"[{summary['att_matched_ci_lower']:.4f}, {summary['att_matched_ci_upper']:.4f}]")

        return 0

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


def run_legacy_dragonnet(args):
    """Run legacy DragonNet experiment (deprecated)."""
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)

    print("WARNING: Legacy DragonNet mode is deprecated. Consider using propensity matching.")

    try:
        config = ExperimentConfig.from_json(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    if args.device:
        config.device = args.device
    if args.workers:
        config.num_workers = args.workers
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.skip_pretraining:
        config.pretraining.enabled = False
    if args.skip_plasmode:
        config.plasmode_experiments.enabled = False

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    from .experiments.runner import ExperimentRunner

    runner = ExperimentRunner(config)

    try:
        results = runner.run()
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {config.output_dir}")
        return 0

    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
