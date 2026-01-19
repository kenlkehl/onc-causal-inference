"""Command-line interface for CDT experiments."""

import argparse
import sys
from pathlib import Path
import logging

from .config import ExperimentConfig, create_default_config
from .experiments.runner import ExperimentRunner
from .utils.system import setup_logging, limit_threads


def main():
    """Main entry point for CDT CLI."""
    limit_threads(n_threads=1)

    parser = argparse.ArgumentParser(
        description="Causal DragonNet Text: Causal inference from clinical text with PSM validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default config
  cdt init --output config.json

  # Run experiment with config
  cdt run --config config.json

  # Run with custom settings
  cdt run --config config.json --device cuda:0 --workers 4

  # Skip PSM analysis (only run DragonNet)
  cdt run --config config.json --skip-psm
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # INIT command
    init_parser = subparsers.add_parser('init', help='Create default configuration file')
    init_parser.add_argument(
        '--output', '-o',
        default='cdt_config.json',
        help='Output path for config file (default: cdt_config.json)'
    )

    # RUN command
    run_parser = subparsers.add_parser('run', help='Run experiment from config')
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
        '--workers',
        type=int,
        help='Override number of workers from config'
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
        '--skip-pretraining',
        action='store_true',
        help='Skip pretraining even if enabled in config'
    )
    run_parser.add_argument(
        '--skip-plasmode',
        action='store_true',
        help='Skip plasmode experiments even if enabled in config'
    )
    run_parser.add_argument(
        '--skip-psm',
        action='store_true',
        help='Skip propensity score matching analysis'
    )
    run_parser.add_argument(
        '--matching-method',
        choices=['nearest', 'optimal', 'caliper'],
        help='Override PSM matching method'
    )
    run_parser.add_argument(
        '--caliper',
        type=float,
        help='Override caliper value for PSM'
    )

    args = parser.parse_args()

    if args.command == 'init':
        create_default_config(args.output)
        print(f"\nEdit {args.output} and then run:")
        print(f"  cdt run --config {args.output}")
        return 0

    elif args.command == 'run':
        level = logging.DEBUG if args.verbose else logging.INFO
        setup_logging(level=level)

        try:
            config = ExperimentConfig.from_json(args.config)
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1

        # Apply overrides
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
        if args.skip_psm:
            config.applied_inference.matching_analysis.enabled = False
        if args.matching_method:
            config.applied_inference.matching_analysis.method = args.matching_method
        if args.caliper is not None:
            config.applied_inference.matching_analysis.caliper = args.caliper

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

            if results.get('plasmode_experiments'):
                print(f"Plasmode experiment results: {results['plasmode_experiments']}")

            return 0

        except Exception as e:
            logging.error(f"Experiment failed: {e}", exc_info=True)
            return 1

    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
