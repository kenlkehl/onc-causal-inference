#!/usr/bin/env python
"""Meta-script to test effect of new training options on tau learning.

This script runs ablation experiments to evaluate the impact of:
1. stop_grad_propensity: Detach features before propensity loss
2. attention_entropy_weight: Regularize attention to be more focused
3. use_mean_pooling: Mean pooling instead of [CLS] token

It tests these options on both:
- Sentence-level Gated MIL (run_gated_mil_hierarchical_experiment.py)
- Token-level Gated MIL (run_gated_mil_token_level_experiment_rlearner.py)

Usage:
    python oracle_experiment_scripts/run_new_options_ablation.py

Or with custom settings:
    python oracle_experiment_scripts/run_new_options_ablation.py \
        --device cuda:1 \
        --epochs 50 \
        --dry-run
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_DATASET = "/data1/ken/pcori_dev/pcori_experiments/explicit_confounder_experiments_1-19-26/dataset_with_extraction.parquet"
DEFAULT_OUTPUT_BASE = "/data1/ken/pcori_dev/pcori_experiments/explicit_confounder_experiments_1-19-26/new_options_ablation"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_EPOCHS = 25
DEFAULT_N_FOLDS = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_GAMMA_RLEARNER = 1.0

# Scripts to run (relative to oracle_experiment_scripts/)
SCRIPTS = {
    "sentence_level": "run_gated_mil_hierarchical_experiment.py",
    "token_level": "run_gated_mil_token_level_experiment_rlearner.py",
}

# Condition to run (most realistic scenario with natural clinical text)
# Note: sentence-level uses "2_clinical_text", token-level uses "2_clinical_text_token_level"
CONDITIONS = {
    "sentence_level": "2_clinical_text",
    "token_level": "2_clinical_text_token_level",
}

# Experiments to run with different option combinations
EXPERIMENTS = [
    {
        "name": "baseline",
        "stop_grad": False,
        "entropy_weight": 0.0,
        "mean_pool": False,
        "description": "Baseline: no new options enabled"
    },
    {
        "name": "stop_grad_only",
        "stop_grad": True,
        "entropy_weight": 0.0,
        "mean_pool": False,
        "description": "Stop gradient from propensity to feature extractor"
    },
    {
        "name": "entropy_0.1",
        "stop_grad": False,
        "entropy_weight": 0.1,
        "mean_pool": False,
        "description": "Attention entropy regularization (weight=0.1)"
    },
    {
        "name": "mean_pool_only",
        "stop_grad": False,
        "entropy_weight": 0.0,
        "mean_pool": True,
        "description": "Mean pooling instead of [CLS] token"
    },
    {
        "name": "stop_grad_entropy",
        "stop_grad": True,
        "entropy_weight": 0.1,
        "mean_pool": False,
        "description": "Stop gradient + entropy regularization"
    },
    {
        "name": "all_options",
        "stop_grad": True,
        "entropy_weight": 0.1,
        "mean_pool": True,
        "description": "All new options enabled together"
    },
]


def run_experiment(
    script_path: Path,
    exp_config: Dict[str, Any],
    condition: str,
    output_dir: Path,
    dataset: str,
    device: str,
    epochs: int,
    n_folds: int,
    batch_size: int,
    gamma_rlearner: float,
    dry_run: bool = False
) -> bool:
    """Run a single experiment and return success status."""
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset", dataset,
        "--output-dir", str(output_dir),
        "--device", device,
        "--epochs", str(epochs),
        "--n-folds", str(n_folds),
        "--batch-size", str(batch_size),
        "--gamma-rlearner", str(gamma_rlearner),
        "--conditions", condition,
        "--save-attention",
        "--skip-llm-condition",
    ]

    if exp_config["stop_grad"]:
        cmd.append("--stop-grad-propensity")
    if exp_config["entropy_weight"] > 0:
        cmd.extend(["--attention-entropy-weight", str(exp_config["entropy_weight"])])
    if exp_config["mean_pool"]:
        cmd.append("--use-mean-pooling")

    logger.info(f"Running command: {' '.join(cmd)}")

    if dry_run:
        logger.info("[DRY RUN] Would execute above command")
        return True

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Let output flow to terminal
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment failed with return code {e.returncode}")
        return False


def collect_metrics(output_dir: Path, script_name: str, condition: str) -> Dict[str, Any]:
    """Collect metrics from a completed experiment."""
    metrics_file = output_dir / "metrics_summary.csv"

    if not metrics_file.exists():
        logger.warning(f"Metrics file not found: {metrics_file}")
        return {}

    try:
        df = pd.read_csv(metrics_file, index_col=0)
        # Find the row for our condition
        if condition in df.index:
            return df.loc[condition].to_dict()
        else:
            # Check for partial match (token_level suffix)
            for idx in df.index:
                if condition in idx or idx in condition:
                    return df.loc[idx].to_dict()
            logger.warning(f"Condition {condition} not found in metrics. Available: {list(df.index)}")
            return {}
    except Exception as e:
        logger.error(f"Failed to read metrics: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study on new training options"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Path to dataset parquet file"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=DEFAULT_OUTPUT_BASE,
        help="Base output directory for all experiments"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device to use (cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs per experiment"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size"
    )
    parser.add_argument(
        "--gamma-rlearner",
        type=float,
        default=DEFAULT_GAMMA_RLEARNER,
        help="Weight for R-learner loss"
    )
    parser.add_argument(
        "--scripts",
        type=str,
        nargs="+",
        choices=list(SCRIPTS.keys()),
        default=list(SCRIPTS.keys()),
        help="Which scripts to run (sentence_level, token_level, or both)"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to run (by name). Default: run all."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )

    args = parser.parse_args()

    # Setup paths
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent

    # Validate dataset exists
    if not Path(args.dataset).exists():
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    # Filter experiments if specified
    experiments = EXPERIMENTS
    if args.experiments:
        experiments = [e for e in EXPERIMENTS if e["name"] in args.experiments]
        if not experiments:
            logger.error(f"No matching experiments found. Available: {[e['name'] for e in EXPERIMENTS]}")
            sys.exit(1)

    # Log configuration
    logger.info("=" * 80)
    logger.info("NEW OPTIONS ABLATION STUDY")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output base: {output_base}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"N folds: {args.n_folds}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gamma R-learner: {args.gamma_rlearner}")
    logger.info(f"Scripts to run: {args.scripts}")
    logger.info(f"Experiments: {[e['name'] for e in experiments]}")
    logger.info(f"Total runs: {len(args.scripts) * len(experiments)}")
    logger.info("=" * 80)

    # Track results
    all_results = []
    failed_experiments = []

    # Run all experiments
    start_time = datetime.now()

    for script_name in args.scripts:
        script_file = SCRIPTS[script_name]
        script_path = script_dir / script_file
        condition = CONDITIONS[script_name]

        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            continue

        for exp in experiments:
            exp_name = f"{script_name}_{exp['name']}"
            output_dir = output_base / exp_name

            logger.info(f"\n{'='*60}")
            logger.info(f"EXPERIMENT: {exp_name}")
            logger.info(f"Description: {exp['description']}")
            logger.info(f"Options: stop_grad={exp['stop_grad']}, entropy={exp['entropy_weight']}, mean_pool={exp['mean_pool']}")
            logger.info(f"{'='*60}")

            success = run_experiment(
                script_path=script_path,
                exp_config=exp,
                condition=condition,
                output_dir=output_dir,
                dataset=args.dataset,
                device=args.device,
                epochs=args.epochs,
                n_folds=args.n_folds,
                batch_size=args.batch_size,
                gamma_rlearner=args.gamma_rlearner,
                dry_run=args.dry_run
            )

            if success and not args.dry_run:
                # Collect metrics
                metrics = collect_metrics(output_dir, script_name, condition)
                if metrics:
                    result = {
                        "experiment": exp_name,
                        "script": script_name,
                        "condition": condition,
                        "stop_grad_propensity": exp["stop_grad"],
                        "attention_entropy_weight": exp["entropy_weight"],
                        "use_mean_pooling": exp["mean_pool"],
                        "description": exp["description"],
                        **metrics
                    }
                    all_results.append(result)
                    logger.info(f"Collected metrics for {exp_name}")
            elif not success:
                failed_experiments.append(exp_name)

    # Save combined results
    if all_results and not args.dry_run:
        results_df = pd.DataFrame(all_results)
        results_path = output_base / "combined_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"\nSaved combined results to: {results_path}")

        # Print summary table
        logger.info("\n" + "=" * 80)
        logger.info("ABLATION STUDY RESULTS SUMMARY")
        logger.info("=" * 80)

        # Key metrics to display
        key_cols = [
            "experiment", "stop_grad_propensity", "attention_entropy_weight",
            "use_mean_pooling", "ite_mse", "ite_corr", "ite_spearman_corr", "ate_bias", "propensity_auroc"
        ]
        display_cols = [c for c in key_cols if c in results_df.columns]
        logger.info("\n" + results_df[display_cols].to_string(index=False))

        # Detailed comparison by script
        for script_name in args.scripts:
            script_results = results_df[results_df["script"] == script_name]
            if not script_results.empty:
                logger.info(f"\n--- {script_name.upper()} ---")
                if "ite_mse" in script_results.columns:
                    best_ite_mse = script_results.loc[script_results["ite_mse"].idxmin()]
                    logger.info(f"Best ITE MSE: {best_ite_mse['experiment']} ({best_ite_mse['ite_mse']:.4f})")
                if "ite_corr" in script_results.columns:
                    best_ite_corr = script_results.loc[script_results["ite_corr"].idxmax()]
                    logger.info(f"Best ITE Corr: {best_ite_corr['experiment']} ({best_ite_corr['ite_corr']:.4f})")

    # Save experiment configuration
    config_info = {
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "dataset": args.dataset,
        "output_base": str(output_base),
        "device": args.device,
        "epochs": args.epochs,
        "n_folds": args.n_folds,
        "batch_size": args.batch_size,
        "gamma_rlearner": args.gamma_rlearner,
        "scripts_run": args.scripts,
        "experiments_run": [e["name"] for e in experiments],
        "failed_experiments": failed_experiments,
        "experiment_configs": experiments
    }
    if not args.dry_run:
        config_path = output_base / "ablation_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_info, f, indent=2)
        logger.info(f"\nSaved configuration to: {config_path}")

    # Summary
    elapsed = datetime.now() - start_time
    logger.info(f"\n{'='*80}")
    logger.info("ABLATION STUDY COMPLETE")
    logger.info(f"Total experiments: {len(args.scripts) * len(experiments)}")
    logger.info(f"Successful: {len(all_results)}")
    logger.info(f"Failed: {len(failed_experiments)}")
    logger.info(f"Elapsed time: {elapsed}")
    logger.info(f"Results saved to: {output_base}")
    logger.info("=" * 80)

    if failed_experiments:
        logger.warning(f"Failed experiments: {failed_experiments}")
        sys.exit(1)


if __name__ == "__main__":
    main()
