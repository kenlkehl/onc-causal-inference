# cdt/experiments/matched_pair_runner.py
"""Experiment runner for matched pair ITE estimation.

This module provides:
- run_matched_pair_experiment: Standalone function to run the full pipeline
- MatchedPairExperimentRunner: Class-based runner integrated with ExperimentConfig

The matched pair approach is an alternative to DragonNet/R-Learner for
cases where treatment effect estimation benefits from explicit matching.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import torch
import pandas as pd

from ..config import ExperimentConfig, MatchedPairConfig, AppliedInferenceConfig
from ..data import load_dataset, validate_dataset
from ..utils import set_seed, ensure_dir, get_device
from ..inference.matched_pair_applied import run_matched_pair_applied_inference


logger = logging.getLogger(__name__)


def run_matched_pair_experiment(
    dataset: pd.DataFrame,
    matched_pair_config: MatchedPairConfig,
    output_dir: Path,
    device: torch.device,
    gpu_ids: Optional[list] = None,
    num_workers: int = 1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run full matched pair ITE estimation pipeline.

    This is a convenience function for running the matched pair pipeline
    without the full ExperimentConfig framework.

    Stages:
    1. Train propensity model
    2. Extract representations
    3. Match patients
    4. Train outcome/tau model
    5. Predict ITE for all patients

    Args:
        dataset: DataFrame with text, treatment, outcome columns
        matched_pair_config: Configuration for matched pair estimation
        output_dir: Directory to save results
        device: PyTorch device
        gpu_ids: Optional list of GPU IDs for parallel processing
        num_workers: Number of parallel workers
        seed: Random seed

    Returns:
        DataFrame with ITE predictions for all patients
    """
    set_seed(seed)
    output_dir = ensure_dir(output_dir)

    logger.info("=" * 80)
    logger.info("MATCHED PAIR ITE ESTIMATION EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Samples: {len(dataset)}")

    # Save config
    config_path = output_dir / "matched_pair_config.json"
    with open(config_path, 'w') as f:
        json.dump(matched_pair_config.to_dict(), f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")

    # Create minimal AppliedInferenceConfig for column names
    applied_config = AppliedInferenceConfig(
        text_column=matched_pair_config.text_column,
        outcome_column=matched_pair_config.outcome_column,
        treatment_column=matched_pair_config.treatment_column,
        cv_folds=matched_pair_config.cv_folds
    )

    # Run inference
    predictions_path = output_dir / "predictions.parquet"

    run_matched_pair_applied_inference(
        dataset=dataset,
        config=applied_config,
        matched_pair_config=matched_pair_config,
        output_path=predictions_path,
        device=device,
        gpu_ids=gpu_ids,
        num_workers=num_workers
    )

    # Load and return predictions
    results_df = pd.read_parquet(predictions_path)

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)

    return results_df


class MatchedPairExperimentRunner:
    """
    Experiment runner for matched pair ITE estimation.

    Integrates with ExperimentConfig for consistent experiment management.
    Can be used standalone or as part of the ExperimentRunner workflow.

    Example usage:
        config = ExperimentConfig.from_json("config.json")
        runner = MatchedPairExperimentRunner(config)
        results = runner.run()
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration with matched_pair field
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        ensure_dir(self.output_dir)

        set_seed(config.seed)

        # Determine device
        if config.device:
            self.device = get_device(config.device)
        elif config.gpu_ids:
            self.device = torch.device(f"cuda:{config.gpu_ids[0]}")
        else:
            self.device = get_device("cuda:0")

        logger.info("MatchedPairExperimentRunner initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Seed: {config.seed}")

    def run(self) -> Dict[str, Any]:
        """
        Run the matched pair experiment.

        Returns:
            Dictionary with results paths and summaries
        """
        results = {}

        self._save_config()

        if self.config.matched_pair is None:
            logger.warning("No matched_pair config found, skipping")
            return results

        if self.config.matched_pair.skip:
            logger.info("Matched pair experiment skipped (config.matched_pair.skip=True)")
            return results

        logger.info("\n" + "=" * 80)
        logger.info("MATCHED PAIR ITE ESTIMATION")
        logger.info("=" * 80)

        matched_pair_results = self._run_matched_pair_inference()
        results['matched_pair'] = matched_pair_results

        self._save_results_summary(results)

        return results

    def _save_config(self) -> None:
        """Save configuration to output directory."""
        config_path = self.output_dir / "config.json"
        self.config.to_json(str(config_path))
        logger.info(f"Configuration saved to: {config_path}")

    def _run_matched_pair_inference(self) -> str:
        """
        Run matched pair inference.

        Returns:
            Path to predictions file
        """
        mp_config = self.config.matched_pair

        # Determine dataset path
        dataset_path = mp_config.dataset_path or self.config.applied_inference.dataset_path

        logger.info(f"Loading dataset: {dataset_path}")
        df = load_dataset(dataset_path)

        # Validate dataset
        # Only validate split_column if using fixed split mode
        split_col_to_validate = mp_config.split_column if mp_config.cv_folds <= 1 else None
        validate_dataset(
            df,
            text_column=mp_config.text_column,
            outcome_column=mp_config.outcome_column,
            treatment_column=mp_config.treatment_column,
            split_column=split_col_to_validate
        )

        output_dir = ensure_dir(self.output_dir / "matched_pair")
        predictions_path = output_dir / "predictions.parquet"

        # Create minimal AppliedInferenceConfig for column names
        applied_config = AppliedInferenceConfig(
            text_column=mp_config.text_column,
            outcome_column=mp_config.outcome_column,
            treatment_column=mp_config.treatment_column,
            split_column=mp_config.split_column,
            cv_folds=mp_config.cv_folds
        )

        run_matched_pair_applied_inference(
            dataset=df,
            config=applied_config,
            matched_pair_config=mp_config,
            output_path=predictions_path,
            device=self.device,
            gpu_ids=self.config.gpu_ids,
            num_workers=self.config.num_workers
        )

        logger.info(f"Matched pair inference complete: {predictions_path}")
        return str(predictions_path)

    def _save_results_summary(self, results: Dict[str, Any]) -> None:
        """Save summary of results."""
        summary_path = self.output_dir / "matched_pair_summary.json"

        summary = {
            'config_hash': self.config.get_hash(),
            'seed': self.config.seed,
            'device': str(self.device),
            'results': results
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to: {summary_path}")


def run_matched_pair_from_config(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Convenience function to run matched pair experiment from config.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary with results
    """
    runner = MatchedPairExperimentRunner(config)
    return runner.run()
