# cdt/experiments/runner.py

"""Main experiment runner that orchestrates CDT workflows - CNN-based approach."""

import logging
from pathlib import Path
from typing import Dict, Any
import json
import pandas as pd

from ..config import ExperimentConfig
from ..utils import set_seed, ensure_dir, get_device
from ..data import load_dataset, validate_dataset


logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates CDT experiments including applied inference and plasmode."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        ensure_dir(self.output_dir)

        set_seed(config.seed)

        # Determine device
        if config.device:
            self.device = get_device(config.device)
        elif config.gpu_ids:
            import torch
            self.device = torch.device(f"cuda:{config.gpu_ids[0]}")
        else:
            self.device = get_device("cuda:0")

        logger.info("Experiment initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Seed: {config.seed}")
        logger.info(f"Workers: {config.num_workers}")

    def run(self) -> Dict[str, Any]:
        """
        Run complete experiment workflow.

        Supports two inference modes:
        1. Matched Pair ITE (if matched_pair is configured and not skipped)
        2. DragonNet/CausalText (via applied_inference)

        Returns:
            Dictionary with results paths and summaries
        """
        results = {}

        self._save_config()

        # Check if matched_pair is configured and should be used
        use_matched_pair = (
            self.config.matched_pair is not None
            and not getattr(self.config.matched_pair, 'skip', False)
        )

        if use_matched_pair:
            # Use matched pair pipeline
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 1: MATCHED PAIR ITE ESTIMATION")
            logger.info("=" * 80)
            matched_pair_results = self._run_matched_pair_inference()
            results['matched_pair'] = matched_pair_results
        elif self.config.applied_inference.skip:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 1: APPLIED INFERENCE (SKIPPED)")
            logger.info("=" * 80)
            logger.info("Applied inference skipped via config.applied_inference.skip=True")
        else:
            # Use DragonNet/CausalText pipeline
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 1: APPLIED INFERENCE (DragonNet/CausalText)")
            logger.info("=" * 80)
            applied_results = self._run_applied_inference()
            results['applied_inference'] = applied_results

        if self.config.plasmode_experiments.enabled:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 2: PLASMODE EXPERIMENTS")
            logger.info("=" * 80)
            plasmode_results = self._run_plasmode_experiments()
            results['plasmode_experiments'] = plasmode_results
        else:
            logger.info("Plasmode experiments disabled, skipping")

        self._save_results_summary(results)

        return results

    def _save_config(self) -> None:
        """Save configuration to output directory."""
        config_path = self.output_dir / "config.json"
        self.config.to_json(str(config_path))
        logger.info(f"Configuration saved to: {config_path}")

    def _run_applied_inference(self) -> str:
        """
        Run applied inference on real data.

        Returns:
            Path to predictions file
        """
        from ..inference.applied import run_applied_inference

        applied_config = self.config.applied_inference

        logger.info(f"Loading dataset: {applied_config.dataset_path}")
        df = load_dataset(applied_config.dataset_path)

        # Only validate split_column if using fixed split mode (cv_folds <= 1)
        split_col_to_validate = applied_config.split_column if applied_config.cv_folds <= 1 else None
        validate_dataset(
            df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column,
            split_column=split_col_to_validate
        )

        output_dir = ensure_dir(self.output_dir / "applied_inference")
        predictions_path = output_dir / "predictions.parquet"

        run_applied_inference(
            dataset=df,
            config=applied_config,
            output_path=predictions_path,
            device=self.device,
            gpu_ids=self.config.gpu_ids,
            num_workers=self.config.num_workers,
            save_filter_interpretations=self.config.save_filter_interpretations,
            filter_interpretation_top_k=self.config.filter_interpretation_top_k,
            save_confounder_interpretations=self.config.save_confounder_interpretations,
            confounder_interpretation_top_k=self.config.confounder_interpretation_top_k
        )

        logger.info(f"Applied inference complete: {predictions_path}")
        return str(predictions_path)

    def _run_matched_pair_inference(self) -> str:
        """
        Run matched pair ITE estimation.

        Uses the matched_pair config for all settings including dataset path.

        Returns:
            Path to predictions file
        """
        from ..inference.matched_pair_applied import run_matched_pair_applied_inference
        from ..config import AppliedInferenceConfig

        mp_config = self.config.matched_pair

        # Get dataset path from matched_pair config
        dataset_path = mp_config.dataset_path
        if not dataset_path:
            raise ValueError(
                "matched_pair.dataset_path is required. "
                "Please specify the path to your dataset in the matched_pair section."
            )

        logger.info(f"Loading dataset: {dataset_path}")
        df = load_dataset(dataset_path)

        # Validate dataset
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

    def _run_plasmode_experiments(self) -> str:
        """
        Run plasmode sensitivity experiments.

        Returns:
            Path to plasmode results CSV
        """
        from ..training.plasmode import run_plasmode_experiments

        plasmode_config = self.config.plasmode_experiments
        applied_config = self.config.applied_inference

        logger.info(f"Loading dataset: {applied_config.dataset_path}")
        df = load_dataset(applied_config.dataset_path)

        # Plasmode uses all data without splits
        validate_dataset(
            df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column,
            split_column=None
        )

        output_dir = ensure_dir(self.output_dir / "plasmode_experiments")
        results_path = output_dir / "results.csv"

        run_plasmode_experiments(
            dataset=df,
            applied_config=applied_config,
            plasmode_config=plasmode_config,
            output_path=results_path,
            device=self.device,
            num_repeats=plasmode_config.num_repeats,
            num_workers=self.config.num_workers,
            gpu_ids=self.config.gpu_ids
        )

        logger.info(f"Plasmode experiments complete: {results_path}")
        return str(results_path)

    def _save_results_summary(self, results: Dict[str, Any]) -> None:
        """Save summary of all results."""
        summary_path = self.output_dir / "summary.json"

        summary = {
            'config_hash': self.config.get_hash(),
            'seed': self.config.seed,
            'device': str(self.device),
            'results': results
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to: {summary_path}")
