# oci/experiments/runner.py

"""Main experiment runner that orchestrates OCI workflows."""

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
    """Orchestrates OCI experiments including applied inference."""

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
            # gpu_ids only apply to CUDA devices
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{config.gpu_ids[0]}")
            else:
                self.device = get_device("cuda:0")  # Will fall back to CPU
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

        Returns:
            Dictionary with results paths and summaries
        """
        results = {}

        self._save_config()

        logger.info("\n" + "=" * 80)
        logger.info("APPLIED INFERENCE")
        logger.info("=" * 80)
        applied_results = self._run_applied_inference()
        results['applied_inference'] = applied_results

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
            save_confounder_interpretations=self.config.save_confounder_interpretations,
            confounder_interpretation_top_k=self.config.confounder_interpretation_top_k
        )

        logger.info(f"Applied inference complete: {predictions_path}")
        return str(predictions_path)

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
