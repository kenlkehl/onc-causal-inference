"""Lazy owner of the Stage 1 HTR trainer for one exact discovery context."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import torch

from ..config import AppliedInferenceConfig
from .htr_trainer import HTRTrainer
from .stage1_checkpointing import Stage1CheckpointStore


class HTREvidenceProvider:
    """Create the HTR trainer only when an HTR lane actually runs."""

    def __init__(
        self,
        *,
        config: AppliedInferenceConfig,
        device: Optional[Any] = None,
        checkpoint_store: Optional[Stage1CheckpointStore] = None,
        num_workers: int = 1,
    ) -> None:
        self.config = config
        self.device = torch.device(device or "cpu")
        self.checkpoint_store = checkpoint_store
        self.num_workers = num_workers
        self._runner: Optional[HTRTrainer] = None

    def _ensure_runner(self, discovery_df: pd.DataFrame) -> HTRTrainer:
        del discovery_df
        if self._runner is None:
            self._runner = HTRTrainer(
                config=self.config, device=self.device, num_workers=self.num_workers
            )
        return self._runner


__all__ = ["HTREvidenceProvider"]
