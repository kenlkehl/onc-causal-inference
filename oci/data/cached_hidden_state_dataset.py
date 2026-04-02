# oci/data/cached_hidden_state_dataset.py
"""Dataset and collator for cached hidden state mode.

When using pre-computed LLM hidden states, each sample includes either:
- A cache_index for deferred loading (legacy path), or
- The actual hidden states loaded directly by the DataLoader workers (optimized path)

The optimized path moves I/O into DataLoader workers, overlapping disk reads with
GPU compute and enabling prefetching. Hidden states are cast to float32 in the
collate function to avoid dtype mismatches with trainable model parameters.
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

logger = logging.getLogger(__name__)


class CachedHiddenStateDataset(Dataset):
    """Dataset for pre-computed hidden state lookup.

    When cache arrays are provided, hidden states are loaded directly in
    __getitem__() so DataLoader workers handle I/O in parallel. When not
    provided, falls back to returning cache_index for deferred loading.

    Args:
        data: DataFrame with text, outcomes, and treatments.
        text_column: Name of text column.
        outcome_column: Name of outcome column.
        treatment_column: Name of treatment column.
        dataset_indices: Array mapping local position -> global cache index.
        explicit_confounder_columns: Optional list of explicit confounder column names.
        cache_hidden_states: Optional numpy array of hidden states (N, seq_len, hidden_size).
            When provided, __getitem__ returns hidden states directly.
        cache_attention_masks: Optional numpy array of attention masks (N, seq_len).
            Required when cache_hidden_states is provided.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str,
        outcome_column: str,
        treatment_column: str,
        dataset_indices: np.ndarray,
        explicit_confounder_columns: Optional[List[str]] = None,
        cache_hidden_states: Optional[np.ndarray] = None,
        cache_attention_masks: Optional[np.ndarray] = None,
        cache_chunk_counts: Optional[List[int]] = None,
    ):
        self.data = data.reset_index(drop=True)
        self.text_column = text_column
        self.dataset_indices = dataset_indices

        # Store cache arrays for direct loading in __getitem__
        self._cache_hs = cache_hidden_states
        self._cache_mask = cache_attention_masks
        self._chunk_counts = cache_chunk_counts

        self.texts = data[text_column].tolist()
        self.outcomes = torch.tensor(
            data[outcome_column].values,
            dtype=torch.float32
        )
        self.treatments = torch.tensor(
            data[treatment_column].values,
            dtype=torch.float32
        )

        # Extract explicit confounder values if columns provided
        self.explicit_confounder_columns = explicit_confounder_columns or []
        self.explicit_confounder_values = None
        if self.explicit_confounder_columns:
            self.explicit_confounder_values = []
            for idx in range(len(data)):
                row_values = {}
                for col in self.explicit_confounder_columns:
                    if col.startswith("explicit_conf_"):
                        spec_name = col[len("explicit_conf_"):]
                    else:
                        spec_name = col
                    row_values[spec_name] = data[col].iloc[idx]
                    missing_col = f"{col}_missing"
                    if missing_col in data.columns:
                        row_values[f"{spec_name}_missing"] = data[missing_col].iloc[idx]
                    else:
                        row_values[f"{spec_name}_missing"] = pd.isna(row_values[spec_name])
                self.explicit_confounder_values.append(row_values)

        mode = "inline loading" if self._cache_hs is not None else "deferred (cache_index)"
        logger.info(
            f"CachedHiddenStateDataset created: {len(self)} samples, "
            f"cache indices range [{dataset_indices.min()}, {dataset_indices.max()}], "
            f"mode={mode}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            'text': self.texts[idx],
            'outcome': self.outcomes[idx],
            'treatment': self.treatments[idx],
            'text_id': idx,
        }

        global_idx = self.dataset_indices[idx]

        if self._cache_hs is not None:
            # Optimized path: load hidden states directly (DataLoader workers handle I/O)
            item['hidden_states'] = self._cache_hs[global_idx]    # float16 numpy
            item['attention_mask'] = self._cache_mask[global_idx]  # uint8 numpy
        else:
            # Legacy path: return index for deferred loading in training loop
            item['cache_index'] = int(global_idx)

        if self._chunk_counts is not None:
            global_idx_for_chunks = self.dataset_indices[idx]
            item['chunk_count'] = self._chunk_counts[global_idx_for_chunks]

        if self.explicit_confounder_values is not None:
            item['explicit_confounder_values'] = self.explicit_confounder_values[idx]

        return item


def collate_cached_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch for cached hidden state dataset.

    When items contain 'hidden_states' (optimized path), stacks them into
    float32 tensors for GPU transfer. When items contain 'cache_index'
    (legacy path), collects indices for deferred loading.

    Args:
        batch: List of samples from CachedHiddenStateDataset.

    Returns:
        Batched data with texts, tensors, and either cached_hidden_states or cache_indices.
    """
    texts = [item['text'] for item in batch]
    outcomes = torch.stack([item['outcome'] for item in batch])
    treatments = torch.stack([item['treatment'] for item in batch])
    text_ids = [item['text_id'] for item in batch]

    result = {
        'texts': texts,
        'outcome': outcomes,
        'treatment': treatments,
        'text_id': text_ids,
    }

    if 'hidden_states' in batch[0]:
        # Optimized path: pad variable-length hidden states to batch-local max
        lengths = [item['hidden_states'].shape[0] for item in batch]
        max_len = max(lengths)
        hidden_size = batch[0]['hidden_states'].shape[-1]
        hs = np.zeros((len(batch), max_len, hidden_size), dtype=np.float32)
        mask = np.zeros((len(batch), max_len), dtype=np.float32)
        for i, item in enumerate(batch):
            l = lengths[i]
            hs[i, :l] = np.asarray(item['hidden_states'], dtype=np.float32)
            mask[i, :l] = 1.0
        result['cached_hidden_states'] = torch.from_numpy(hs).float()  # enforce float32
        result['cached_attention_mask'] = torch.from_numpy(mask)
    elif 'cache_index' in batch[0]:
        # Legacy path: collect indices for deferred loading
        result['cache_indices'] = [item['cache_index'] for item in batch]

    if 'chunk_count' in batch[0]:
        result['sample_chunk_counts'] = [item['chunk_count'] for item in batch]

    if 'explicit_confounder_values' in batch[0]:
        result['explicit_confounder_values'] = [
            item['explicit_confounder_values'] for item in batch
        ]

    return result


def prepare_cached_batch(
    batch: Dict[str, Any],
    device: torch.device,
    hidden_state_cache=None,
    gpu_store=None,
) -> None:
    """Move cached hidden states to device, loading from cache if needed.

    Supports three paths:
    - Optimized: hidden states already in batch (loaded by DataLoader workers).
      Moves float32 tensors to GPU.
    - GPU store: batch has cache_indices, loads from GPU-resident store (zero-copy).
    - Legacy: batch has cache_indices, loads from disk-backed HiddenStateCache.

    Args:
        batch: Batch dict from DataLoader.
        device: Target device.
        hidden_state_cache: Optional HiddenStateCache for disk-based fallback.
        gpu_store: Optional GPUHiddenStateStore for GPU-resident hidden states.
    """
    if 'cached_hidden_states' in batch:
        # Optimized path: loaded by DataLoader, transfer to GPU as float32
        batch['cached_hidden_states'] = batch['cached_hidden_states'].to(device).float()
        batch['cached_attention_mask'] = batch['cached_attention_mask'].to(device)
    elif gpu_store is not None and 'cache_indices' in batch:
        # GPU store path: hidden states already on GPU, zero-copy gather+pad
        hs, mask = gpu_store.load_batch(batch['cache_indices'])
        batch['cached_hidden_states'] = hs
        batch['cached_attention_mask'] = mask
    elif 'cache_indices' in batch and hidden_state_cache is not None:
        # Legacy fallback: load from disk-backed cache
        hs, mask = hidden_state_cache.load_batch(batch['cache_indices'], device)
        batch['cached_hidden_states'] = hs
        batch['cached_attention_mask'] = mask
