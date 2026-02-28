# cdt/data/cached_hidden_state_dataset.py
"""Dataset and collator for cached hidden state mode.

When using pre-computed LLM hidden states, each sample includes a cache_index
that maps to its position in the global hidden state cache. The collator
produces batch dicts with cache_indices that are used to load hidden states
from the cache in the training loop.
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

logger = logging.getLogger(__name__)


class CachedHiddenStateDataset(Dataset):
    """Dataset that adds cache_index for pre-computed hidden state lookup.

    Same interface as ClinicalTextDataset but includes a cache_index per sample
    that maps the local dataset position to the global cache index.

    Args:
        data: DataFrame with text, outcomes, and treatments.
        text_column: Name of text column.
        outcome_column: Name of outcome column.
        treatment_column: Name of treatment column.
        dataset_indices: Array mapping local position -> global cache index.
        explicit_confounder_columns: Optional list of explicit confounder column names.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str,
        outcome_column: str,
        treatment_column: str,
        dataset_indices: np.ndarray,
        explicit_confounder_columns: Optional[List[str]] = None,
    ):
        self.data = data.reset_index(drop=True)
        self.text_column = text_column
        self.dataset_indices = dataset_indices

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

        logger.info(
            f"CachedHiddenStateDataset created: {len(self)} samples, "
            f"cache indices range [{dataset_indices.min()}, {dataset_indices.max()}]"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            'text': self.texts[idx],
            'outcome': self.outcomes[idx],
            'treatment': self.treatments[idx],
            'text_id': idx,
            'cache_index': int(self.dataset_indices[idx]),
        }

        if self.explicit_confounder_values is not None:
            item['explicit_confounder_values'] = self.explicit_confounder_values[idx]

        return item


def collate_cached_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch for cached hidden state dataset.

    Same as collate_batch but includes cache_indices list. Does NOT load
    hidden states (that happens in the training loop to place them on the
    correct device).

    Args:
        batch: List of samples from CachedHiddenStateDataset.

    Returns:
        Batched data with texts, tensors, and cache_indices.
    """
    texts = [item['text'] for item in batch]
    outcomes = torch.stack([item['outcome'] for item in batch])
    treatments = torch.stack([item['treatment'] for item in batch])
    text_ids = [item['text_id'] for item in batch]
    cache_indices = [item['cache_index'] for item in batch]

    result = {
        'texts': texts,
        'outcome': outcomes,
        'treatment': treatments,
        'text_id': text_ids,
        'cache_indices': cache_indices,
    }

    if 'explicit_confounder_values' in batch[0]:
        result['explicit_confounder_values'] = [
            item['explicit_confounder_values'] for item in batch
        ]

    return result
