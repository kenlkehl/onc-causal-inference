# oci/data/dataset.py
"""Dataset classes for OCI causal inference."""

import logging
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
import pandas as pd


logger = logging.getLogger(__name__)


class ClinicalTextDataset(Dataset):
    """
    Dataset that returns raw text for CNN processing.

    Returns raw text strings that are tokenized by the model during forward pass.
    This is memory-efficient and allows end-to-end training.

    Optionally includes explicit confounder columns if they are present in the data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str,
        outcome_column: str,
        treatment_column: str,
        explicit_confounder_columns: Optional[List[str]] = None
    ):
        """
        Initialize dataset.

        Args:
            data: DataFrame with text, outcomes, and treatments
            text_column: Name of text column
            outcome_column: Name of outcome column
            treatment_column: Name of treatment column
            explicit_confounder_columns: Optional list of explicit confounder column names
                (e.g., ["explicit_conf_performance_status", "explicit_conf_age_at_diagnosis"]).
                If provided, corresponding "_missing" columns are also read.
        """
        self.data = data.reset_index(drop=True)
        self.text_column = text_column
        self.explicit_confounder_columns = explicit_confounder_columns or []

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
        # The featurizer expects keys to match spec.name (e.g., "age"),
        # not the column name (e.g., "explicit_conf_age").
        # Strip the "explicit_conf_" prefix when building value dicts.
        self.explicit_confounder_values = None
        if self.explicit_confounder_columns:
            self.explicit_confounder_values = []
            for idx in range(len(data)):
                row_values = {}
                for col in self.explicit_confounder_columns:
                    # Extract spec name by stripping prefix
                    if col.startswith("explicit_conf_"):
                        spec_name = col[len("explicit_conf_"):]
                    else:
                        spec_name = col
                    # Get value using spec_name as key
                    row_values[spec_name] = data[col].iloc[idx]
                    # Get missing flag (look for corresponding "_missing" column)
                    missing_col = f"{col}_missing"
                    if missing_col in data.columns:
                        row_values[f"{spec_name}_missing"] = data[missing_col].iloc[idx]
                    else:
                        # Infer missing from value
                        row_values[f"{spec_name}_missing"] = pd.isna(row_values[spec_name])
                self.explicit_confounder_values.append(row_values)
            logger.info(f"Loaded {len(self.explicit_confounder_columns)} explicit confounder columns")

        logger.info(f"ClinicalTextDataset created: {len(self)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            'text': self.texts[idx],
            'outcome': self.outcomes[idx],
            'treatment': self.treatments[idx],
            'text_id': idx
        }

        if self.explicit_confounder_values is not None:
            item['explicit_confounder_values'] = self.explicit_confounder_values[idx]

        return item


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate batch for CNN dataset.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched data with texts as list of strings
    """
    texts = [item['text'] for item in batch]
    outcomes = torch.stack([item['outcome'] for item in batch])
    treatments = torch.stack([item['treatment'] for item in batch])
    text_ids = [item['text_id'] for item in batch]

    result = {
        'texts': texts,
        'outcome': outcomes,
        'treatment': treatments,
        'text_id': text_ids
    }

    # Include explicit confounder values if present
    if 'explicit_confounder_values' in batch[0]:
        result['explicit_confounder_values'] = [
            item['explicit_confounder_values'] for item in batch
        ]

    return result


def load_dataset(
    path: str,
    split: Optional[str] = None,
    split_column: str = 'split'
) -> pd.DataFrame:
    """
    Load dataset from file.

    Args:
        path: Path to dataset file (.csv or .parquet)
        split: Optional split to filter (e.g., 'train', 'val', 'test')
        split_column: Name of split column

    Returns:
        DataFrame
    """
    logger.info(f"Loading dataset from {path}")

    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if split is not None:
        if split_column not in df.columns:
            raise ValueError(f"Split column '{split_column}' not found")
        df = df[df[split_column] == split].copy()
        logger.info(f"Filtered to {split} split: {len(df)} samples")

    return df


def validate_dataset(
    df: pd.DataFrame,
    text_column: str,
    outcome_column: str,
    treatment_column: str,
    split_column: Optional[str] = None
) -> None:
    """
    Validate dataset has required columns and correct format.

    Args:
        df: DataFrame to validate
        text_column: Expected text column name
        outcome_column: Expected outcome column name
        treatment_column: Expected treatment column name
        split_column: Optional split column name

    Raises:
        ValueError: If validation fails
    """
    required = {text_column, outcome_column, treatment_column}
    if split_column:
        required.add(split_column)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[text_column].isnull().any():
        raise ValueError(f"Null values in {text_column}")

    if df[outcome_column].isnull().any():
        raise ValueError(f"Null values in {outcome_column}")

    if df[treatment_column].isnull().any():
        raise ValueError(f"Null values in {treatment_column}")

    logger.info(f"Dataset validation passed: {len(df)} samples")
