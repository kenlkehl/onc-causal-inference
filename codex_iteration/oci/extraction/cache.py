# oci/extraction/cache.py
"""Caching utilities for LLM-based explicit feature extraction results.

The cache helps avoid redundant LLM calls by storing extraction results
keyed by (dataset path hash + extraction config hash). Cache files are
stored as Parquet files alongside the dataset.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a deterministic hash of extraction configuration.

    Includes the full extraction contract: feature specs, prompt-relevant
    descriptions, model settings, and prompt/text truncation settings.
    """
    # Extract relevant fields for hashing
    hash_dict = {
        'features': [
            {
                'name': c.get('name') if isinstance(c, dict) else c.name,
                'type': c.get('type') if isinstance(c, dict) else c.type,
                'categories': c.get('categories') if isinstance(c, dict) else c.categories,
                'description': c.get('description') if isinstance(c, dict) else c.description,
                'roles': c.get('roles') if isinstance(c, dict) else c.roles,
            }
            for c in config.get('features', config.get('confounders', []))
        ],
        'prompt_template_version': config.get('prompt_template_version', ''),
        'vllm_model_name': config.get('vllm_model_name', ''),
        'extraction_temperature': config.get('extraction_temperature', 0.0),
        'extraction_max_tokens': config.get('extraction_max_tokens', 1024),
        'extraction_max_text_length': config.get(
            'extraction_max_text_length',
            config.get('max_text_length', 8000),
        ),
    }
    config_str = json.dumps(hash_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def _compute_dataset_hash(dataset_path: str) -> str:
    """Compute hash of dataset path for cache key."""
    # Use path as cache key (not content, for performance)
    return hashlib.md5(str(dataset_path).encode()).hexdigest()[:12]


class ExtractionCache:
    """Cache for LLM-based explicit feature extraction results.

    Cache files are stored as:
        {cache_dir}/.oci_cache/extraction_{dataset_hash}_{config_hash}.parquet

    Usage:
        cache = ExtractionCache()
        cached = cache.load_if_valid(dataset_path, config)
        if cached is not None:
            # Use cached extraction results
            df = df.join(cached)
        else:
            # Run extraction
            extracted_df = run_extraction(...)
            cache.save(dataset_path, config, extracted_df)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize cache.

        Args:
            cache_dir: Directory for cache files. If None, uses dataset's parent directory.
        """
        self.cache_dir = cache_dir

    def _get_cache_path(self, dataset_path: str, config: Dict[str, Any]) -> Path:
        """Get cache file path for given dataset and config."""
        dataset_hash = _compute_dataset_hash(dataset_path)
        config_hash = _compute_config_hash(config)

        if self.cache_dir:
            base_dir = Path(self.cache_dir)
        else:
            base_dir = Path(dataset_path).parent

        cache_dir = base_dir / ".oci_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir / f"extraction_{dataset_hash}_{config_hash}.parquet"

    def load_if_valid(
        self,
        dataset_path: str,
        config: Dict[str, Any],
        expected_rows: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Load cached extraction results if valid.

        Args:
            dataset_path: Path to the original dataset
            config: Extraction configuration dict
            expected_rows: Optional expected number of rows for validation

        Returns:
            DataFrame with extracted confounder columns if cache is valid, None otherwise
        """
        cache_path = self._get_cache_path(dataset_path, config)

        if not cache_path.exists():
            logger.info(f"Cache miss: {cache_path} does not exist")
            return None

        try:
            cached_df = pd.read_parquet(cache_path)
            logger.info(f"Loaded cache from: {cache_path} ({len(cached_df)} rows)")

            # Validate row count if provided
            if expected_rows is not None and len(cached_df) != expected_rows:
                logger.warning(
                    f"Cache row count mismatch: expected {expected_rows}, got {len(cached_df)}. "
                    f"Invalidating cache."
                )
                return None

            # Verify expected columns exist
            features = config.get('features', config.get('confounders', []))
            expected_cols = []
            for c in features:
                name = c.get('name') if isinstance(c, dict) else c.name
                expected_cols.append(f"explicit_feat_{name}")
                expected_cols.append(f"explicit_feat_{name}_missing")

            missing_cols = set(expected_cols) - set(cached_df.columns)
            if missing_cols:
                logger.warning(f"Cache missing columns: {missing_cols}. Invalidating cache.")
                return None

            return cached_df

        except Exception as e:
            logger.warning(f"Error loading cache: {e}. Invalidating cache.")
            return None

    def save(
        self,
        dataset_path: str,
        config: Dict[str, Any],
        extracted_df: pd.DataFrame
    ) -> Path:
        """Save extraction results to cache.

        Args:
            dataset_path: Path to the original dataset
            config: Extraction configuration dict
            extracted_df: DataFrame with extracted confounder columns

        Returns:
            Path to saved cache file
        """
        cache_path = self._get_cache_path(dataset_path, config)
        extracted_df.to_parquet(cache_path, index=False)
        logger.info(f"Saved extraction cache to: {cache_path} ({len(extracted_df)} rows)")
        return cache_path

    def invalidate(self, dataset_path: str, config: Dict[str, Any]) -> bool:
        """Invalidate (delete) cached results.

        Args:
            dataset_path: Path to the original dataset
            config: Extraction configuration dict

        Returns:
            True if cache was deleted, False if it didn't exist
        """
        cache_path = self._get_cache_path(dataset_path, config)
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Invalidated cache: {cache_path}")
            return True
        return False
