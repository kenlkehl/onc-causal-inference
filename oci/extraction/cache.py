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
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)
ROW_INDEX_COLUMN = "__oci_cache_row_index"
PATIENT_TEXT_HASH_COLUMN = "__oci_patient_text_hash"


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
                'value_aliases': (
                    c.get('value_aliases')
                    if isinstance(c, dict)
                    else getattr(c, 'value_aliases', None)
                ),
                'roles': c.get('roles') if isinstance(c, dict) else c.roles,
            }
            for c in config.get('features', config.get('confounders', []))
        ],
        'prompt_template_version': config.get('prompt_template_version', ''),
        'vllm_model_name': config.get('vllm_model_name', ''),
        'vllm_endpoint_model_inventory': config.get(
            'vllm_endpoint_model_inventory', {}
        ),
        'vllm_max_model_len': config.get('vllm_max_model_len'),
        'vllm_reasoning_parser': config.get('vllm_reasoning_parser', ''),
        'vllm_reasoning_parser_inventory': config.get(
            'vllm_reasoning_parser_inventory', {}
        ),
        'vllm_enable_thinking': config.get('vllm_enable_thinking'),
        'extraction_temperature': config.get('extraction_temperature', 0.0),
        'extraction_max_tokens': config.get('extraction_max_tokens', 1024),
        'extraction_max_text_length': config.get(
            'extraction_max_text_length',
            config.get('max_text_length'),
        ),
        'extraction_grouping_strategy': config.get(
            'extraction_grouping_strategy', 'clinical_domain'
        ),
        'extraction_grouping_version': config.get('extraction_grouping_version', ''),
        'extraction_request_group_sha256': config.get(
            'extraction_request_group_sha256', ''
        ),
        'extraction_context_strategy': config.get('extraction_context_strategy', 'tail'),
        'colbert': config.get('colbert'),
        'extraction_context_compactor_version': config.get(
            'extraction_context_compactor_version', ''
        ),
        'patient_text_hash': config.get('patient_text_hash', ''),
        'temporal_cutoff': config.get('temporal_cutoff', ''),
        'max_variables_per_extraction_request': config.get(
            'max_variables_per_extraction_request', 10
        ),
        'extraction_batch_size': config.get('extraction_batch_size'),
        'extraction_provider': config.get('extraction_provider', ''),
        'codex_cli_executable': config.get('codex_cli_executable', ''),
        'codex_cli_model_name': config.get('codex_cli_model_name', ''),
        'codex_cli_reasoning_effort': config.get('codex_cli_reasoning_effort', ''),
        'codex_cli_extra_args': config.get('codex_cli_extra_args', []),
        'complete_document': config.get('complete_document', False),
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
        return self._get_cache_path_with_prefix(dataset_path, config, "extraction")

    def _get_row_cache_path(self, dataset_path: str, config: Dict[str, Any]) -> Path:
        """Get partial row-level cache file path for given dataset and config."""
        return self._get_cache_path_with_prefix(dataset_path, config, "extraction_rows")

    def _get_patient_cache_path(
        self, dataset_path: str, config: Dict[str, Any]
    ) -> Path:
        """Get a fold-independent cache path for one exact feature contract."""
        contract_config = dict(config)
        # The individual text hash is stored as a row key below. Excluding the
        # aggregate frame hash lets identical patients/contracts be reused across
        # folds without sharing any label- or fold-adaptive state.
        contract_config["patient_text_hash"] = ""
        return self._get_cache_path_with_prefix(
            dataset_path, contract_config, "extraction_patients"
        )

    def _get_cache_path_with_prefix(
        self,
        dataset_path: str,
        config: Dict[str, Any],
        prefix: str,
    ) -> Path:
        """Get cache file path for given dataset/config and filename prefix."""
        dataset_hash = _compute_dataset_hash(dataset_path)
        config_hash = _compute_config_hash(config)

        if self.cache_dir:
            base_dir = Path(self.cache_dir)
        else:
            base_dir = Path(dataset_path).parent

        cache_dir = base_dir / ".oci_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir / f"{prefix}_{dataset_hash}_{config_hash}.parquet"

    def _expected_columns(self, config: Dict[str, Any]) -> List[str]:
        """Return value/missing columns expected for this extraction config."""
        features = config.get('features', config.get('confounders', []))
        expected_cols = []
        for c in features:
            name = c.get('name') if isinstance(c, dict) else c.name
            expected_cols.append(f"explicit_feat_{name}")
            expected_cols.append(f"explicit_feat_{name}_missing")
        return expected_cols

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
            expected_cols = self._expected_columns(config)
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

    def load_rows_if_valid(
        self,
        dataset_path: str,
        config: Dict[str, Any],
        expected_rows: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Load partial row-level extraction cache if structurally valid.

        Row caches contain only processed row positions. A row with
        ``*_missing=True`` is still considered processed because the extractor
        completed and determined the value was unavailable.
        """
        cache_path = self._get_row_cache_path(dataset_path, config)
        if not cache_path.exists():
            logger.info(f"Row cache miss: {cache_path} does not exist")
            return None

        try:
            cached_df = pd.read_parquet(cache_path)
            if ROW_INDEX_COLUMN not in cached_df.columns:
                logger.warning(
                    f"Row cache missing {ROW_INDEX_COLUMN}. Invalidating cache."
                )
                return None

            expected_cols = self._expected_columns(config)
            missing_cols = set(expected_cols) - set(cached_df.columns)
            if missing_cols:
                logger.warning(f"Row cache missing columns: {missing_cols}. Invalidating cache.")
                return None

            row_indices = pd.to_numeric(cached_df[ROW_INDEX_COLUMN], errors="coerce")
            if row_indices.isna().any():
                logger.warning("Row cache has non-numeric row indices. Invalidating cache.")
                return None
            row_indices = row_indices.astype(int)
            if expected_rows is not None:
                out_of_range = (row_indices < 0) | (row_indices >= expected_rows)
                if out_of_range.any():
                    logger.warning(
                        "Row cache has row indices outside expected range. Invalidating cache."
                    )
                    return None

            cached_df = cached_df.copy()
            cached_df[ROW_INDEX_COLUMN] = row_indices.values
            cached_df = (
                cached_df.drop_duplicates(subset=[ROW_INDEX_COLUMN], keep="last")
                .sort_values(ROW_INDEX_COLUMN)
                .reset_index(drop=True)
            )
            logger.info(
                f"Loaded row cache from: {cache_path} "
                f"({len(cached_df)} processed rows)"
            )
            return cached_df[[ROW_INDEX_COLUMN, *expected_cols]]
        except Exception as e:
            logger.warning(f"Error loading row cache: {e}. Invalidating cache.")
            return None

    def save_rows(
        self,
        dataset_path: str,
        config: Dict[str, Any],
        row_indices: Sequence[int],
        extracted_df: pd.DataFrame,
    ) -> Path:
        """Merge extracted rows into a partial row-level cache.

        Existing rows are kept unless the same row index is saved again, in
        which case the newer extraction wins.
        """
        row_indices = [int(idx) for idx in row_indices]
        if len(row_indices) != len(extracted_df):
            raise ValueError(
                "row_indices length must match extracted_df rows: "
                f"{len(row_indices)} != {len(extracted_df)}"
            )

        expected_cols = self._expected_columns(config)
        missing_cols = set(expected_cols) - set(extracted_df.columns)
        if missing_cols:
            raise ValueError(f"extracted_df missing expected columns: {sorted(missing_cols)}")

        cache_path = self._get_row_cache_path(dataset_path, config)
        row_df = extracted_df[expected_cols].copy()
        row_df.insert(0, ROW_INDEX_COLUMN, row_indices)

        existing_df = self.load_rows_if_valid(dataset_path, config)
        if existing_df is not None:
            row_df = pd.concat([existing_df, row_df], ignore_index=True)
        row_df = (
            row_df.drop_duplicates(subset=[ROW_INDEX_COLUMN], keep="last")
            .sort_values(ROW_INDEX_COLUMN)
            .reset_index(drop=True)
        )

        temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        row_df.to_parquet(temp_path, index=False)
        temp_path.replace(cache_path)
        logger.info(
            f"Saved extraction row cache to: {cache_path} "
            f"({len(row_df)} processed rows)"
        )
        return cache_path

    def load_patient_values(
        self,
        dataset_path: str,
        config: Dict[str, Any],
        patient_text_hashes: Sequence[str],
    ) -> Optional[pd.DataFrame]:
        """Load exact patient/text + feature-contract cache hits positionally."""
        cache_path = self._get_patient_cache_path(dataset_path, config)
        if not cache_path.exists():
            return None
        expected_cols = self._expected_columns(config)
        try:
            cached = pd.read_parquet(cache_path)
            required = {PATIENT_TEXT_HASH_COLUMN, *expected_cols}
            if not required.issubset(cached.columns):
                logger.warning(
                    "Patient extraction cache missing columns %s; invalidating read",
                    sorted(required - set(cached.columns)),
                )
                return None
            cached = cached.drop_duplicates(
                subset=[PATIENT_TEXT_HASH_COLUMN], keep="last"
            ).set_index(PATIENT_TEXT_HASH_COLUMN)
            rows: List[Dict[str, Any]] = []
            for position, text_hash in enumerate(patient_text_hashes):
                if str(text_hash) not in cached.index:
                    continue
                value = cached.loc[str(text_hash)]
                if isinstance(value, pd.DataFrame):
                    value = value.iloc[-1]
                rows.append(
                    {
                        ROW_INDEX_COLUMN: int(position),
                        **{column: value[column] for column in expected_cols},
                    }
                )
            if not rows:
                return None
            return pd.DataFrame(rows, columns=[ROW_INDEX_COLUMN, *expected_cols])
        except Exception as exc:
            logger.warning("Error loading patient extraction cache: %s", exc)
            return None

    def save_patient_values(
        self,
        dataset_path: str,
        config: Dict[str, Any],
        patient_text_hashes: Sequence[str],
        extracted_df: pd.DataFrame,
    ) -> Path:
        """Atomically merge label-free exact patient/contract extraction values."""
        hashes = [str(value) for value in patient_text_hashes]
        if len(hashes) != len(extracted_df):
            raise ValueError(
                "patient_text_hashes length must match extracted_df rows: "
                f"{len(hashes)} != {len(extracted_df)}"
            )
        expected_cols = self._expected_columns(config)
        missing_cols = set(expected_cols) - set(extracted_df.columns)
        if missing_cols:
            raise ValueError(
                f"extracted_df missing expected columns: {sorted(missing_cols)}"
            )
        cache_path = self._get_patient_cache_path(dataset_path, config)
        values = extracted_df[expected_cols].copy().reset_index(drop=True)
        values.insert(0, PATIENT_TEXT_HASH_COLUMN, hashes)
        if cache_path.exists():
            try:
                existing = pd.read_parquet(cache_path)
                if {PATIENT_TEXT_HASH_COLUMN, *expected_cols}.issubset(
                    existing.columns
                ):
                    values = pd.concat(
                        [existing[[PATIENT_TEXT_HASH_COLUMN, *expected_cols]], values],
                        ignore_index=True,
                    )
            except Exception as exc:
                logger.warning(
                    "Ignoring unreadable patient extraction cache during merge: %s",
                    exc,
                )
        values = values.drop_duplicates(
            subset=[PATIENT_TEXT_HASH_COLUMN], keep="last"
        ).reset_index(drop=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        values.to_parquet(temporary, index=False)
        temporary.replace(cache_path)
        return cache_path

    def rows_to_complete_dataframe(
        self,
        rows_df: Optional[pd.DataFrame],
        expected_rows: int,
    ) -> Optional[pd.DataFrame]:
        """Convert a row cache into a complete positional DataFrame if possible."""
        if rows_df is None:
            return None
        unique_rows = rows_df.drop_duplicates(subset=[ROW_INDEX_COLUMN], keep="last")
        if len(unique_rows) != expected_rows:
            return None
        row_indices = set(unique_rows[ROW_INDEX_COLUMN].astype(int).tolist())
        if row_indices != set(range(expected_rows)):
            return None
        return (
            unique_rows.sort_values(ROW_INDEX_COLUMN)
            .drop(columns=[ROW_INDEX_COLUMN])
            .reset_index(drop=True)
        )

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
