# oci/data/collators.py
"""Collator utilities for CDT data loading.

The only remaining extractor (frozen_llm_pooler) does not require chunking
collators; it handles tokenization internally. create_collator() is kept as
a public API that always returns None so callers do not need to change.
"""

import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


def create_collator(
    feature_extractor,
) -> Optional[Callable]:
    """Return a collator for the given feature extractor.

    frozen_llm_pooler does not need a chunking collator, so this always
    returns None.  Callers should fall back to ``collate_batch``.

    Args:
        feature_extractor: The model's feature extractor (already initialized)

    Returns:
        None (frozen_llm_pooler handles tokenization internally)
    """
    return None
