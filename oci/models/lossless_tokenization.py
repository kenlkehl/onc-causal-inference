"""Fail-closed error type for configuration-bounded model tokenization.

``max_length`` and ``max_chunks`` are resource/capacity settings.  They must
never silently select a prefix or suffix of a clinical note.  Tokenizing
callers use ``truncation=False`` and raise :class:`SemanticTruncationError`
when a configured capacity would bind.  The caller can then raise the relevant
configuration value or choose an explicitly lossless windowing architecture.
"""

from __future__ import annotations


class SemanticTruncationError(ValueError):
    """A configured text capacity would discard part of an input."""
