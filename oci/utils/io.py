# oci/utils/io.py

"""I/O utilities for file operations and caching."""

import hashlib
import uuid
from pathlib import Path
from typing import Any
import torch


def hash_text(text: str) -> str:
    """
    Create stable hash for text content.
    
    Args:
        text: Input text
    
    Returns:
        SHA1 hash of normalized text
    """
    if not isinstance(text, str):
        text = str(text)
    normalized = " ".join(text.strip().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def safe_filename(s: str, max_length: int = 100) -> str:
    """
    Convert string to safe filename.
    
    Args:
        s: Input string
        max_length: Maximum filename length
    
    Returns:
        Safe filename string
    """
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in s)
    return safe[:max_length]


def atomic_save(obj: Any, path: Path) -> None:
    """
    Atomically save PyTorch object to avoid partial writes.
    
    Args:
        obj: Object to save
        path: Destination path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_path = path.parent / f"{path.name}.tmp.{uuid.uuid4().hex}"
    
    try:
        torch.save(obj, temp_path)
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
