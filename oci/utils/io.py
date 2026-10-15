# oci/utils/io.py

"""I/O utilities for file operations."""

from pathlib import Path


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
