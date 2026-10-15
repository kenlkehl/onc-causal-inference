# oci/utils/__init__.py

"""Shared I/O, reproducibility, device, and logging utilities.

Package-level names resolve lazily, so importing a light submodule (for
example ``oci.utils.calibration`` inside a Stage 2 worker thread) does not
import torch. A torch import racing with threadpoolctl's shared-library scan in
another thread can deadlock the dynamic loader.
"""

_SYSTEM_NAMES = ("limit_threads", "set_seed", "setup_logging", "get_device")


def __getattr__(name):
    if name in _SYSTEM_NAMES:
        from . import system

        return getattr(system, name)
    if name == "ensure_dir":
        from .io import ensure_dir

        return ensure_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'limit_threads',
    'set_seed',
    'setup_logging',
    'get_device',
    'ensure_dir',
]
