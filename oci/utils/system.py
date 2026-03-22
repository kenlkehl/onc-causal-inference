# oci/utils/system.py

"""System utilities for thread management, logging, and resource cleanup."""

import os
import random
import logging
from typing import Optional
import numpy as np
import torch


def limit_threads(n_threads: int = 1):
    """
    Limit threading to prevent resource contention.
    Should be called early before importing heavy libraries.
    
    Args:
        n_threads: Number of threads to use (default: 1)
    """
    n_threads_str = str(n_threads)
    
    thread_vars = [
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]
    
    for var in thread_vars:
        os.environ[var] = n_threads_str
    
    try:
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)
    except Exception:
        pass


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility.

    Note: torch.manual_seed() sets seeds for CPU, CUDA, and MPS devices.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cuda_cleanup():
    """Clean up CUDA memory. Deprecated: use device_cleanup() instead."""
    device_cleanup()


def device_cleanup(device: Optional[torch.device] = None):
    """Clean up accelerator memory (CUDA or MPS).

    Args:
        device: Optional device to synchronize. If None, cleans up all available accelerators.
    """
    try:
        import gc
        import time

        gc.collect()

        if torch.cuda.is_available():
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            else:
                torch.cuda.synchronize()
            for _ in range(2):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
                time.sleep(0.05)

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
            gc.collect()
    except Exception:
        pass


def get_memory_info() -> str:
    """Get memory usage information."""
    try:
        info_parts = []

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            info_parts.append(f"GPU: {allocated:.0f}MB alloc, {reserved:.0f}MB reserved")

        if torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory() / (1024**2)
                info_parts.append(f"MPS: {allocated:.0f}MB alloc")
            except AttributeError:
                # current_allocated_memory may not be available in older PyTorch versions
                info_parts.append("MPS: active")

        try:
            import psutil
            process = psutil.Process(os.getpid())
            rss = process.memory_info().rss / (1024**2)
            info_parts.append(f"RAM: {rss:.0f}MB")
        except ImportError:
            pass

        return " | ".join(info_parts) if info_parts else "Memory info unavailable"
    except Exception:
        return "Memory info unavailable"


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Configure logging for the package.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to write logs to file
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )


def get_device(device_str: str = "cuda:0") -> torch.device:
    """
    Get PyTorch device, falling back to CPU if requested accelerator unavailable.

    Args:
        device_str: Device string (e.g., "cuda:0", "mps", "cpu")

    Returns:
        PyTorch device
    """
    # Handle MPS (Apple Silicon)
    if device_str == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            logging.warning("MPS requested but not available, falling back to CPU")
            return torch.device("cpu")

    # Handle CUDA
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")

    try:
        device = torch.device(device_str)
        if device.type == "cuda":
            gpu_id = device.index or 0
            if gpu_id >= torch.cuda.device_count():
                logging.warning(f"GPU {gpu_id} not available, using GPU 0")
                device = torch.device("cuda:0")
        return device
    except Exception as e:
        logging.warning(f"Invalid device '{device_str}': {e}, falling back to CPU")
        return torch.device("cpu")
