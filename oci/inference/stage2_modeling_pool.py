"""One bounded CPU process pool shared by all Stage 2 outer-fold threads.

Only leaf numerical work goes into this pool. Coordinators, LLM clients and
dependent penalty paths stay in the parent; workers never create nested pools.
"""

from concurrent.futures import Future
from contextlib import contextmanager
import logging
import threading

from joblib.externals import cloudpickle
from joblib.externals.loky import ProcessPoolExecutor
from threadpoolctl import threadpool_limits

LOGGER = logging.getLogger(__name__)
_session_lock = threading.Lock()
_pool = None


def _execute(payload):
    with threadpool_limits(limits=1):
        return cloudpickle.loads(payload)()


class ModelingPool:
    def __init__(self, workers):
        self.workers = workers
        # Bound queued data across all folds, not separately for every producer.
        self.slots = threading.BoundedSemaphore(2 * workers)
        self.executor = ProcessPoolExecutor(
            max_workers=workers,
            env={name: "1" for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            )},
        )

    def submit(self, function):
        self.slots.acquire()
        try:
            # Freeze closure variables before the coordinator advances its loops.
            payload = cloudpickle.dumps(function)
            future = self.executor.submit(_execute, payload)
        except BaseException:
            self.slots.release()
            raise
        future.add_done_callback(lambda _: self.slots.release())
        return future


def enabled():
    return _pool is not None


def completed(value):
    future = Future()
    future.set_result(value)
    return future


def submit(function):
    if _pool is not None:
        return _pool.submit(function)
    with threadpool_limits(limits=1):
        return completed(function())


def call(function):
    return submit(function).result()


@contextmanager
def modeling_session(workers):
    """Own the pool for a run; one worker retains the serial execution path."""
    global _pool
    if type(workers) is not int or workers < 1:
        raise ValueError("stage2.modeling_workers must be a positive integer")
    if not _session_lock.acquire(blocking=False):
        raise RuntimeError("a Stage 2 modeling session is already active in this process")
    try:
        _pool = ModelingPool(workers) if workers > 1 else None
        LOGGER.info("Stage 2 CPU modeling workers=%s shared_across_outer_folds=true threads_per_fit=1", workers)
        with threadpool_limits(limits=1):
            yield
    finally:
        pool, _pool = _pool, None
        try:
            if pool is not None:
                pool.executor.shutdown(wait=True)
        finally:
            _session_lock.release()
