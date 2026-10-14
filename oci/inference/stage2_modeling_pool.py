"""One bounded CPU process pool shared by all Stage 2 outer-fold threads.

Only leaf numerical work goes into this pool. Coordinators and LLM clients stay
in the parent; penalty warm starts remain sequential within each submitted path.
Workers never create nested pools. Threaded final forests reserve the same budget.
"""

from concurrent.futures import Future
from collections import deque
from contextlib import contextmanager
import logging
import threading

from joblib.externals import cloudpickle
from joblib.externals.loky import ProcessPoolExecutor
from joblib import parallel_backend
from threadpoolctl import threadpool_limits

LOGGER = logging.getLogger(__name__)
_session_lock = threading.Lock()
_pool = None


def _execute(payload):
    with threadpool_limits(limits=1):
        return cloudpickle.loads(payload)()


class ModelingPool:
    def __init__(self, workers, parallel_fits=1):
        self.workers = workers
        self.threads_per_fit = max(1, workers // parallel_fits)
        # Submitted process jobs and in-parent threaded fits share this budget.
        # Reserve before submission, including queued work, to prevent nested
        # estimator parallelism from oversubscribing the machine.
        self.capacity = threading.Condition()
        self.available = workers
        self.waiters = deque()
        self.executor = ProcessPoolExecutor(
            max_workers=workers,
            env={
                name: "1"
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        )

    def acquire(self, maximum=1):
        with self.capacity:
            ticket = object()
            self.waiters.append(ticket)
            try:
                # FIFO reservations let a forest acquire its full bounded share
                # even while other folds continuously submit single-fit jobs.
                self.capacity.wait_for(
                    lambda: self.waiters[0] is ticket and self.available >= maximum
                )
                self.available -= maximum
                return maximum
            finally:
                self.waiters.remove(ticket)
                self.capacity.notify_all()

    def release(self, count):
        with self.capacity:
            self.available += count
            self.capacity.notify_all()

    def submit(self, function):
        self.acquire()
        try:
            # Freeze closure variables before the coordinator advances its loops.
            payload = cloudpickle.dumps(function)
            future = self.executor.submit(_execute, payload)
        except BaseException:
            self.release(1)
            raise
        future.add_done_callback(lambda _: self.release(1))
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
def threaded_fit():
    """Reserve CPU capacity for a joblib estimator running in the coordinator.

    Never use this inside a submitted job: workers see no shared pool and get
    one job. Native BLAS/OpenMP limits remain one; only joblib work is threaded.
    """
    pool = _pool
    jobs = pool.acquire(pool.threads_per_fit) if pool is not None else 1
    try:
        with parallel_backend("threading", n_jobs=jobs):
            yield jobs
    finally:
        if pool is not None:
            pool.release(jobs)


@contextmanager
def modeling_session(workers, *, parallel_fits=1):
    """Own the pool for a run; one worker retains the serial execution path."""
    global _pool
    if type(workers) is not int or workers < 1:
        raise ValueError("stage2.modeling_workers must be a positive integer")
    if type(parallel_fits) is not int or parallel_fits < 1:
        raise ValueError("parallel_fits must be a positive integer")
    if not _session_lock.acquire(blocking=False):
        raise RuntimeError("a Stage 2 modeling session is already active in this process")
    try:
        _pool = ModelingPool(workers, parallel_fits) if workers > 1 else None
        LOGGER.info(
            "Stage 2 CPU modeling workers=%s shared_across_outer_folds=true threads_per_fit=1",
            workers,
        )
        with threadpool_limits(limits=1):
            yield
    finally:
        pool, _pool = _pool, None
        try:
            if pool is not None:
                pool.executor.shutdown(wait=True)
        finally:
            _session_lock.release()
