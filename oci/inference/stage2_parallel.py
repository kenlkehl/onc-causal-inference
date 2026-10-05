"""Bounded independent feature work with deterministic result ordering."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def ordered_map(function, items, *, workers=1, thread_name="stage2-feature"):
    if type(workers) is not int or workers < 1:
        raise ValueError("feature request workers must be a positive integer")
    if workers == 1:
        return list(map(function, items))
    source = iter(enumerate(items))
    pending, results = {}, {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name) as pool:
        def submit():
            item = next(source, None)
            if item is not None:
                index, value = item
                pending[pool.submit(function, value)] = index

        try:
            for _ in range(workers):
                submit()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Observe all errors in this completed batch before submitting
                # additional work. Already completed feature checkpoints survive.
                for future in done:
                    results[pending.pop(future)] = future.result()
                for _ in done:
                    submit()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return [results[index] for index in range(len(results))]
