"""
Threading-based batch prefetcher for overlapping CPU batch preparation
with GPU inference.

Numba's ``prange`` releases the GIL, so the CPU-bound
``_make_batch_in_parallel`` truly runs in parallel with the main thread's
GPU work when driven from a background ``threading.Thread``.
"""

import threading
from typing import Callable, Iterable, Iterator, Optional, TypeVar
from queue import Queue

import numpy as np
import torch

T = TypeVar("T")
U = TypeVar("U")


class Prefetcher(Iterable[U]):
    """Thread-based prefetcher that eagerly evaluates items from an iterable
    on a background thread, optionally applying a *transform* before buffering.

    Args:
        iterable: Source iterable whose ``__next__`` may do expensive CPU work.
        transform: Optional function applied to each item before buffering.
            When *None*, items are yielded as-is.
        prefetch_count: How many transformed items to buffer ahead.
        thread_initializer: Optional callable invoked once in the producer
            thread before iteration begins.  Useful for setting thread-local
            state such as ``numba.set_num_threads()``.
    """

    def __init__(
        self,
        iterable: Iterable[T],
        transform: Optional[Callable[[T], U]] = None,
        prefetch_count: int = 2,
        thread_initializer: Optional[Callable[[], None]] = None,
    ):
        self._iterable = iterable
        self._transform = transform
        self._prefetch_count = prefetch_count
        self._thread_initializer = thread_initializer

    def __iter__(self) -> Iterator[U]:
        q: Queue = Queue(maxsize=self._prefetch_count)
        sentinel = object()
        error_box: list = []

        transform = self._transform
        initializer = self._thread_initializer

        def _producer():
            try:
                if initializer is not None:
                    initializer()
                for item in self._iterable:
                    q.put(transform(item) if transform is not None else item)
            except Exception as e:
                error_box.append(e)
            finally:
                q.put(sentinel)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item

        thread.join()
        if error_box:
            raise error_box[0]


# ---------------------------------------------------------------------------
# Built-in transforms for common use cases
# ---------------------------------------------------------------------------


def pin_numpy_tuple(batch):
    """Convert a tuple of numpy arrays to pinned-memory tensors."""
    return tuple(
        torch.from_numpy(np.ascontiguousarray(arr)).pin_memory() for arr in batch
    )


def pin_tensor_dict(batch):
    """Pin every tensor value in a dict batch."""
    return {
        k: v.pin_memory() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
