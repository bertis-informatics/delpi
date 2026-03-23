"""
Threading-based batch prefetcher for overlapping CPU batch preparation
with GPU inference.

Numba's ``prange`` releases the GIL, so the CPU-bound
``_make_batch_in_parallel`` truly runs in parallel with the main thread's
GPU work when driven from a background ``threading.Thread``.
"""

import threading
from queue import Queue

import numpy as np
import torch


def prefetch_batches(batch_iter, prefetch_count=2):
    """
    Wrap a batch iterator so the next batch is prepared on a background
    thread while the current batch is being consumed (GPU inference).

    Each yielded element is a tuple of **pinned-memory torch tensors** ready
    for ``non_blocking`` transfer to GPU.

    Args:
        batch_iter: The generator from ``generate_batches`` (double-buffered).
        prefetch_count: How many batches to buffer ahead.  2 is usually
            enough to keep both CPU and GPU saturated.
    """
    q: Queue = Queue(maxsize=prefetch_count)
    sentinel = object()
    error_box: list = []

    def _producer():
        try:
            for batch in batch_iter:
                pinned = tuple(
                    torch.from_numpy(np.ascontiguousarray(arr)).pin_memory()
                    for arr in batch
                )
                q.put(pinned)
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


def prefetch_dataloader(dataloader, prefetch_count=2):
    """
    Prefetch batches from a :class:`~torch.utils.data.DataLoader` on a
    background thread while the main thread runs GPU inference.

    The producer thread calls ``next(dataloader)`` and pins every tensor
    in the resulting dict so that subsequent ``.to(device, non_blocking=True)``
    calls use fast DMA transfers.

    Args:
        dataloader: A PyTorch DataLoader yielding ``dict[str, Tensor | Any]``.
        prefetch_count: How many batches to buffer ahead (default 2).
    """
    q: Queue = Queue(maxsize=prefetch_count)
    sentinel = object()
    error_box: list = []

    def _producer():
        try:
            for batch in dataloader:
                pinned = {
                    k: v.pin_memory() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                q.put(pinned)
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
