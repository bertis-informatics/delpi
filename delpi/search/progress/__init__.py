"""Hierarchical progress tracking with tqdm, callback (GUI) or silent back-ends.

::

    from delpi.search.progress import create_progress_tracker

    # CLI: overall and subtask tqdm bars
    tracker = create_progress_tracker("tqdm", total=100, description="Overall")

    # callback: receives a ProgressSnapshot on every update
    def on_progress(snap):
        print(f"{snap.overall_percent:.0f}% {snap.time_display}")

    tracker = create_progress_tracker("callback", total=100, callback=on_progress)

    # silent
    tracker = create_progress_tracker("dummy")

See `ProgressTracker` for the child/portion workflow.
"""

from delpi.search.progress.tracker import ProgressTracker
from delpi.search.progress.dummy_tracker import DummyProgressTracker
from delpi.search.progress.snapshot import ProgressSnapshot
from delpi.search.progress.callback_tracker import CallbackProgressTracker
from delpi.search.progress.tqdm_tracker import TqdmProgressTracker

__all__ = [
    "ProgressTracker",
    "DummyProgressTracker",
    "CallbackProgressTracker",
    "ProgressSnapshot",
    "TqdmProgressTracker",
    "create_progress_tracker",
]


def create_progress_tracker(
    mode: str = "tqdm",
    total: int = 100,
    description: str = "",
    callback=None,
) -> ProgressTracker:
    """Create a progress tracker.

    Parameters
    ----------
    mode : str
        ``"tqdm"`` (two CLI bars), ``"callback"`` (headless, calls `callback` on
        every state change) or ``"dummy"``/``"silent"``/``"none"`` (no-op).
    total : int
        Total steps of the root tracker.
    description : str
        Label of the root bar.
    callback : callable, optional
        ``callback(snapshot: ProgressSnapshot) -> None``; used by ``"callback"``
        and also by ``"tqdm"``.
    """
    if mode == "tqdm":
        return TqdmProgressTracker(
            total=total,
            description=description,
            callback=callback,
        )
    elif mode == "callback":
        return CallbackProgressTracker(
            total=total,
            description=description,
            callback=callback,
        )
    elif mode in ("dummy", "silent", "none"):
        return DummyProgressTracker()
    else:
        raise ValueError(f"Unknown progress tracker mode: {mode!r}")
