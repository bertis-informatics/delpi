"""
Progress tracking module for DelPi search.

Provides a hierarchical progress tracking framework with pluggable
display back-ends (tqdm CLI, GUI callback, or silent dummy).

Quick start
-----------
::

    from delpi.search.progress import create_progress_tracker

    # CLI: two tqdm bars (overall + subtask)
    tracker = create_progress_tracker("tqdm", total=100, description="Overall")

    prep = tracker.create_child("Data-Prep", total=10, portion=20)
    for i in range(10):
        prep.advance(1)
    prep.complete()

    tracker.complete()
    tracker.close()

    # Callback mode: receive ProgressSnapshot on every update
    def on_progress(snap):
        print(f"{snap.overall_percent:.0f}% {snap.time_display}")

    tracker = create_progress_tracker("callback", total=100, callback=on_progress)

    # Silent mode: no display
    tracker = create_progress_tracker("dummy")
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
        ``"tqdm"`` for a CLI display with two tqdm bars,
        ``"callback"`` for a headless tracker that fires *callback*
        on every state change,
        ``"dummy"`` (or ``"silent"``) for a no-op tracker.
    total : int
        Total steps for the root tracker.
    description : str
        Label for the root progress bar.
    callback : callable, optional
        ``callback(snapshot: ProgressSnapshot) -> None``.
        Used when *mode* is ``"callback"`` (or ``"tqdm"`` if you want
        both CLI bars **and** a callback).

    Returns
    -------
    ProgressTracker
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
