"""Abstract base for hierarchical progress tracking.

Modeled on the BiscuitTracker pattern in ``_tracker.py``: each child covers a
portion of its parent's total, and its progress is rescaled and propagated up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProgressTracker(ABC):
    """Common interface for hierarchical progress tracking (GUI / CLI).

    Create a root tracker, then one child per phase with ``create_child``;
    ``portion`` is how many of the parent's steps the child covers from 0 to
    100 %. Call ``advance(n)`` while working and ``complete()`` when done, which
    passes any unreported progress to the parent. ``close()`` releases display
    resources and is only needed on the root.

    Example::

        tracker = TqdmProgressTracker(total=100, description="Overall")

        # Phase 1 covers 20 of the root's 100 steps
        phase1 = tracker.create_child("Data-Prep", total=10, portion=20)
        for i in range(10):
            phase1.advance(1)
        phase1.complete()          # root now at ~20 %

        # Phase 2 covers 80 of the root's 100 steps
        phase2 = tracker.create_child("Search", total=30, portion=80)
        for win in range(30):
            phase2.advance(1)
        phase2.complete()          # root now at 100 %

        tracker.complete()
        tracker.close()
    """

    @abstractmethod
    def advance(self, n: int = 1) -> None:
        """Advance this tracker by *n* steps.

        If this is a child, progress is propagated proportionally to its parent.
        """
        ...

    @abstractmethod
    def create_child(
        self, description: str, total: int, portion: float
    ) -> ProgressTracker:
        """Create a child tracker for a subtask; it forwards updates to self.

        Parameters
        ----------
        description : str
            Label of the subtask.
        total : int
            Steps in the subtask's own work.
        portion : float
            Steps of this tracker the child covers once complete. Portions
            usually sum to ``self.total``; this is not enforced.
        """
        ...

    @abstractmethod
    def complete(self) -> None:
        """Mark this tracker as complete and propagate remaining progress."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release display resources (only the root tracker needs this)."""
        ...

    # context manager

    def __enter__(self) -> ProgressTracker:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # tqdm-compatible alias

    def update(self, n: int = 1) -> None:
        """Alias for :meth:`advance` (tqdm-compatible duck typing)."""
        self.advance(n)
