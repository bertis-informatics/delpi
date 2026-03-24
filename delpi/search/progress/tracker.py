"""Abstract base class for hierarchical progress tracking.

Design inspired by the BiscuitTracker pattern in ``_tracker.py``:
each child has a *portion* of its parent's total, and child-level progress
is automatically rescaled and propagated upward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProgressTracker(ABC):
    """Common interface for hierarchical progress tracking (GUI / CLI).

    Workflow
    --------
    1. Create a root tracker with a ``total`` and ``description``.
    2. Spawn child trackers via ``create_child`` for each phase/subtask.
       The *portion* parameter says how many of the **parent's** steps this
       child covers when it runs from 0 % → 100 %.
    3. Inside each subtask, call ``advance(n)`` to report progress.
    4. When a subtask finishes, call ``complete()``; any un-reported progress
       is propagated to the parent so the overall bar stays accurate.
    5. ``close()`` releases display resources (only needed on the root).

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
        """Create a child tracker representing a subtask.

        Parameters
        ----------
        description : str
            Label shown for this subtask.
        total : int
            Number of steps in the subtask's own work.
        portion : float
            How many of **this** tracker's steps the child covers when it
            completes (0 → 100 %).  ``sum(portions) == self.total`` is a
            good convention but not enforced.

        Returns
        -------
        ProgressTracker
            A new tracker that forwards updates back to *self*.
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

    # Context-manager support -------------------------------------------------

    def __enter__(self) -> ProgressTracker:
        return self

    def __exit__(self, *args) -> None:
        self.close()
