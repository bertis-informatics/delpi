"""
Callback-based progress tracker with time estimation.

Core hierarchical progress logic: child trackers propagate proportionally
to parents, and a user-supplied callback is fired on every update.

``TqdmProgressTracker`` inherits from this class, adding only the tqdm
display layer on top.
"""

from __future__ import annotations

import time as _time
from typing import Callable, Optional

from delpi.search.progress.tracker import ProgressTracker
from delpi.search.progress.snapshot import ProgressSnapshot


class CallbackProgressTracker(ProgressTracker):
    """Hierarchical progress tracker with an optional callback.

    On every ``advance`` / ``complete`` / ``create_child`` event the root
    tracker's callback (if set) is called with a :class:`ProgressSnapshot`
    containing the overall percent, active subtask info, elapsed time, and
    estimated remaining time.

    Parameters
    ----------
    total : int
        Number of logical steps for this tracker.
    description : str
        Human-readable label.
    callback : callable, optional
        ``callback(snapshot: ProgressSnapshot) -> None``.
        Only the **root** tracker's callback is used; children forward
        events upward until it fires.
    """

    def __init__(
        self,
        total: int,
        description: str = "",
        callback: Optional[Callable[[ProgressSnapshot], None]] = None,
        *,
        _parent: Optional[CallbackProgressTracker] = None,
        _root: Optional[CallbackProgressTracker] = None,
        _portion: float = 0.0,
    ):
        self._total = total
        self._current: float = 0.0
        self._description = description
        self._parent = _parent
        self._portion = _portion
        self._completed = False

        # Root reference — every node can reach the root in O(1)
        if _parent is None:
            self._root: CallbackProgressTracker = self
            self._callback = callback
            self._start_time: float = _time.monotonic()
            self._overall_current: float = 0.0
            self._overall_total: int = total
            # Active subtask metadata (updated by the leaf that last called advance)
            self._active_description: str = description
            self._active_current: int = 0
            self._active_total: int = total
        else:
            self._root = _root or _parent._root
            # Non-root nodes don't own these; access via self._root
            self._callback = None
            self._start_time = None

    # -- public read-only properties ----------------------------------------

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since the root tracker was created."""
        return _time.monotonic() - self._root._start_time

    @property
    def overall_percent(self) -> float:
        root = self._root
        if root._overall_total <= 0:
            return 0.0
        return min(root._overall_current / root._overall_total * 100.0, 100.0)

    @property
    def eta(self) -> Optional[float]:
        """Estimated seconds remaining, or ``None`` if not enough data."""
        pct = self.overall_percent
        if pct < 0.1:
            return None
        elapsed = self.elapsed
        return elapsed / pct * (100.0 - pct)

    def get_snapshot(self) -> ProgressSnapshot:
        """Build a snapshot of the current root-level state."""
        root = self._root
        return ProgressSnapshot(
            overall_percent=self.overall_percent,
            overall_current=root._overall_current,
            overall_total=root._overall_total,
            subtask_description=root._active_description,
            subtask_current=root._active_current,
            subtask_total=root._active_total,
            elapsed=self.elapsed,
            eta=self.eta,
        )

    # -- ProgressTracker API ------------------------------------------------

    def advance(self, n: int = 1) -> None:
        if self._completed:
            return
        self._current += n

        root = self._root
        if self.is_root:
            root._overall_current = min(root._overall_current + n, root._overall_total)
        else:
            # Update active subtask metadata on root
            root._active_description = self._description
            root._active_current = int(self._current)
            root._active_total = self._total
            # Propagate proportionally to parent
            if self._total > 0:
                parent_inc = (n / self._total) * self._portion
                self._parent._advance_from_child(parent_inc)

        self._on_update()

    def create_child(
        self, description: str, total: int, portion: float
    ) -> CallbackProgressTracker:
        child = CallbackProgressTracker(
            total=total,
            description=description,
            _parent=self,
            _root=self._root,
            _portion=portion,
        )
        # Register new subtask on root
        root = self._root
        root._active_description = description
        root._active_current = 0
        root._active_total = total
        self._on_child_created(child)
        self._on_update()
        return child

    def complete(self) -> None:
        if self._completed:
            return
        self._completed = True

        if self.is_root:
            self._root._overall_current = self._root._overall_total
        elif self._parent is not None:
            # Propagate remaining un-reported progress
            if self._total > 0:
                already = (self._current / self._total) * self._portion
            else:
                already = 0.0
            remaining = self._portion - already
            if remaining > 1e-9:
                self._parent._advance_from_child(remaining)

        self._on_completed()
        self._on_update()

    def close(self) -> None:
        pass

    # -- internal propagation -----------------------------------------------

    def _advance_from_child(self, amount: float) -> None:
        """Receive a proportional progress increment from a child."""
        self._current += amount
        if self.is_root:
            self._root._overall_current = min(
                self._root._overall_current + amount, self._root._overall_total
            )
        elif self._parent is not None:
            if self._total > 0:
                parent_inc = (amount / self._total) * self._portion
                self._parent._advance_from_child(parent_inc)

    # -- hooks for subclasses (template method pattern) ---------------------

    def _on_update(self) -> None:
        """Called after any state change.  Fires the root callback."""
        root = self._root
        if root._callback is not None:
            root._callback(self.get_snapshot())

    def _on_child_created(self, child: CallbackProgressTracker) -> None:
        """Called when ``create_child`` produces a new tracker."""
        pass

    def _on_completed(self) -> None:
        """Called when ``complete()`` is invoked on this tracker."""
        pass

    # -- public forwarding ---------------------------------------------------

    def forward_snapshot(self, snapshot: ProgressSnapshot) -> None:
        """Forward an externally-produced snapshot to this tracker's callback.

        This is used by the queue-based subprocess bridge: the parent
        process reads :class:`ProgressSnapshot` objects from a
        ``multiprocessing.Queue`` and hands them to this method so that
        the user-supplied callback fires in the parent process.
        """
        if self._callback is not None:
            self._callback(snapshot)
