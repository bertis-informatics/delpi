"""
CLI progress tracker using tqdm with a two-bar display.

* Bar 0 (position 0): overall progress of the root tracker
* Bar 1 (position 1): currently active subtask

Inherits all hierarchical logic and time-tracking from
``CallbackProgressTracker``; this file only adds the tqdm display layer.
"""

from __future__ import annotations

from typing import Callable, Optional

from tqdm import tqdm as _tqdm

from delpi.search.progress.callback_tracker import CallbackProgressTracker
from delpi.search.progress.snapshot import ProgressSnapshot


# ---------------------------------------------------------------------------
# Internal display manager
# ---------------------------------------------------------------------------


class _TqdmDisplay:
    """Manages the two tqdm bars visible in the terminal."""

    def __init__(self, total: int, description: str):
        self.overall_bar = _tqdm(
            total=total,
            position=0,
            desc=description,
            leave=True,
        )
        self.subtask_bar: Optional[_tqdm] = None
        self._overall_n: float = 0.0  # float accumulator for precise tracking
        self.active_child: Optional[TqdmProgressTracker] = None

    # -- overall bar --------------------------------------------------------

    def update_overall(self, increment: float) -> None:
        """Advance the overall bar by a (possibly fractional) amount."""
        old_n = self.overall_bar.n
        self._overall_n = min(
            self._overall_n + increment,
            self.overall_bar.total or 0,
        )
        new_n = int(self._overall_n)
        delta = new_n - old_n
        if delta > 0:
            self.overall_bar.update(delta)

    def complete_overall(self) -> None:
        """Force the overall bar to 100 %."""
        remaining = (self.overall_bar.total or 0) - self.overall_bar.n
        if remaining > 0:
            self.overall_bar.update(remaining)

    # -- subtask bar --------------------------------------------------------

    def set_subtask(self, total: int, description: str) -> None:
        """Create (or replace) the subtask bar."""
        if self.subtask_bar is not None:
            self.subtask_bar.close()
        self.subtask_bar = _tqdm(
            total=total,
            position=1,
            desc=description,
            leave=False,
        )

    def update_subtask(self, n: int) -> None:
        if self.subtask_bar is not None:
            self.subtask_bar.update(n)

    def close_subtask(self) -> None:
        if self.subtask_bar is not None:
            self.subtask_bar.close()
            self.subtask_bar = None
        self.active_child = None

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self.close_subtask()
        self.overall_bar.close()


# ---------------------------------------------------------------------------
# Public tracker
# ---------------------------------------------------------------------------


class TqdmProgressTracker(CallbackProgressTracker):
    """CLI progress tracker backed by two tqdm bars.

    Inherits hierarchical progress propagation, time-tracking, and callback
    support from :class:`CallbackProgressTracker`.  This subclass adds:

    * **Bar 0** (``position=0``): overall progress of the *root* tracker.
    * **Bar 1** (``position=1``): the most-recently created child tracker
      (the *active subtask*).  Each call to ``create_child`` replaces
      this bar.

    Parameters
    ----------
    total : int
        Number of steps for this tracker.
    description : str
        Label shown on the progress bar.
    callback : callable, optional
        Forwarded to ``CallbackProgressTracker``.

    Notes
    -----
    The private keyword arguments ``_parent``, ``_display``, ``_portion``,
    ``_root`` are used internally when spawning child trackers.  Callers
    should not pass them.
    """

    def __init__(
        self,
        total: int,
        description: str = "",
        callback: Optional[Callable[[ProgressSnapshot], None]] = None,
        *,
        _parent: Optional[TqdmProgressTracker] = None,
        _root: Optional[CallbackProgressTracker] = None,
        _display: Optional[_TqdmDisplay] = None,
        _portion: float = 0.0,
    ):
        super().__init__(
            total=total,
            description=description,
            callback=callback,
            _parent=_parent,
            _root=_root,
            _portion=_portion,
        )

        if _parent is None:
            self._display: Optional[_TqdmDisplay] = _TqdmDisplay(total, description)
        else:
            self._display = _display

    # -- hook overrides (display layer) -------------------------------------

    def _on_update(self) -> None:
        """Update tqdm bars, then fire the callback via super."""
        if self._display is not None:
            # Sync overall bar to the root's accumulated value.
            # All nodes share the same _display, so any node can trigger this.
            root = self._root
            delta = root._overall_current - self._display._overall_n
            if delta > 0:
                self._display.update_overall(delta)

        super()._on_update()

    def _on_child_created(self, child: CallbackProgressTracker) -> None:
        """Replace subtask bar for the newly created child."""
        if self._display is not None:
            self._display.set_subtask(child._total, child._description)
            self._display.active_child = child

    def _on_completed(self) -> None:
        """Force bars to completion / close subtask bar."""
        if self.is_root:
            if self._display is not None:
                self._display.complete_overall()
        else:
            # Close the subtask bar if this tracker owns it
            if self._display is not None and self._display.active_child is self:
                self._display.close_subtask()

    # -- override advance to also tick the subtask bar ----------------------

    def advance(self, n: int = 1) -> None:
        if self._completed:
            return

        # Tick the subtask bar *before* the parent logic runs
        if (
            not self.is_root
            and self._display is not None
            and self._display.active_child is self
        ):
            self._display.update_subtask(n)

        super().advance(n)

    # -- override create_child to produce TqdmProgressTracker instances -----

    def create_child(
        self, description: str, total: int, portion: float
    ) -> TqdmProgressTracker:
        child = TqdmProgressTracker(
            total=total,
            description=description,
            _parent=self,
            _root=self._root,
            _display=self._display,
            _portion=portion,
        )
        # _on_child_created hook is called by super().create_child, but
        # we build the child ourselves so call hooks manually
        self._on_child_created(child)

        # Update root metadata
        root = self._root
        root._active_description = description
        root._active_current = 0
        root._active_total = total
        self._on_update()
        return child

    # -- override close to release tqdm resources ---------------------------

    def close(self) -> None:
        """Release tqdm resources (only the root tracker should call this)."""
        if self.is_root and self._display is not None:
            self._display.close()
            self._display = None
