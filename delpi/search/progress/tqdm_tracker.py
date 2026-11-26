"""CLI progress tracker: a tqdm display on top of ``CallbackProgressTracker``."""

from __future__ import annotations

from typing import Callable, Optional

from tqdm import tqdm as _tqdm

from delpi.search.progress.callback_tracker import CallbackProgressTracker
from delpi.search.progress.snapshot import ProgressSnapshot


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
        self._overall_n: float = 0.0  # fractional progress; the bar holds ints
        self.active_child: Optional[TqdmProgressTracker] = None

    # overall bar

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

    # subtask bar

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

    # lifecycle

    def close(self) -> None:
        self.close_subtask()
        self.overall_bar.close()


class TqdmProgressTracker(CallbackProgressTracker):
    """CLI progress tracker backed by two tqdm bars.

    Bar 0 (``position=0``) shows the root's overall progress; bar 1
    (``position=1``) shows the most recently created child and is replaced on
    each ``create_child``. ``callback`` is passed to
    :class:`CallbackProgressTracker`. The private keyword arguments are only
    for spawning children.
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

    # hook overrides (display layer)

    def _on_update(self) -> None:
        """Update tqdm bars, then fire the callback via super."""
        if self._display is not None:
            # sync the overall bar to the root; all nodes share _display
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

    def advance(self, n: int = 1) -> None:
        if self._completed:
            return

        # tick the subtask bar before the parent logic runs
        if (
            not self.is_root
            and self._display is not None
            and self._display.active_child is self
        ):
            self._display.update_subtask(n)

        super().advance(n)

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
        # super().create_child is not called, so do its steps here
        self._on_child_created(child)

        root = self._root
        root._active_description = description
        root._active_current = 0
        root._active_total = total
        self._on_update()
        return child

    def close(self) -> None:
        """Release tqdm resources (only the root tracker should call this)."""
        if self.is_root and self._display is not None:
            self._display.close()
            self._display = None
