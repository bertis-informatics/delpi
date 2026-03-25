"""Immutable snapshot of root tracker state at one point in time."""

from __future__ import annotations

from typing import Optional


class ProgressSnapshot:
    """Immutable snapshot of the root tracker state at one point in time.

    Attributes
    ----------
    overall_percent : float
        0-100 overall completion.
    overall_current : float
        Raw accumulated steps on the root tracker.
    overall_total : int
        Root tracker's total steps.
    subtask_description : str
        Label of the currently active subtask.
    subtask_current : int
        Steps completed in the current subtask.
    subtask_total : int
        Total steps in the current subtask.
    elapsed : float
        Seconds since root tracker creation.
    eta : float | None
        Estimated seconds remaining (``None`` if too early to estimate).
    """

    __slots__ = (
        "overall_percent",
        "overall_current",
        "overall_total",
        "subtask_description",
        "subtask_current",
        "subtask_total",
        "elapsed",
        "eta",
    )

    def __init__(
        self,
        overall_percent: float,
        overall_current: float,
        overall_total: int,
        subtask_description: str,
        subtask_current: int,
        subtask_total: int,
        elapsed: float,
        eta: Optional[float],
    ):
        self.overall_percent = overall_percent
        self.overall_current = overall_current
        self.overall_total = overall_total
        self.subtask_description = subtask_description
        self.subtask_current = subtask_current
        self.subtask_total = subtask_total
        self.elapsed = elapsed
        self.eta = eta

    def format_time(self, seconds: Optional[float]) -> str:
        """Format seconds as ``MM:SS`` or ``HH:MM:SS``."""
        if seconds is None:
            return "??:??"
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    @property
    def time_display(self) -> str:
        """``[elapsed / eta]`` string like tqdm shows."""
        return f"[{self.format_time(self.elapsed)}/{self.format_time(self.eta)}]"

    def __repr__(self) -> str:
        return (
            f"ProgressSnapshot("
            f"{self.overall_percent:.1f}%, "
            f"subtask={self.subtask_description!r} "
            f"{self.subtask_current}/{self.subtask_total}, "
            f"{self.time_display})"
        )
