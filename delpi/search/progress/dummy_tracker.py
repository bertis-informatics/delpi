"""No-op progress tracker for when no display is needed."""

from __future__ import annotations

from delpi.search.progress.tracker import ProgressTracker


class DummyProgressTracker(ProgressTracker):
    """Progress tracker that silently discards all updates."""

    def advance(self, n: int = 1) -> None:
        pass

    def create_child(
        self, description: str, total: int, portion: float
    ) -> DummyProgressTracker:
        return DummyProgressTracker()

    def complete(self) -> None:
        pass

    def close(self) -> None:
        pass
