import sys
import multiprocessing


def get_multiprocessing_context(
    preferred_unix_method: str = "forkserver",
) -> multiprocessing.context.BaseContext:
    """Multiprocessing context: 'spawn' on Windows, else preferred_unix_method
    with a 'spawn' fallback if it is unavailable."""
    if sys.platform == "win32":
        return multiprocessing.get_context("spawn")

    try:
        return multiprocessing.get_context(preferred_unix_method)
    except ValueError:
        return multiprocessing.get_context("spawn")
