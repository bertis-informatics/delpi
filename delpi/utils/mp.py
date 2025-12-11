import sys
import multiprocessing


def get_multiprocessing_context(
    preferred_unix_method: str = "forkserver",
) -> multiprocessing.context.BaseContext:
    """
    Returns an appropriate multiprocessing context based on the OS.

    On Windows: returns 'spawn'
    On Unix-like OS: returns 'forkserver' (default), or fallback to 'spawn' if unavailable
    """
    if sys.platform == "win32":
        return multiprocessing.get_context("spawn")

    # For Unix-like (Linux, macOS)
    try:
        return multiprocessing.get_context(preferred_unix_method)
    except ValueError:
        # If 'forkserver' not available, fallback to 'spawn'
        return multiprocessing.get_context("spawn")
