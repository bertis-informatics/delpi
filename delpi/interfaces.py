class BaseProgressReporter:
    """Base interface for reporting progress (intended for inheritance)"""
    def update(self, current, total, msg=""):
        pass  # Default: do nothing (to be overridden in GUI, etc.)