# readers/__init__.py
from .factory import ReaderFactory

get_reader = ReaderFactory.get_reader

__all__ = ["ReaderFactory", "get_reader"]