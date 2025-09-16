"""Utility functions for Vibe."""

from .logging import setup_logging, get_logger

# Import only basic functionality that exists
__all__ = [
    "setup_logging",
    "get_logger",
]

# Optional imports that may not be available
try:
    from .fallbacks import plt
    if plt is not None:
        __all__.extend(["visualize_predictions", "plot_training_curves"])
except ImportError:
    pass