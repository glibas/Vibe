"""
Vibe: Vision transformer for Interface Beauty Evaluation

A comprehensive framework for predicting web interface aesthetics using 
saliency-guided Vision Transformers with multi-scale ROI selection.
"""

__version__ = "0.1.0"
__author__ = "Vibe Team"
__email__ = "contact@vibe-ai.com"

from .models import BeautyPredictor
from .evaluation import InterfaceEvaluator
from .config import load_config, create_default_config
from .utils import setup_logging

__all__ = [
    "BeautyPredictor",
    "InterfaceEvaluator",
    "load_config",
    "create_default_config", 
    "setup_logging",
]