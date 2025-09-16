"""Data processing utilities for interface beauty evaluation."""

from .dataset import InterfaceDataset, InterfaceDataLoader, create_sample_dataset

# Colab utilities (import conditionally)
try:
    from .colab_utils import (
        load_webdesign_dataset,
        setup_webdesign_dataset, 
        create_train_val_split,
        setup_colab_environment,
        mount_drive_and_setup,
        quick_setup
    )
    __all__ = [
        "InterfaceDataset",
        "InterfaceDataLoader",
        "create_sample_dataset",
        "load_webdesign_dataset",
        "setup_webdesign_dataset",
        "create_train_val_split", 
        "setup_colab_environment",
        "mount_drive_and_setup",
        "quick_setup",
    ]
except ImportError:
    # Pandas not available, skip colab utilities
    __all__ = [
        "InterfaceDataset",
        "InterfaceDataLoader",
        "create_sample_dataset",
    ]