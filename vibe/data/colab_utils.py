"""
Google Colab utilities for Vibe interface beauty evaluation.

This module provides convenient functions for working with datasets
and models in Google Colab environment.
"""

import os
import shutil
import zipfile
from typing import Dict, Optional, Tuple
import pandas as pd
from pathlib import Path


def load_webdesign_dataset(
    drive_path: str,
    extract_to: str = '/content/webdesignprototypicality',
    force_reload: bool = False
) -> str:
    """
    Load WebDesignPrototypicality dataset from Google Drive.
    
    Args:
        drive_path: Path to the dataset zip file in Google Drive
        extract_to: Local path to extract dataset
        force_reload: Whether to re-extract even if already exists
        
    Returns:
        Path to extracted dataset directory
        
    Example:
        ```python
        from vibe.data.colab_utils import load_webdesign_dataset
        
        dataset_path = load_webdesign_dataset(
            '/content/drive/MyDrive/datasets/webdesignprototypicality.zip'
        )
        ```
    """
    local_zip = '/content/webdesignprototypicality.zip'
    
    if os.path.exists(extract_to) and not force_reload:
        print(f"📊 Dataset already available at: {extract_to}")
        return extract_to
    
    try:
        # Copy zip from Drive to local storage
        print("📥 Copying dataset from Google Drive...")
        shutil.copy(drive_path, local_zip)
        
        # Extract dataset
        print("📦 Extracting dataset...")
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Clean up zip file
        os.remove(local_zip)
        
        print(f"✅ Dataset extracted to: {extract_to}")
        return extract_to
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Make sure the dataset zip file exists in your Google Drive")
        raise


def setup_webdesign_dataset(dataset_path: str) -> Dict[str, str]:
    """
    Setup WebDesignPrototypicality dataset paths and structure.
    
    Args:
        dataset_path: Path to extracted dataset directory
        
    Returns:
        Dictionary with dataset paths
    """
    dataset_path = Path(dataset_path)
    
    # Look for common dataset structures
    possible_structures = [
        dataset_path / "images",
        dataset_path / "webdesignprototypicality" / "images",
        dataset_path / "data" / "images",
    ]
    
    images_dir = None
    for possible_path in possible_structures:
        if possible_path.exists():
            images_dir = possible_path
            break
    
    if images_dir is None:
        raise FileNotFoundError(f"Could not find images directory in {dataset_path}")
    
    # Look for annotations file
    possible_annotations = [
        dataset_path / "annotations.csv",
        dataset_path / "webdesignprototypicality" / "annotations.csv", 
        dataset_path / "data.csv",
        dataset_path / "labels.csv",
    ]
    
    annotations_file = None
    for possible_file in possible_annotations:
        if possible_file.exists():
            annotations_file = possible_file
            break
    
    paths = {
        "dataset_root": str(dataset_path),
        "images_dir": str(images_dir),
        "annotations_file": str(annotations_file) if annotations_file else None,
    }
    
    print(f"📁 Dataset structure detected:")
    print(f"  Root: {paths['dataset_root']}")
    print(f"  Images: {paths['images_dir']}")
    print(f"  Annotations: {paths['annotations_file']}")
    
    return paths


def create_train_val_split(
    annotations_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    output_dir: str = '/content/splits'
) -> Dict[str, str]:
    """
    Create train/validation/test splits from annotations file.
    
    Args:
        annotations_file: Path to annotations CSV file
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        output_dir: Directory to save split files
        
    Returns:
        Dictionary with paths to split files
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, val, and test ratios must sum to 1.0")
    
    # Load annotations
    df = pd.read_csv(annotations_file)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split data
    train_df = df[:n_train]
    val_df = df[n_train:n_train + n_val]
    test_df = df[n_train + n_val:]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    split_files = {}
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_path = os.path.join(output_dir, f'{split_name}_annotations.csv')
        split_df.to_csv(split_path, index=False)
        split_files[split_name] = split_path
        print(f"📊 {split_name.capitalize()} split: {len(split_df)} samples -> {split_path}")
    
    return split_files


def setup_colab_environment():
    """
    Setup Google Colab environment for Vibe.
    
    This function configures the environment variables and
    paths needed for optimal performance in Colab.
    """
    import matplotlib.pyplot as plt
    import warnings
    
    # Configure matplotlib for Colab
    plt.style.use('default')
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Set environment variables for optimal performance
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ GPU not available, using CPU")
    
    print("✅ Colab environment configured for Vibe")


def mount_drive_and_setup(dataset_zip_path: str = None) -> str:
    """
    Convenience function to mount Google Drive and setup dataset.
    
    Args:
        dataset_zip_path: Optional path to dataset zip in Drive
        
    Returns:
        Path to extracted dataset
    """
    from google.colab import drive
    
    # Mount Google Drive
    print("📱 Mounting Google Drive...")
    drive.mount('/content/drive')
    
    # Setup environment
    setup_colab_environment()
    
    if dataset_zip_path:
        # Load dataset if path provided
        return load_webdesign_dataset(dataset_zip_path)
    else:
        print("💡 Dataset zip path not provided. Use load_webdesign_dataset() to load data.")
        return None


# Convenience imports for notebooks
def quick_setup():
    """
    Quick setup function for notebook imports and environment.
    
    Example usage at the start of a notebook:
    ```python
    from vibe.data.colab_utils import quick_setup
    quick_setup()
    ```
    """
    # Setup environment
    setup_colab_environment()
    
    # Print quick start info
    print("🎨 Vibe: Vision transformer for Interface Beauty Evaluation")
    print("=" * 60)
    print("Quick setup complete! Ready for interface beauty analysis.")
    print("\nNext steps:")
    print("1. Mount Drive: mount_drive_and_setup()")
    print("2. Load dataset: load_webdesign_dataset(drive_path)")
    print("3. Create model: BeautyPredictor(model_type='saliency_guided')")
    print("=" * 60)