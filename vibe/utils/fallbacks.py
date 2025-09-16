"""
Simple fallback implementations for missing dependencies.
"""

import warnings


def create_missing_import_warning(module_name, feature):
    """Create a warning for missing optional dependencies."""
    return f"""
{module_name} is not installed. Some features may not work properly.
To install: pip install {module_name}
Affected feature: {feature}
"""


# CV2 fallback
try:
    import cv2
except ImportError:
    warnings.warn(create_missing_import_warning("opencv-python", "Image processing and ROI visualization"))
    cv2 = None


# Pandas fallback  
try:
    import pandas as pd
except ImportError:
    warnings.warn(create_missing_import_warning("pandas", "Data loading from CSV files"))
    pd = None


# Sklearn fallback
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    warnings.warn(create_missing_import_warning("scikit-learn scipy", "Advanced metrics computation"))
    
    def mean_squared_error(y_true, y_pred):
        """Simple MSE implementation."""
        import torch
        if isinstance(y_true, torch.Tensor):
            return torch.mean((y_true - y_pred) ** 2).item()
        else:
            return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
    
    def mean_absolute_error(y_true, y_pred):
        """Simple MAE implementation.""" 
        import torch
        if isinstance(y_true, torch.Tensor):
            return torch.mean(torch.abs(y_true - y_pred)).item()
        else:
            return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)
    
    def pearsonr(x, y):
        """Simple Pearson correlation implementation."""
        import torch
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            y = y.cpu().numpy()
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x_sq = sum(xi**2 for xi in x)
        sum_y_sq = sum(yi**2 for yi in y)
        sum_xy = sum(xi*yi for xi, yi in zip(x, y))
        
        num = n * sum_xy - sum_x * sum_y
        den = ((n * sum_x_sq - sum_x**2) * (n * sum_y_sq - sum_y**2))**0.5
        
        if den == 0:
            return (0.0, 1.0)
        return (num / den, 0.0)  # Return correlation and dummy p-value
    
    def spearmanr(x, y):
        """Simple Spearman correlation - just use Pearson as fallback."""
        return pearsonr(x, y)


# Matplotlib fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    warnings.warn(create_missing_import_warning("matplotlib seaborn", "Visualization and plotting"))
    plt = None
    sns = None


# TQDM fallback
try:
    from tqdm import tqdm
except ImportError:
    warnings.warn(create_missing_import_warning("tqdm", "Progress bars"))
    def tqdm(iterable, **kwargs):
        """Simple tqdm fallback that just returns the iterable."""
        return iterable