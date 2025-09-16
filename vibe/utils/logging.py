"""
Logging utilities for Vibe.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "./logs",
    logger_name: str = "vibe"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file name (auto-generated if None)
        log_dir: Directory to save log files
        logger_name: Name of the logger
        
    Returns:
        Configured logger
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"vibe_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(logger_name)
    logger.info(f"Logging initialized. Log file: {log_path}")
    
    return logger


def get_logger(name: str = "vibe") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


class ModelLogger:
    """Logger for model training and evaluation."""
    
    def __init__(self, logger_name: str = "vibe.model"):
        self.logger = get_logger(logger_name)
    
    def log_model_info(self, model):
        """Log model architecture information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {model.__class__.__name__}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def log_training_start(self, config):
        """Log training configuration."""
        self.logger.info("=" * 50)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 50)
        self.logger.info(f"Experiment: {config.get('name', 'unnamed')}")
        self.logger.info(f"Model type: {config.get('model_type', 'unknown')}")
        self.logger.info(f"Batch size: {config.get('batch_size', 'unknown')}")
        self.logger.info(f"Learning rate: {config.get('learning_rate', 'unknown')}")
        self.logger.info(f"Epochs: {config.get('num_epochs', 'unknown')}")
        
    def log_epoch_results(self, epoch, train_loss, val_loss=None, val_metrics=None):
        """Log epoch results."""
        msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}"
        
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
            
        if val_metrics:
            for metric_name, value in val_metrics.items():
                msg += f" | {metric_name}: {value:.4f}"
        
        self.logger.info(msg)
    
    def log_best_model(self, epoch, metric_value, metric_name="loss"):
        """Log when best model is found."""
        self.logger.info(f"🏆 New best model at epoch {epoch} with {metric_name}: {metric_value:.4f}")
    
    def log_training_complete(self, total_time, best_metric):
        """Log training completion."""
        self.logger.info("=" * 50)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 50)
        self.logger.info(f"Total training time: {total_time:.2f} seconds")
        self.logger.info(f"Best validation metric: {best_metric:.4f}")


class EvaluationLogger:
    """Logger for model evaluation."""
    
    def __init__(self, logger_name: str = "vibe.evaluation"):
        self.logger = get_logger(logger_name)
    
    def log_evaluation_start(self, dataset_name, num_samples):
        """Log evaluation start."""
        self.logger.info(f"Starting evaluation on {dataset_name} ({num_samples} samples)")
    
    def log_evaluation_results(self, results):
        """Log evaluation results."""
        self.logger.info("Evaluation Results:")
        self.logger.info("-" * 30)
        
        if 'overall' in results:
            overall = results['overall']
            self.logger.info(f"RMSE: {overall.get('rmse', 0):.4f}")
            self.logger.info(f"MAE: {overall.get('mae', 0):.4f}")
            self.logger.info(f"Pearson: {overall.get('pearson', 0):.4f}")
            self.logger.info(f"Spearman: {overall.get('spearman', 0):.4f}")
        
        if 'aspects' in results:
            self.logger.info("\nBeauty Aspects:")
            for aspect, metrics in results['aspects'].items():
                self.logger.info(f"  {aspect}: MAE={metrics.get('mae', 0):.4f}, "
                               f"Pearson={metrics.get('pearson', 0):.4f}")
        
        if 'principles' in results:
            self.logger.info("\nDesign Principles:")
            for principle, metrics in results['principles'].items():
                self.logger.info(f"  {principle}: MAE={metrics.get('mae', 0):.4f}, "
                               f"Pearson={metrics.get('pearson', 0):.4f}")


# Global logger instances
model_logger = ModelLogger()
eval_logger = EvaluationLogger()