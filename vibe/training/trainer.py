"""
Training pipeline for interface beauty evaluation models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import json
import time

# Handle optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    from tqdm import tqdm
except ImportError:
    from ..utils.fallbacks import tqdm

from ..models import BeautyPredictor
from ..evaluation import InterfaceEvaluator


class BeautyLoss(nn.Module):
    """Multi-task loss for beauty evaluation."""
    
    def __init__(
        self,
        beauty_weight: float = 1.0,
        aspect_weight: float = 0.5,
        principle_weight: float = 0.3,
        consistency_weight: float = 0.2,
    ):
        super().__init__()
        self.beauty_weight = beauty_weight
        self.aspect_weight = aspect_weight
        self.principle_weight = principle_weight
        self.consistency_weight = consistency_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, predictions, targets):
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dictionary with model predictions
            targets: Dictionary with ground truth targets
            
        Returns:
            Total loss and loss components
        """
        total_loss = 0.0
        loss_components = {}
        
        # Main beauty score loss
        if 'beauty_score' in predictions and 'beauty_score' in targets:
            beauty_loss = self.mse_loss(predictions['beauty_score'], targets['beauty_score'])
            total_loss += self.beauty_weight * beauty_loss
            loss_components['beauty'] = beauty_loss.item()
        
        # Beauty aspects loss
        if 'beauty_aspects' in predictions and 'beauty_aspects' in targets:
            aspect_loss = 0.0
            n_aspects = 0
            
            for aspect in predictions['beauty_aspects']:
                if aspect in targets['beauty_aspects']:
                    loss = self.mse_loss(
                        predictions['beauty_aspects'][aspect],
                        targets['beauty_aspects'][aspect]
                    )
                    aspect_loss += loss
                    n_aspects += 1
            
            if n_aspects > 0:
                aspect_loss /= n_aspects
                total_loss += self.aspect_weight * aspect_loss
                loss_components['aspects'] = aspect_loss.item()
        
        # Design principles loss
        if 'design_principles' in predictions and 'design_principles' in targets:
            principle_loss = 0.0
            n_principles = 0
            
            for principle in predictions['design_principles']:
                if principle in targets['design_principles']:
                    loss = self.mse_loss(
                        predictions['design_principles'][principle],
                        targets['design_principles'][principle]
                    )
                    principle_loss += loss
                    n_principles += 1
            
            if n_principles > 0:
                principle_loss /= n_principles
                total_loss += self.principle_weight * principle_loss
                loss_components['principles'] = principle_loss.item()
        
        # Consistency loss (beauty score should be related to aspects)
        if ('beauty_score' in predictions and 'beauty_aspects' in predictions and
            len(predictions['beauty_aspects']) > 0):
            
            # Average of aspect scores should correlate with beauty score
            aspect_values = torch.stack(list(predictions['beauty_aspects'].values()), dim=1)
            aspect_mean = aspect_values.mean(dim=1, keepdim=True)
            
            consistency_loss = self.l1_loss(predictions['beauty_score'], aspect_mean)
            total_loss += self.consistency_weight * consistency_loss
            loss_components['consistency'] = consistency_loss.item()
        
        loss_components['total'] = total_loss.item()
        return total_loss, loss_components


class BeautyTrainer:
    """
    Trainer for interface beauty evaluation models.
    """
    
    def __init__(
        self,
        model: BeautyPredictor,
        train_loader,
        val_loader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[BeautyLoss] = None,
        device: str = 'cuda',
        log_dir: str = './logs',
        save_dir: str = './checkpoints',
        eval_interval: int = 5,
        save_interval: int = 10,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Beauty prediction model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            loss_fn: Loss function (defaults to BeautyLoss)
            device: Device for training
            log_dir: Directory for tensorboard logs
            save_dir: Directory for saving checkpoints
            eval_interval: Epochs between validation evaluations
            save_interval: Epochs between checkpoint saves
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or BeautyLoss()
        self.device = device
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # Setup logging and saving
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir) if SummaryWriter else None
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_pearson': [],
            'learning_rate': [],
        }
        
        # Evaluator for validation
        self.evaluator = InterfaceEvaluator(model, device)
        
    def train(self, num_epochs: int):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch()
            
            # Log training metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Learning_Rate', 
                                     self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validation phase
            if (epoch + 1) % self.eval_interval == 0:
                val_metrics = self._validate_epoch()
                
                val_loss = val_metrics['val_loss']
                val_pearson = val_metrics['val_pearson']
                
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_pearson'].append(val_pearson)
                
                if self.writer:
                    self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                    self.writer.add_scalar('Metrics/Pearson', val_pearson, epoch)
                
                print(f"         Val Loss: {val_loss:.4f}, Val Pearson: {val_pearson:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint('best_model.pth', is_best=True)
                    print(f"         New best model saved!")
            
            # Save regular checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if 'val_loss' in locals() else train_loss)
                else:
                    self.scheduler.step()
        
        # Final save
        self._save_checkpoint('final_model.pth')
        self._save_training_history()
        
        print("Training completed!")
        if self.writer:
            self.writer.close()
    
    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        
        for batch in progress_bar:
            # Move data to device
            images = batch['image'].to(self.device)
            
            # Prepare targets
            targets = {}
            if 'beauty_score' in batch:
                targets['beauty_score'] = batch['beauty_score'].to(self.device)
            if 'beauty_aspects' in batch:
                targets['beauty_aspects'] = {
                    k: v.to(self.device) for k, v in batch['beauty_aspects'].items()
                }
            if 'design_principles' in batch:
                targets['design_principles'] = {
                    k: v.to(self.device) for k, v in batch['design_principles'].items()
                }
            
            # Forward pass
            outputs = self.model(
                images,
                return_aspects=True,
                return_principles=True,
            )
            
            # Prepare predictions
            predictions = {}
            if len(outputs) >= 1:
                predictions['beauty_score'] = outputs[0].squeeze()
            if len(outputs) >= 2 and outputs[1]:
                predictions['beauty_aspects'] = outputs[1]
            if len(outputs) >= 3 and outputs[2]:
                predictions['design_principles'] = outputs[2]
            
            # Compute loss
            loss, loss_components = self.loss_fn(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        return total_loss / num_batches
    
    def _validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                images = batch['image'].to(self.device)
                
                # Prepare targets
                targets = {}
                if 'beauty_score' in batch:
                    targets['beauty_score'] = batch['beauty_score'].to(self.device)
                    all_targets.extend(batch['beauty_score'].cpu().numpy())
                if 'beauty_aspects' in batch:
                    targets['beauty_aspects'] = {
                        k: v.to(self.device) for k, v in batch['beauty_aspects'].items()
                    }
                if 'design_principles' in batch:
                    targets['design_principles'] = {
                        k: v.to(self.device) for k, v in batch['design_principles'].items()
                    }
                
                # Forward pass
                outputs = self.model(
                    images,
                    return_aspects=True,
                    return_principles=True,
                )
                
                # Prepare predictions
                predictions = {}
                if len(outputs) >= 1:
                    predictions['beauty_score'] = outputs[0].squeeze()
                    pred_scores = torch.sigmoid(outputs[0]).squeeze().cpu().numpy()
                    all_predictions.extend(pred_scores if pred_scores.ndim > 0 else [pred_scores])
                if len(outputs) >= 2 and outputs[1]:
                    predictions['beauty_aspects'] = outputs[1]
                if len(outputs) >= 3 and outputs[2]:
                    predictions['design_principles'] = outputs[2]
                
                # Compute loss
                loss, _ = self.loss_fn(predictions, targets)
                total_loss += loss.item()
                num_batches += 1
        
        # Calculate correlation
        val_pearson = 0.0
        if len(all_predictions) > 0 and len(all_targets) > 0:
            from scipy.stats import pearsonr
            val_pearson = pearsonr(all_targets, all_predictions)[0]
        
        return {
            'val_loss': total_loss / num_batches,
            'val_pearson': val_pearson,
        }
    
    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            # Also save as best.pth
            best_path = os.path.join(self.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)
    
    def _save_training_history(self):
        """Save training history as JSON."""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def evaluate_on_test(self, test_loader, save_dir: str):
        """Evaluate model on test set."""
        print("Evaluating on test set...")
        results = self.evaluator.evaluate_dataset(test_loader, save_dir)
        
        # Print summary
        print("\nTest Results:")
        print(f"RMSE: {results['overall']['rmse']:.4f}")
        print(f"MAE: {results['overall']['mae']:.4f}")
        print(f"Pearson: {results['overall']['pearson']:.4f}")
        print(f"Spearman: {results['overall']['spearman']:.4f}")
        
        return results


def create_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Create optimizer from configuration."""
    optimizer_type = config.get('type', 'adamw')
    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    
    if optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(optimizer: optim.Optimizer, config: Dict) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration."""
    if not config:
        return None
    
    scheduler_type = config.get('type', 'none')
    
    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = config.get('T_max', 100)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'plateau':
        patience = config.get('patience', 10)
        factor = config.get('factor', 0.5)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=patience, factor=factor, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")