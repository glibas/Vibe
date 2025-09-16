#!/usr/bin/env python3
"""
Example script for training a basic Vibe model.

This script demonstrates how to train a Vision Transformer model
for interface beauty evaluation using the Vibe framework.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vibe.models import BeautyPredictor
from vibe.data import InterfaceDataLoader, create_sample_dataset
from vibe.training import BeautyTrainer, BeautyLoss, create_optimizer, create_scheduler
from vibe.config import create_default_config, ConfigManager
from vibe.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description='Train Vibe model')
    parser.add_argument('--data-dir', default='./sample_data', 
                       help='Directory containing training data')
    parser.add_argument('--output-dir', default='./outputs',
                       help='Output directory for logs and checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--model-type', default='saliency_guided',
                       choices=['basic', 'saliency_guided', 'roi_enhanced'],
                       help='Model type to train')
    parser.add_argument('--create-sample-data', action='store_true',
                       help='Create sample dataset if it does not exist')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir=os.path.join(args.output_dir, 'logs'))
    logger.info("Starting Vibe training example...")
    
    # Create sample data if requested or if data directory doesn't exist
    annotations_file = os.path.join(args.data_dir, 'annotations.csv')
    if args.create_sample_data or not os.path.exists(annotations_file):
        logger.info("Creating sample dataset...")
        create_sample_dataset(args.data_dir, n_samples=200)
        logger.info(f"Sample dataset created in {args.data_dir}")
    
    # Create configuration
    config_dict = create_default_config()
    config_dict['model']['model_type'] = args.model_type
    config_dict['training']['num_epochs'] = args.epochs
    config_dict['training']['batch_size'] = args.batch_size
    config_dict['training']['learning_rate'] = args.learning_rate
    config_dict['experiment']['output_dir'] = args.output_dir
    
    model_config, training_config, data_config, experiment_config = (
        ConfigManager.create_configs_from_dict(config_dict)
    )
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Save configuration
    ConfigManager.save_config(
        config_dict, os.path.join(args.output_dir, 'config.yaml')
    )
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = BeautyPredictor(
        model_type=model_config.model_type,
        img_size=model_config.img_size,
        embed_dim=model_config.embed_dim,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        use_beauty_tokens=model_config.use_beauty_tokens,
        saliency_weight=model_config.saliency_weight,
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = InterfaceDataLoader.create_train_loader(
        data_path=args.data_dir,
        annotations_file=annotations_file,
        batch_size=args.batch_size,
        augmentation=True,
    )
    
    val_loader = InterfaceDataLoader.create_val_loader(
        data_path=args.data_dir,
        annotations_file=annotations_file,
        batch_size=args.batch_size,
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, {
        'type': 'adamw',
        'lr': args.learning_rate,
        'weight_decay': 0.01,
    })
    
    scheduler = create_scheduler(optimizer, {
        'type': 'cosine',
        'T_max': args.epochs,
    })
    
    # Create loss function
    loss_fn = BeautyLoss()
    
    # Create trainer
    trainer = BeautyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir=os.path.join(args.output_dir, 'logs'),
        save_dir=os.path.join(args.output_dir, 'checkpoints'),
        eval_interval=5,
        save_interval=10,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train(args.epochs)
    
    logger.info("Training completed!")
    logger.info(f"Best model saved in {os.path.join(args.output_dir, 'checkpoints', 'best.pth')}")
    
    # Quick evaluation
    logger.info("Running quick evaluation...")
    eval_results = trainer.evaluate_on_test(
        val_loader, 
        os.path.join(args.output_dir, 'evaluation')
    )
    
    print("\nTraining Summary:")
    print("=" * 50)
    print(f"Model Type: {args.model_type}")
    print(f"Total Epochs: {args.epochs}")
    print(f"Final RMSE: {eval_results['overall']['rmse']:.4f}")
    print(f"Final Pearson: {eval_results['overall']['pearson']:.4f}")
    print(f"Best model: {os.path.join(args.output_dir, 'checkpoints', 'best.pth')}")


if __name__ == "__main__":
    main()