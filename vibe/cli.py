"""
Command Line Interface for Vibe - Vision transformer for Interface Beauty Evaluation.
"""

import click
import torch
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vibe.models import BeautyPredictor
from vibe.data import InterfaceDataLoader, create_sample_dataset
from vibe.training import BeautyTrainer, BeautyLoss, create_optimizer, create_scheduler
from vibe.evaluation import InterfaceEvaluator
from vibe.config import ConfigManager, create_default_config
from vibe.utils import setup_logging, model_logger, eval_logger


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Vibe: Vision transformer for Interface Beauty Evaluation."""
    pass


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to configuration file')
@click.option('--data-path', type=click.Path(exists=True), required=True,
              help='Path to training data directory')
@click.option('--annotations', type=click.Path(exists=True), required=True,
              help='Path to annotations file (CSV or JSON)')
@click.option('--val-data-path', type=click.Path(exists=True),
              help='Path to validation data directory (if different from training)')
@click.option('--val-annotations', type=click.Path(exists=True),
              help='Path to validation annotations file')
@click.option('--output-dir', '-o', default='./outputs',
              help='Output directory for logs and checkpoints')
@click.option('--epochs', '-e', default=100, type=int,
              help='Number of training epochs')
@click.option('--batch-size', '-b', default=32, type=int,
              help='Batch size for training')
@click.option('--learning-rate', '-lr', default=1e-4, type=float,
              help='Learning rate')
@click.option('--model-type', default='saliency_guided',
              type=click.Choice(['basic', 'saliency_guided', 'roi_enhanced']),
              help='Type of model to train')
@click.option('--resume', type=click.Path(exists=True),
              help='Path to checkpoint to resume training from')
@click.option('--device', default='cuda', help='Device to use for training')
def train(config, data_path, annotations, val_data_path, val_annotations,
          output_dir, epochs, batch_size, learning_rate, model_type, resume, device):
    """Train a beauty evaluation model."""
    
    # Setup logging
    logger = setup_logging(log_dir=os.path.join(output_dir, 'logs'))
    logger.info("Starting Vibe training...")
    
    # Load or create configuration
    if config:
        config_dict = ConfigManager.load_config(config)
        model_config, training_config, data_config, experiment_config = (
            ConfigManager.create_configs_from_dict(config_dict)
        )
    else:
        # Create default config and override with CLI arguments
        config_dict = create_default_config()
        config_dict['model']['model_type'] = model_type
        config_dict['training']['num_epochs'] = epochs
        config_dict['training']['batch_size'] = batch_size
        config_dict['training']['learning_rate'] = learning_rate
        config_dict['data']['train_data_path'] = data_path
        config_dict['data']['train_annotations'] = annotations
        config_dict['experiment']['output_dir'] = output_dir
        config_dict['experiment']['device'] = device
        
        if val_data_path:
            config_dict['data']['val_data_path'] = val_data_path
        else:
            config_dict['data']['val_data_path'] = data_path
            
        if val_annotations:
            config_dict['data']['val_annotations'] = val_annotations
        else:
            config_dict['data']['val_annotations'] = annotations
        
        model_config, training_config, data_config, experiment_config = (
            ConfigManager.create_configs_from_dict(config_dict)
        )
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    
    # Save configuration
    ConfigManager.save_config(
        config_dict, os.path.join(output_dir, 'config.yaml')
    )
    
    # Set device
    if not torch.cuda.is_available() and device == 'cuda':
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create model
    model = BeautyPredictor(
        model_type=model_config.model_type,
        img_size=model_config.img_size,
        patch_size=model_config.patch_size,
        embed_dim=model_config.embed_dim,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        mlp_ratio=model_config.mlp_ratio,
        dropout=model_config.dropout,
        use_beauty_tokens=model_config.use_beauty_tokens,
        saliency_weight=model_config.saliency_weight,
        roi_scales=model_config.roi_scales,
        roi_fusion_method=model_config.roi_fusion_method,
    )
    
    model_logger.log_model_info(model)
    
    # Create data loaders
    train_loader = InterfaceDataLoader.create_train_loader(
        data_path=data_config.train_data_path,
        annotations_file=data_config.train_annotations,
        batch_size=training_config.batch_size,
        img_size=data_config.img_size,
        num_workers=data_config.num_workers,
        augmentation=data_config.augmentation,
    )
    
    val_loader = InterfaceDataLoader.create_val_loader(
        data_path=data_config.val_data_path,
        annotations_file=data_config.val_annotations,
        batch_size=training_config.batch_size,
        img_size=data_config.img_size,
        num_workers=data_config.num_workers,
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, {
        'type': training_config.optimizer,
        'lr': training_config.learning_rate,
        'weight_decay': training_config.weight_decay,
    })
    
    scheduler = create_scheduler(optimizer, {
        'type': training_config.scheduler,
        'T_max': training_config.num_epochs,
    })
    
    # Create loss function
    loss_fn = BeautyLoss(
        beauty_weight=training_config.beauty_weight,
        aspect_weight=training_config.aspect_weight,
        principle_weight=training_config.principle_weight,
        consistency_weight=training_config.consistency_weight,
    )
    
    # Create trainer
    trainer = BeautyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        log_dir=os.path.join(output_dir, 'logs'),
        save_dir=os.path.join(output_dir, 'checkpoints'),
        eval_interval=training_config.eval_interval,
        save_interval=training_config.save_interval,
    )
    
    # Resume from checkpoint if provided
    if resume:
        trainer.load_checkpoint(resume)
        logger.info(f"Resumed training from {resume}")
    
    # Start training
    model_logger.log_training_start(config_dict)
    trainer.train(training_config.num_epochs)
    
    logger.info("Training completed!")


@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model checkpoint')
@click.option('--data-path', type=click.Path(exists=True), required=True,
              help='Path to evaluation data directory')
@click.option('--annotations', type=click.Path(exists=True), required=True,
              help='Path to annotations file')
@click.option('--output-dir', '-o', default='./evaluation_results',
              help='Output directory for evaluation results')
@click.option('--batch-size', '-b', default=32, type=int,
              help='Batch size for evaluation')
@click.option('--device', default='cuda', help='Device to use for evaluation')
@click.option('--analysis-samples', default=10, type=int,
              help='Number of samples for detailed analysis')
def evaluate(model_path, data_path, annotations, output_dir, batch_size, device, analysis_samples):
    """Evaluate a trained beauty evaluation model."""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Vibe evaluation...")
    
    # Set device
    if not torch.cuda.is_available() and device == 'cuda':
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model (need to match training configuration)
    # For now, use default configuration - in practice, should save model config
    model = BeautyPredictor(model_type='saliency_guided')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    
    # Create data loader
    test_loader = InterfaceDataLoader.create_val_loader(
        data_path=data_path,
        annotations_file=annotations,
        batch_size=batch_size,
    )
    
    # Create evaluator
    evaluator = InterfaceEvaluator(model, device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    eval_logger.log_evaluation_start("test_set", len(test_loader.dataset))
    results = evaluator.evaluate_dataset(test_loader, output_dir)
    eval_logger.log_evaluation_results(results)
    
    # Run detailed analysis on sample images
    if analysis_samples > 0:
        logger.info(f"Running detailed analysis on {analysis_samples} samples...")
        analysis_dir = os.path.join(output_dir, 'detailed_analysis')
        evaluator.analyze_predictions(test_loader, analysis_samples, analysis_dir)
    
    # Generate summary report
    report = evaluator.get_summary_report()
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write(report)
    
    print("\n" + report)
    logger.info(f"Evaluation completed! Results saved to {output_dir}")


@cli.command()
@click.option('--image-path', type=click.Path(exists=True), required=True,
              help='Path to interface image')
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model checkpoint')
@click.option('--output-dir', '-o', default='./prediction_results',
              help='Output directory for prediction results')
@click.option('--device', default='cuda', help='Device to use for prediction')
@click.option('--save-visualizations', is_flag=True,
              help='Save attention and saliency visualizations')
def predict(image_path, model_path, output_dir, device, save_visualizations):
    """Predict beauty score for a single interface image."""
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Predicting beauty score for {image_path}")
    
    # Set device
    if not torch.cuda.is_available() and device == 'cuda':
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = BeautyPredictor(model_type='saliency_guided')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # Load and preprocess image
    from PIL import Image
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        analysis = model.analyze_interface(image_tensor)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate explanation
    explanation = model.get_beauty_explanation(analysis)
    
    # Save results
    with open(os.path.join(output_dir, 'prediction.txt'), 'w') as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Beauty Score: {analysis['overall_beauty_score']:.3f}\n\n")
        f.write(explanation)
    
    # Print results
    print(f"\nBeauty Score: {analysis['overall_beauty_score']:.3f}")
    print(f"\n{explanation}")
    
    # Save visualizations if requested
    if save_visualizations and 'saliency_maps' in analysis:
        import matplotlib.pyplot as plt
        
        # Save saliency map
        saliency = analysis['saliency_maps'][0, 0]
        plt.figure(figsize=(8, 6))
        plt.imshow(saliency, cmap='hot')
        plt.colorbar()
        plt.title('Saliency Map')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'saliency_map.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations saved")
    
    logger.info(f"Prediction completed! Results saved to {output_dir}")


@cli.command()
@click.option('--save-dir', default='./sample_data',
              help='Directory to save sample dataset')
@click.option('--num-samples', default=100, type=int,
              help='Number of sample images to generate')
def create_sample_data(save_dir, num_samples):
    """Create a sample dataset for testing."""
    
    logger = setup_logging()
    logger.info(f"Creating sample dataset with {num_samples} images...")
    
    annotations_file = create_sample_dataset(save_dir, num_samples)
    
    print(f"Sample dataset created!")
    print(f"Images saved to: {os.path.join(save_dir, 'images')}")
    print(f"Annotations saved to: {annotations_file}")
    print(f"\nTo train on this data, use:")
    print(f"vibe-train --data-path {save_dir} --annotations {annotations_file}")


@cli.command()
@click.option('--save-path', default='./config/default.yaml',
              help='Path to save default configuration')
def create_config(save_path):
    """Create a default configuration file."""
    
    config = create_default_config()
    ConfigManager.save_config(config, save_path)
    
    print(f"Default configuration saved to: {save_path}")
    print("Edit this file and use it with --config option")


@cli.command()
def info():
    """Show information about Vibe."""
    
    print("Vibe: Vision transformer for Interface Beauty Evaluation")
    print("=" * 55)
    print()
    print("A comprehensive framework for predicting web interface aesthetics")
    print("using saliency-guided Vision Transformers with multi-scale ROI selection.")
    print()
    print("Features:")
    print("  • Vision Transformer backbone for interface analysis")
    print("  • Saliency-guided attention mechanism")
    print("  • Multi-scale ROI selection")
    print("  • Beauty aspect evaluation (color, layout, typography, balance)")
    print("  • Design principles assessment")
    print("  • Comprehensive evaluation metrics")
    print()
    print("Model Types:")
    print("  • basic: Standard Vision Transformer")
    print("  • saliency_guided: ViT with saliency guidance")
    print("  • roi_enhanced: ViT with ROI selection and saliency")
    print()
    print("For help with specific commands, use: vibe-<command> --help")


if __name__ == '__main__':
    cli()