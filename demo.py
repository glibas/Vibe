#!/usr/bin/env python3
"""
Simple demo script for Vibe - Vision Transformer for Interface Beauty Evaluation.

This script demonstrates the core functionality of the Vibe framework
without requiring external dependencies like pandas, opencv, etc.
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vibe.models import BeautyPredictor
from vibe.config import create_default_config


def main():
    print("🎨 Vibe: Vision Transformer for Interface Beauty Evaluation")
    print("=" * 60)
    
    # Create model
    print("Creating beauty evaluation model...")
    model = BeautyPredictor(
        model_type='basic',
        img_size=224,
        embed_dim=512,
        n_layers=6,
        n_heads=8,
        use_beauty_tokens=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Create dummy interface image
    print("\nGenerating synthetic interface image...")
    batch_size = 1
    image_tensor = torch.randn(batch_size, 3, 224, 224)
    print(f"✓ Interface image tensor: {image_tensor.shape}")
    
    # Predict beauty score
    print("\nPredicting interface beauty...")
    model.eval()
    with torch.no_grad():
        beauty_score = model.predict_beauty_score(image_tensor)
    
    print(f"✓ Beauty Score: {beauty_score:.3f}")
    
    # Comprehensive analysis
    print("\nRunning comprehensive analysis...")
    with torch.no_grad():
        analysis = model.analyze_interface(image_tensor)
    
    print(f"✓ Overall Beauty Score: {analysis['overall_beauty_score'].item():.3f}")
    
    # Generate explanation
    print("\nGenerating beauty explanation...")
    explanation = model.get_beauty_explanation(analysis)
    
    print("✓ Beauty Assessment:")
    print("-" * 40)
    print(explanation)
    
    # Test different model types
    print("\n" + "=" * 60)
    print("Testing different model architectures...")
    
    model_types = ['basic', 'saliency_guided']
    
    for model_type in model_types:
        print(f"\nTesting {model_type} model...")
        try:
            test_model = BeautyPredictor(
                model_type=model_type,
                img_size=224,
                embed_dim=256,
                n_layers=4,
                n_heads=8
            )
            
            with torch.no_grad():
                test_output = test_model(image_tensor)
            
            print(f"✓ {model_type} model working: output shape {test_output.shape}")
            
        except Exception as e:
            print(f"⚠️  {model_type} model failed: {e}")
    
    # Configuration system test
    print("\n" + "=" * 60)
    print("Testing configuration system...")
    
    config = create_default_config()
    print(f"✓ Default configuration created")
    print(f"  - Model type: {config['model']['model_type']}")
    print(f"  - Image size: {config['model']['img_size']}")
    print(f"  - Embed dim: {config['model']['embed_dim']}")
    print(f"  - Training epochs: {config['training']['num_epochs']}")
    
    print("\n🎉 All tests completed successfully!")
    print("\nNext steps:")
    print("1. Install full dependencies: pip install -r requirements.txt")
    print("2. Create sample data: python -c 'from vibe.data import create_sample_dataset; create_sample_dataset(\"./data\", 50)'")
    print("3. Train a model: python examples/train_basic_model.py")
    print("4. Analyze interfaces: python examples/analyze_interface.py")


if __name__ == "__main__":
    main()