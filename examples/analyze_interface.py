#!/usr/bin/env python3
"""
Example script for evaluating and analyzing interface beauty.

This script demonstrates how to use a trained Vibe model to:
1. Evaluate interface beauty scores
2. Analyze beauty aspects and design principles
3. Generate explanations and visualizations
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vibe.models import BeautyPredictor
from vibe.data import create_sample_dataset
from vibe.utils import setup_logging


def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with same configuration as training
    model = BeautyPredictor(model_type='saliency_guided')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    return model


def preprocess_image(image_path, img_size=224):
    """Preprocess image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image


def analyze_single_image(model, image_path, output_dir, device='cpu'):
    """Analyze a single interface image."""
    print(f"Analyzing: {image_path}")
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Run analysis
    with torch.no_grad():
        analysis = model.analyze_interface(image_tensor)
    
    # Generate explanation
    explanation = model.get_beauty_explanation(analysis)
    
    # Print results
    print(f"Beauty Score: {analysis['overall_beauty_score']:.3f}")
    print(f"\n{explanation}")
    
    # Create output directory for this image
    image_name = Path(image_path).stem
    image_output_dir = os.path.join(output_dir, f"analysis_{image_name}")
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Save analysis text
    with open(os.path.join(image_output_dir, 'analysis.txt'), 'w') as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Beauty Score: {analysis['overall_beauty_score']:.3f}\n\n")
        f.write(explanation)
    
    print(f"Analysis saved to: {image_output_dir}")
    return analysis


if __name__ == "__main__":
    print("Vibe Interface Analysis Example")
    print("This example requires a trained model to run.")
    print("Please train a model first using train_basic_model.py")