"""
Dataset and data loading utilities for interface beauty evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union
import json

# Handle optional pandas import
try:
    import pandas as pd
except ImportError:
    pd = None


class InterfaceDataset(Dataset):
    """
    Dataset for interface beauty evaluation.
    
    Supports loading interface images with beauty scores and optional
    beauty aspect annotations.
    """
    
    def __init__(
        self,
        data_path: str,
        annotations_file: str,
        img_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        include_aspects: bool = True,
        include_principles: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to directory containing interface images
            annotations_file: Path to CSV/JSON file with annotations
            img_size: Size to resize images to
            transform: Optional transforms to apply
            include_aspects: Whether to load beauty aspect scores
            include_principles: Whether to load design principle scores
        """
        self.data_path = data_path
        self.img_size = img_size
        self.include_aspects = include_aspects
        self.include_principles = include_principles
        
        # Load annotations
        if pd is None:
            raise ImportError("pandas is required for loading annotations. Install with: pip install pandas")
            
        if annotations_file.endswith('.csv'):
            self.annotations = pd.read_csv(annotations_file)
        elif annotations_file.endswith('.json'):
            with open(annotations_file, 'r') as f:
                annotations_data = json.load(f)
            self.annotations = pd.DataFrame(annotations_data)
        else:
            raise ValueError("Annotations file must be CSV or JSON")
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Validate required columns
        required_cols = ['image_path', 'beauty_score']
        for col in required_cols:
            if col not in self.annotations.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Beauty aspects columns
        self.aspect_columns = ['color_score', 'layout_score', 'typography_score', 'balance_score']
        if include_aspects:
            missing_aspects = [col for col in self.aspect_columns if col not in self.annotations.columns]
            if missing_aspects:
                print(f"Warning: Missing aspect columns: {missing_aspects}")
                self.include_aspects = False
        
        # Design principles columns  
        self.principle_columns = ['symmetry_score', 'contrast_score', 'hierarchy_score', 
                                 'alignment_score', 'whitespace_score']
        if include_principles:
            missing_principles = [col for col in self.principle_columns if col not in self.annotations.columns]
            if missing_principles:
                print(f"Warning: Missing principle columns: {missing_principles}")
                self.include_principles = False
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        row = self.annotations.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.data_path, row['image_path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (self.img_size, self.img_size), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Prepare sample
        sample = {
            'image': image,
            'beauty_score': torch.tensor(row['beauty_score'], dtype=torch.float32),
            'image_path': row['image_path'],
        }
        
        # Add beauty aspects if available
        if self.include_aspects:
            aspects = {}
            for col in self.aspect_columns:
                if col in row and not pd.isna(row[col]):
                    aspect_name = col.replace('_score', '')
                    aspects[aspect_name] = torch.tensor(row[col], dtype=torch.float32)
            if aspects:
                sample['beauty_aspects'] = aspects
        
        # Add design principles if available
        if self.include_principles:
            principles = {}
            for col in self.principle_columns:
                if col in row and not pd.isna(row[col]):
                    principle_name = col.replace('_score', '')
                    principles[principle_name] = torch.tensor(row[col], dtype=torch.float32)
            if principles:
                sample['design_principles'] = principles
        
        # Add metadata if available
        metadata_cols = ['interface_type', 'domain', 'complexity_score', 'resolution']
        metadata = {}
        for col in metadata_cols:
            if col in row and not pd.isna(row[col]):
                metadata[col] = row[col]
        if metadata:
            sample['metadata'] = metadata
        
        return sample


class InterfaceDataLoader:
    """Convenience wrapper for creating data loaders."""
    
    @staticmethod
    def create_train_loader(
        data_path: str,
        annotations_file: str,
        batch_size: int = 32,
        img_size: int = 224,
        num_workers: int = 4,
        pin_memory: bool = True,
        augmentation: bool = True,
    ) -> DataLoader:
        """Create training data loader with augmentation."""
        
        if augmentation:
            transform = transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                # Add interface-specific augmentations
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        dataset = InterfaceDataset(
            data_path=data_path,
            annotations_file=annotations_file,
            img_size=img_size,
            transform=transform,
            include_aspects=True,
            include_principles=True,
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    
    @staticmethod
    def create_val_loader(
        data_path: str,
        annotations_file: str,
        batch_size: int = 32,
        img_size: int = 224,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Create validation data loader without augmentation."""
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        dataset = InterfaceDataset(
            data_path=data_path,
            annotations_file=annotations_file,
            img_size=img_size,
            transform=transform,
            include_aspects=True,
            include_principles=True,
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    
    @staticmethod
    def create_test_loader(
        data_path: str,
        annotations_file: str,
        batch_size: int = 1,
        img_size: int = 224,
        num_workers: int = 1,
    ) -> DataLoader:
        """Create test data loader for inference."""
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        dataset = InterfaceDataset(
            data_path=data_path,
            annotations_file=annotations_file,
            img_size=img_size,
            transform=transform,
            include_aspects=False,
            include_principles=False,
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )


def collate_fn(batch):
    """
    Custom collate function for handling variable-length data.
    """
    # Separate different types of data
    images = torch.stack([item['image'] for item in batch])
    beauty_scores = torch.stack([item['beauty_score'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    
    collated = {
        'image': images,
        'beauty_score': beauty_scores,
        'image_path': image_paths,
    }
    
    # Handle beauty aspects
    if 'beauty_aspects' in batch[0]:
        aspects_dict = {}
        aspect_names = batch[0]['beauty_aspects'].keys()
        for aspect in aspect_names:
            aspects_dict[aspect] = torch.stack([
                item['beauty_aspects'][aspect] for item in batch 
                if aspect in item.get('beauty_aspects', {})
            ])
        collated['beauty_aspects'] = aspects_dict
    
    # Handle design principles
    if 'design_principles' in batch[0]:
        principles_dict = {}
        principle_names = batch[0]['design_principles'].keys()
        for principle in principle_names:
            principles_dict[principle] = torch.stack([
                item['design_principles'][principle] for item in batch
                if principle in item.get('design_principles', {})
            ])
        collated['design_principles'] = principles_dict
    
    # Handle metadata
    if 'metadata' in batch[0]:
        metadata_list = [item.get('metadata', {}) for item in batch]
        collated['metadata'] = metadata_list
    
    return collated


def create_sample_dataset(save_dir: str, n_samples: int = 100):
    """
    Create a sample dataset for testing purposes.
    
    Args:
        save_dir: Directory to save sample data
        n_samples: Number of sample images to create
    """
    import os
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    
    # Generate sample interface images
    annotations = []
    
    for i in range(n_samples):
        # Create synthetic interface image
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some interface elements
        # Header
        draw.rectangle([0, 0, 800, 80], fill='#2c3e50')
        draw.text((20, 25), f"Interface {i+1}", fill='white')
        
        # Sidebar
        draw.rectangle([0, 80, 200, 600], fill='#ecf0f1')
        
        # Main content area
        content_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        color = random.choice(content_colors)
        draw.rectangle([220, 100, 780, 300], fill=color)
        
        # Footer
        draw.rectangle([0, 520, 800, 600], fill='#34495e')
        
        # Save image
        img_name = f"interface_{i+1:03d}.png"
        img_path = os.path.join(save_dir, 'images', img_name)
        img.save(img_path)
        
        # Generate random beauty scores
        beauty_score = random.uniform(0.2, 0.9)
        color_score = random.uniform(0.3, 0.8)
        layout_score = random.uniform(0.4, 0.9)
        typography_score = random.uniform(0.3, 0.7)
        balance_score = random.uniform(0.2, 0.8)
        
        # Design principles
        symmetry_score = random.uniform(0.3, 0.8)
        contrast_score = random.uniform(0.4, 0.9)
        hierarchy_score = random.uniform(0.3, 0.8)
        alignment_score = random.uniform(0.5, 0.9)
        whitespace_score = random.uniform(0.2, 0.7)
        
        annotation = {
            'image_path': os.path.join('images', img_name),
            'beauty_score': beauty_score,
            'color_score': color_score,
            'layout_score': layout_score,
            'typography_score': typography_score,
            'balance_score': balance_score,
            'symmetry_score': symmetry_score,
            'contrast_score': contrast_score,
            'hierarchy_score': hierarchy_score,
            'alignment_score': alignment_score,
            'whitespace_score': whitespace_score,
            'interface_type': random.choice(['landing_page', 'dashboard', 'e_commerce', 'blog']),
            'domain': random.choice(['technology', 'business', 'creative', 'educational']),
            'complexity_score': random.uniform(0.1, 0.9),
        }
        
        annotations.append(annotation)
    
    # Save annotations
    if pd is None:
        # Fallback: save as JSON if pandas not available
        with open(os.path.join(save_dir, 'annotations.json'), 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"Sample dataset created with {n_samples} images in {save_dir}")
        return os.path.join(save_dir, 'annotations.json')
    else:
        df = pd.DataFrame(annotations)
        df.to_csv(os.path.join(save_dir, 'annotations.csv'), index=False)
        print(f"Sample dataset created with {n_samples} images in {save_dir}")
        return os.path.join(save_dir, 'annotations.csv')