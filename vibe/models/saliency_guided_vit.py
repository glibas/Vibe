"""
Saliency-guided Vision Transformer for interface beauty evaluation.

This module implements a Vision Transformer that incorporates saliency maps
to focus attention on visually important regions of web interfaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np

from .vision_transformer import VibeTransformer, PatchEmbedding, TransformerBlock

# Handle optional cv2 import
try:
    import cv2
except ImportError:
    cv2 = None


class SaliencyExtractor(nn.Module):
    """Extract saliency maps from interface images."""
    
    def __init__(self, method='grad_cam', feature_extractor='resnet50'):
        super().__init__()
        self.method = method
        
        if feature_extractor == 'resnet50':
            import torchvision.models as models
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Identity()  # Remove classification head
            self.feature_dim = 2048
        
        # Freeze backbone weights for saliency extraction
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Saliency prediction head
        self.saliency_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((14, 14)),  # Match patch grid
            nn.Conv2d(self.feature_dim, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Extract saliency maps from input images.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Saliency maps (B, 1, 14, 14) matching patch grid
        """
        # Extract features
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        # Generate saliency map
        saliency = self.saliency_head(features)
        return saliency


class SaliencyGuidedAttention(nn.Module):
    """Multi-head attention with saliency guidance."""
    
    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1, saliency_weight=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.saliency_weight = saliency_weight
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Saliency integration layer
        self.saliency_proj = nn.Linear(1, self.head_dim)
        
    def forward(self, x, saliency_map=None, mask=None):
        """
        Forward pass with saliency guidance.
        
        Args:
            x: Input tokens (B, N, E)
            saliency_map: Saliency map (B, 1, H, W) 
            mask: Attention mask
            
        Returns:
            Output tokens and attention weights
        """
        B, N, E = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply saliency guidance if provided
        if saliency_map is not None:
            # Flatten saliency map to match patch tokens (excluding cls token)
            sal_flat = saliency_map.flatten(2).transpose(1, 2)  # (B, H*W, 1)
            
            # Add zero saliency for cls token
            cls_sal = torch.zeros(B, 1, 1, device=sal_flat.device)
            sal_tokens = torch.cat([cls_sal, sal_flat], dim=1)  # (B, N, 1)
            
            # Project saliency to head dimension
            sal_proj = self.saliency_proj(sal_tokens)  # (B, N, head_dim)
            sal_proj = sal_proj.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            
            # Modify attention with saliency
            sal_attn = torch.matmul(sal_proj, sal_proj.transpose(-2, -1))
            attn = attn + self.saliency_weight * sal_attn
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, E)
        out = self.out_proj(out)
        
        return out, attn


class SaliencyGuidedTransformerBlock(nn.Module):
    """Transformer block with saliency-guided attention."""
    
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4.0, dropout=0.1, saliency_weight=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SaliencyGuidedAttention(embed_dim, n_heads, dropout, saliency_weight)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, saliency_map=None, mask=None):
        # Saliency-guided attention with residual connection
        attn_out, attn_weights = self.attn(self.norm1(x), saliency_map, mask)
        x = x + attn_out
        
        # MLP with residual connection
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        return x, attn_weights


class SaliencyGuidedVisionTransformer(nn.Module):
    """
    Vision Transformer with saliency guidance for interface beauty evaluation.
    
    This model incorporates saliency maps to guide attention towards
    visually important regions in web interfaces.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        n_classes: int = 1,
        dropout: float = 0.1,
        saliency_weight: float = 0.5,
        use_beauty_tokens: bool = True,
        pretrained_saliency: bool = True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_beauty_tokens = use_beauty_tokens
        self.saliency_weight = saliency_weight
        
        # Saliency extractor
        self.saliency_extractor = SaliencyExtractor()
        if not pretrained_saliency:
            # Allow saliency extractor to be trained
            for param in self.saliency_extractor.parameters():
                param.requires_grad = True
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Beauty tokens for aesthetic aspects
        if use_beauty_tokens:
            self.beauty_tokens = nn.Parameter(torch.randn(1, 4, embed_dim))
            self.token_names = ['color', 'layout', 'typography', 'balance']
        
        # Positional embeddings
        n_tokens = 1 + self.n_patches + (4 if use_beauty_tokens else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, embed_dim))
        
        # Saliency-guided transformer blocks
        self.blocks = nn.ModuleList([
            SaliencyGuidedTransformerBlock(
                embed_dim, n_heads, mlp_ratio, dropout, saliency_weight
            ) for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification heads
        self.head = nn.Linear(embed_dim, n_classes)
        
        if use_beauty_tokens:
            self.beauty_heads = nn.ModuleList([
                nn.Linear(embed_dim, 1) for _ in range(4)
            ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
                
    def forward(
        self, 
        x, 
        return_attention=False, 
        return_beauty_aspects=False,
        return_saliency=False
    ):
        """
        Forward pass with saliency guidance.
        
        Args:
            x: Input images (B, C, H, W)
            return_attention: Whether to return attention weights
            return_beauty_aspects: Whether to return beauty aspect scores
            return_saliency: Whether to return saliency maps
            
        Returns:
            Beauty score and optionally other outputs
        """
        B = x.shape[0]
        
        # Extract saliency maps
        saliency_map = self.saliency_extractor(x)  # (B, 1, 14, 14)
        
        # Patch embedding
        patch_tokens = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        
        # Add beauty tokens if enabled
        if self.use_beauty_tokens:
            beauty_tokens = self.beauty_tokens.expand(B, -1, -1)
            tokens = torch.cat([tokens, beauty_tokens], dim=1)
        
        # Add positional embeddings
        tokens = tokens + self.pos_embed
        tokens = self.dropout(tokens)
        
        # Pass through saliency-guided transformer blocks
        attention_weights = []
        for block in self.blocks:
            tokens, attn = block(tokens, saliency_map)
            if return_attention:
                attention_weights.append(attn)
        
        tokens = self.norm(tokens)
        
        # Extract outputs
        cls_output = tokens[:, 0]  # Class token
        beauty_score = self.head(cls_output)
        
        outputs = [beauty_score]
        
        # Beauty aspect scores
        if self.use_beauty_tokens and return_beauty_aspects:
            beauty_outputs = tokens[:, -4:]  # Last 4 tokens are beauty tokens
            aspect_scores = {}
            for i, (head, name) in enumerate(zip(self.beauty_heads, self.token_names)):
                aspect_scores[name] = head(beauty_outputs[:, i])
            outputs.append(aspect_scores)
        
        # Attention weights
        if return_attention:
            outputs.append(attention_weights)
            
        # Saliency maps
        if return_saliency:
            outputs.append(saliency_map)
            
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
        
    def get_saliency_visualization(self, x):
        """
        Get saliency visualization for input images.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Saliency maps resized to input resolution
        """
        with torch.no_grad():
            saliency = self.saliency_extractor(x)
            # Resize to input resolution
            saliency_viz = F.interpolate(
                saliency, size=(x.shape[2], x.shape[3]), 
                mode='bilinear', align_corners=False
            )
            return saliency_viz