"""
Vision Transformer implementation for interface beauty evaluation.

This module implements a custom Vision Transformer specifically designed 
for analyzing web interface aesthetics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange
except ImportError:
    from ..utils.einops_fallback import rearrange, repeat, Rearrange
import math
from typing import Optional, Tuple, List


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = rearrange(x, 'b e h w -> b (h w) e')  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, N, E = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        
        return out, attn


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""
    
    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_out, attn_weights = self.attn(self.norm1(x), mask)
        x = x + attn_out
        
        # MLP with residual connection
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        return x, attn_weights


class VibeTransformer(nn.Module):
    """
    Vision Transformer for Interface Beauty Evaluation.
    
    This implementation includes specialized components for analyzing
    web interface aesthetics with attention to design principles.
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
        use_beauty_tokens: bool = True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_beauty_tokens = use_beauty_tokens
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Class token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Beauty-specific tokens for aesthetic analysis
        if use_beauty_tokens:
            self.beauty_tokens = nn.Parameter(torch.randn(1, 4, embed_dim))  # Color, Layout, Typography, Balance
            self.token_names = ['color', 'layout', 'typography', 'balance']
        
        # Positional embeddings
        n_tokens = 1 + self.n_patches + (4 if use_beauty_tokens else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, n_classes)
        
        # Beauty aspect heads (if using beauty tokens)
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
                
    def forward(self, x, return_attention=False, return_beauty_aspects=False):
        """
        Forward pass of the VibeTransformer.
        
        Args:
            x: Input images (B, C, H, W)
            return_attention: Whether to return attention weights
            return_beauty_aspects: Whether to return beauty aspect scores
            
        Returns:
            Beauty score and optionally attention weights and aspect scores
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add beauty tokens if enabled
        if self.use_beauty_tokens:
            beauty_tokens = repeat(self.beauty_tokens, '1 n d -> b n d', b=B)
            x = torch.cat([x, beauty_tokens], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Pass through transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x)
            if return_attention:
                attention_weights.append(attn)
        
        x = self.norm(x)
        
        # Extract outputs
        cls_output = x[:, 0]  # Class token
        beauty_score = self.head(cls_output)
        
        outputs = [beauty_score]
        
        # Beauty aspect scores
        if self.use_beauty_tokens and return_beauty_aspects:
            beauty_outputs = x[:, -4:]  # Last 4 tokens are beauty tokens
            aspect_scores = {}
            for i, (head, name) in enumerate(zip(self.beauty_heads, self.token_names)):
                aspect_scores[name] = head(beauty_outputs[:, i])
            outputs.append(aspect_scores)
        
        # Attention weights
        if return_attention:
            outputs.append(attention_weights)
            
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
        
    def get_attention_maps(self, x, layer_idx=-1):
        """
        Extract attention maps for visualization.
        
        Args:
            x: Input images (B, C, H, W)  
            layer_idx: Transformer layer index (-1 for last layer)
            
        Returns:
            Attention maps (B, n_heads, n_tokens, n_tokens)
        """
        with torch.no_grad():
            _, attention_weights = self.forward(x, return_attention=True)
            return attention_weights[layer_idx]