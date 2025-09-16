"""
Beauty predictor that combines Vision Transformer, saliency guidance, 
and multi-scale ROI selection for interface aesthetics evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from .vision_transformer import VibeTransformer
from .saliency_guided_vit import SaliencyGuidedVisionTransformer
from .roi_selector import MultiScaleROISelector


class BeautyPredictor(nn.Module):
    """
    Complete beauty prediction model that integrates multiple components
    for comprehensive interface aesthetics evaluation.
    """
    
    def __init__(
        self,
        model_type: str = 'saliency_guided',  # 'basic', 'saliency_guided', 'roi_enhanced'
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        n_classes: int = 1,
        use_beauty_tokens: bool = True,
        saliency_weight: float = 0.5,
        roi_scales: List[int] = [32, 64, 128],
        roi_fusion_method: str = 'attention',  # 'concat', 'attention', 'weighted'
        beauty_aspects: List[str] = ['color', 'layout', 'typography', 'balance'],
    ):
        super().__init__()
        
        self.model_type = model_type
        self.beauty_aspects = beauty_aspects
        self.roi_fusion_method = roi_fusion_method
        
        # Core Vision Transformer
        if model_type == 'basic':
            self.vit = VibeTransformer(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                n_classes=n_classes,
                dropout=dropout,
                use_beauty_tokens=use_beauty_tokens,
            )
        elif model_type in ['saliency_guided', 'roi_enhanced']:
            self.vit = SaliencyGuidedVisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                n_classes=n_classes,
                dropout=dropout,
                saliency_weight=saliency_weight,
                use_beauty_tokens=use_beauty_tokens,
            )
        
        # Multi-scale ROI selector for enhanced models
        if model_type == 'roi_enhanced':
            self.roi_selector = MultiScaleROISelector(
                roi_sizes=roi_scales,
                max_rois_per_scale=8,
            )
            
            # ROI feature fusion
            roi_feature_dim = 128 * len(roi_scales) * 8  # max_rois_per_scale
            
            if roi_fusion_method == 'attention':
                self.roi_attention = nn.MultiheadAttention(
                    embed_dim=128, num_heads=8, dropout=dropout
                )
                self.roi_fusion = nn.Linear(128, embed_dim)
            elif roi_fusion_method == 'concat':
                self.roi_fusion = nn.Sequential(
                    nn.Linear(roi_feature_dim, embed_dim * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            elif roi_fusion_method == 'weighted':
                self.roi_weights = nn.Parameter(torch.ones(len(roi_scales)))
                self.roi_fusion = nn.Linear(128, embed_dim)
        
        # Final prediction head
        final_input_dim = embed_dim
        if model_type == 'roi_enhanced':
            final_input_dim *= 2  # VIT features + ROI features
            
        self.final_head = nn.Sequential(
            nn.Linear(final_input_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, n_classes),
        )
        
        # Beauty aspect regression heads
        if use_beauty_tokens:
            self.aspect_heads = nn.ModuleDict({
                aspect: nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dim // 4, 1),
                ) for aspect in beauty_aspects
            })
        
        # Aesthetic principles scoring
        self.principles_head = nn.ModuleDict({
            'symmetry': nn.Linear(embed_dim, 1),
            'contrast': nn.Linear(embed_dim, 1),
            'hierarchy': nn.Linear(embed_dim, 1),
            'alignment': nn.Linear(embed_dim, 1),
            'whitespace': nn.Linear(embed_dim, 1),
        })
        
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False,
        return_aspects: bool = False,
        return_principles: bool = False,
        return_saliency: bool = False,
    ) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass for beauty prediction.
        
        Args:
            x: Input images (B, C, H, W)
            return_features: Whether to return intermediate features
            return_attention: Whether to return attention weights
            return_aspects: Whether to return beauty aspect scores
            return_principles: Whether to return design principles scores
            return_saliency: Whether to return saliency maps
            
        Returns:
            Beauty scores and optionally other outputs
        """
        # Vision Transformer forward pass
        if self.model_type == 'basic':
            if return_attention:
                vit_output, beauty_aspects, attention_weights = self.vit(
                    x, return_attention=True, return_beauty_aspects=True
                )
            else:
                if return_aspects:
                    vit_output, beauty_aspects = self.vit(
                        x, return_beauty_aspects=True
                    )
                else:
                    vit_output = self.vit(x)
                    beauty_aspects = None
        else:  # saliency_guided or roi_enhanced
            vit_args = {
                'return_attention': return_attention,
                'return_beauty_aspects': return_aspects,
                'return_saliency': return_saliency,
            }
            vit_outputs = self.vit(x, **vit_args)
            
            if isinstance(vit_outputs, tuple):
                vit_output = vit_outputs[0]
                beauty_aspects = vit_outputs[1] if len(vit_outputs) > 1 and return_aspects else None
                attention_weights = vit_outputs[2] if len(vit_outputs) > 2 and return_attention else None
                saliency_maps = vit_outputs[-1] if return_saliency else None
            else:
                vit_output = vit_outputs
                beauty_aspects = None
                attention_weights = None
                saliency_maps = None
        
        # Extract global features from VIT (should be the class token features, not the final output)
        # For basic models, we need to get features before the final classification head
        if hasattr(self.vit, 'norm') and hasattr(self.vit, 'head'):
            # Get features before the final head
            with torch.no_grad():
                # We need to do a forward pass to get the features
                if self.model_type == 'basic':
                    # Patch embedding
                    patch_tokens = self.vit.patch_embed(x)
                    
                    # Add class token
                    B = x.shape[0]
                    cls_tokens = self.vit.cls_token.expand(B, -1, -1)
                    tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
                    
                    # Add beauty tokens if enabled
                    if self.vit.use_beauty_tokens:
                        beauty_tokens = self.vit.beauty_tokens.expand(B, -1, -1)
                        tokens = torch.cat([tokens, beauty_tokens], dim=1)
                    
                    # Add positional embeddings
                    tokens = tokens + self.vit.pos_embed
                    tokens = self.vit.dropout(tokens)
                    
                    # Pass through transformer blocks
                    for block in self.vit.blocks:
                        tokens, _ = block(tokens)
                    
                    tokens = self.vit.norm(tokens)
                    
                    # Extract class token features
                    vit_features = tokens[:, 0]  # Class token features
                else:
                    # For saliency-guided models, use the output as features
                    vit_features = vit_output
        else:
            vit_features = vit_output
        
        # ROI-enhanced processing
        if self.model_type == 'roi_enhanced':
            roi_features = self.roi_selector(x)  # (B, n_rois, feature_dim)
            
            # Fuse ROI features
            if self.roi_fusion_method == 'attention':
                # Use attention to aggregate ROI features
                B, n_rois, feat_dim = roi_features.shape
                roi_features_flat = roi_features.view(-1, feat_dim).unsqueeze(1)  # (B*n_rois, 1, feat_dim)
                
                # Self-attention over ROI features
                attended_rois, _ = self.roi_attention(
                    roi_features_flat, roi_features_flat, roi_features_flat
                )
                attended_rois = attended_rois.view(B, n_rois, feat_dim)
                
                # Global pooling and projection
                pooled_rois = attended_rois.mean(dim=1)  # (B, feat_dim)
                roi_features_final = self.roi_fusion(pooled_rois)
                
            elif self.roi_fusion_method == 'concat':
                # Concatenate all ROI features
                roi_features_flat = roi_features.view(roi_features.shape[0], -1)
                roi_features_final = self.roi_fusion(roi_features_flat)
                
            elif self.roi_fusion_method == 'weighted':
                # Weighted combination of scale-specific features
                n_scales = len(self.roi_weights)
                scale_features = []
                
                for i in range(n_scales):
                    start_idx = i * 8  # max_rois_per_scale
                    end_idx = (i + 1) * 8
                    scale_feat = roi_features[:, start_idx:end_idx].mean(dim=1)
                    scale_features.append(scale_feat * self.roi_weights[i])
                
                combined_rois = torch.stack(scale_features, dim=1).mean(dim=1)
                roi_features_final = self.roi_fusion(combined_rois)
            
            # Combine VIT and ROI features
            combined_features = torch.cat([vit_features, roi_features_final], dim=-1)
        else:
            combined_features = vit_features
        
        # Final beauty score prediction
        beauty_score = self.final_head(combined_features)
        
        # Prepare outputs
        outputs = [beauty_score]
        
        # Beauty aspect scores
        if return_aspects and beauty_aspects is not None:
            aspect_scores = {}
            for aspect in self.beauty_aspects:
                if aspect in beauty_aspects:
                    aspect_scores[aspect] = beauty_aspects[aspect]
            outputs.append(aspect_scores)
        
        # Design principles scores
        if return_principles:
            # Use global VIT features for principles scoring
            global_features = vit_features if self.model_type != 'roi_enhanced' else vit_features
            principles_scores = {}
            for principle, head in self.principles_head.items():
                principles_scores[principle] = head(global_features)
            outputs.append(principles_scores)
        
        # Additional outputs
        if return_features:
            feature_dict = {
                'vit_features': vit_features,
                'combined_features': combined_features,
            }
            if self.model_type == 'roi_enhanced':
                feature_dict['roi_features'] = roi_features
            outputs.append(feature_dict)
        
        if return_attention and attention_weights is not None:
            outputs.append(attention_weights)
            
        if return_saliency and 'saliency_maps' in locals():
            outputs.append(saliency_maps)
        
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
    
    def predict_beauty_score(self, x: torch.Tensor) -> float:
        """
        Predict beauty score for a single image or batch.
        
        Args:
            x: Input image(s) (B, C, H, W) or (C, H, W)
            
        Returns:
            Beauty score(s) between 0 and 1
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        self.eval()
        with torch.no_grad():
            beauty_score = self(x)
            beauty_score = torch.sigmoid(beauty_score)  # Normalize to [0, 1]
            
        return beauty_score.squeeze().cpu().numpy()
    
    def analyze_interface(self, x: torch.Tensor) -> Dict:
        """
        Comprehensive interface analysis including all components.
        
        Args:
            x: Input image (C, H, W) or (B, C, H, W)
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        self.eval()
        with torch.no_grad():
            # Get all outputs
            outputs = self(
                x,
                return_features=True,
                return_attention=True,
                return_aspects=True,
                return_principles=True,
                return_saliency=(self.model_type != 'basic'),
            )
            
            beauty_score = outputs[0]
            aspect_scores = outputs[1] if len(outputs) > 1 else {}
            principles_scores = outputs[2] if len(outputs) > 2 else {}
            features = outputs[3] if len(outputs) > 3 else {}
            attention_weights = outputs[4] if len(outputs) > 4 else None
            saliency_maps = outputs[5] if len(outputs) > 5 else None
        
        # Normalize scores
        beauty_score = torch.sigmoid(beauty_score).cpu().numpy()
        
        for aspect in aspect_scores:
            aspect_scores[aspect] = torch.sigmoid(aspect_scores[aspect]).cpu().numpy()
            
        for principle in principles_scores:
            principles_scores[principle] = torch.sigmoid(principles_scores[principle]).cpu().numpy()
        
        analysis = {
            'overall_beauty_score': beauty_score.item() if beauty_score.ndim == 1 else beauty_score,
            'beauty_aspects': aspect_scores,
            'design_principles': principles_scores,
            'features': features,
        }
        
        if attention_weights is not None:
            analysis['attention_weights'] = attention_weights
            
        if saliency_maps is not None:
            analysis['saliency_maps'] = saliency_maps.cpu().numpy()
        
        return analysis
    
    def get_beauty_explanation(self, analysis: Dict) -> str:
        """
        Generate human-readable explanation of beauty assessment.
        
        Args:
            analysis: Output from analyze_interface method
            
        Returns:
            Text explanation of the beauty assessment
        """
        score = analysis['overall_beauty_score']
        if hasattr(score, 'item'):
            score = score.item()
        
        aspects = analysis.get('beauty_aspects', {})
        principles = analysis.get('design_principles', {})
        
        # Overall assessment
        if score > 0.8:
            overall = "This interface has excellent aesthetic appeal."
        elif score > 0.6:
            overall = "This interface has good aesthetic quality."
        elif score > 0.4:
            overall = "This interface has moderate aesthetic appeal."
        elif score > 0.2:
            overall = "This interface has below-average aesthetic quality."
        else:
            overall = "This interface has poor aesthetic appeal."
        
        # Aspect analysis
        aspect_analysis = []
        for aspect, value in aspects.items():
            if hasattr(value, 'item'):
                value = value.item()
            if isinstance(value, (list, tuple)) and len(value) > 0:
                value = value[0]
            if value > 0.7:
                aspect_analysis.append(f"Strong {aspect}")
            elif value > 0.5:
                aspect_analysis.append(f"Good {aspect}")
            elif value > 0.3:
                aspect_analysis.append(f"Moderate {aspect}")
            else:
                aspect_analysis.append(f"Weak {aspect}")
        
        # Principles analysis
        principles_analysis = []
        for principle, value in principles.items():
            if hasattr(value, 'item'):
                value = value.item()
            if isinstance(value, (list, tuple)) and len(value) > 0:
                value = value[0]
            if value > 0.7:
                principles_analysis.append(f"Excellent {principle}")
            elif value > 0.5:
                principles_analysis.append(f"Good {principle}")
            elif value < 0.3:
                principles_analysis.append(f"Poor {principle}")
        
        explanation = f"{overall}\n\n"
        
        if aspect_analysis:
            explanation += f"Beauty Aspects: {', '.join(aspect_analysis)}.\n"
            
        if principles_analysis:
            explanation += f"Design Principles: {', '.join(principles_analysis)}.\n"
        
        # Recommendations
        weak_aspects = []
        weak_principles = []
        
        for asp, val in aspects.items():
            if hasattr(val, 'item'):
                val = val.item()
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            if val < 0.4:
                weak_aspects.append(asp)
                
        for pri, val in principles.items():
            if hasattr(val, 'item'):
                val = val.item()
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            if val < 0.4:
                weak_principles.append(pri)
        
        if weak_aspects or weak_principles:
            explanation += "\nRecommendations for improvement:\n"
            for aspect in weak_aspects:
                explanation += f"- Enhance {aspect} design\n"
            for principle in weak_principles:
                explanation += f"- Improve {principle}\n"
        
        return explanation