"""
Multi-scale ROI (Region of Interest) selector for interface analysis.

This module implements multi-scale region selection to focus on different
aspects of web interface design at various scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    from einops import rearrange
except ImportError:
    from ..utils.einops_fallback import rearrange

# Handle optional cv2 import
try:
    import cv2
except ImportError:
    cv2 = None


class ROIDetector(nn.Module):
    """Detect regions of interest in interface images."""
    
    def __init__(self, feature_dim=256, n_scales=3):
        super().__init__()
        self.n_scales = n_scales
        
        # Feature pyramid network for multi-scale detection
        self.fpn = FeaturePyramidNetwork(feature_dim)
        
        # ROI proposal network
        self.roi_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, 1),  # bbox coordinates
            nn.Sigmoid()
        )
        
        # Confidence scoring
        self.conf_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        Detect ROIs from feature maps.
        
        Args:
            features: List of feature maps at different scales
            
        Returns:
            ROI proposals and confidence scores
        """
        pyramid_features = self.fpn(features)
        
        rois = []
        confidences = []
        
        for feat in pyramid_features:
            roi = self.roi_head(feat)
            conf = self.conf_head(feat)
            
            rois.append(roi)
            confidences.append(conf)
            
        return rois, confidences


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction."""
    
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(256, feature_dim, 1),  # P2
            nn.Conv2d(512, feature_dim, 1),  # P3  
            nn.Conv2d(1024, feature_dim, 1), # P4
            nn.Conv2d(2048, feature_dim, 1), # P5
        ])
        
        # Top-down pathway
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1) for _ in range(4)
        ])
        
    def forward(self, features):
        """
        Forward pass through FPN.
        
        Args:
            features: List of backbone features [C2, C3, C4, C5]
            
        Returns:
            List of pyramid features [P2, P3, P4, P5]
        """
        # Lateral connections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        # Apply final convolutions
        pyramid_features = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        
        return pyramid_features


class MultiScaleROISelector(nn.Module):
    """
    Multi-scale ROI selector for interface beauty evaluation.
    
    This module identifies and extracts regions of interest at multiple scales
    to analyze different design aspects of web interfaces.
    """
    
    def __init__(
        self,
        backbone_name='resnet50',
        feature_dim=256,
        roi_sizes=[32, 64, 128],
        max_rois_per_scale=8,
        nms_threshold=0.5,
        conf_threshold=0.3,
    ):
        super().__init__()
        
        self.roi_sizes = roi_sizes
        self.max_rois_per_scale = max_rois_per_scale
        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        
        # Backbone for feature extraction
        if backbone_name == 'resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=True)
            self.backbone_layers = nn.ModuleList([
                nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
                backbone.layer1,  # 256 channels
                backbone.layer2,  # 512 channels
                backbone.layer3,  # 1024 channels
                backbone.layer4,  # 2048 channels
            ])
        
        # ROI detector
        self.roi_detector = ROIDetector(feature_dim, len(roi_sizes))
        
        # ROI pooling for feature extraction
        self.roi_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Feature projection for each scale
        self.roi_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
            ) for _ in roi_sizes
        ])
        
    def extract_backbone_features(self, x):
        """Extract multi-scale features from backbone."""
        features = []
        for layer in self.backbone_layers:
            x = layer(x)
            features.append(x)
        
        return features[1:]  # Skip the first layer, return [C2, C3, C4, C5]
        
    def forward(self, x, return_rois=False):
        """
        Forward pass for multi-scale ROI selection.
        
        Args:
            x: Input images (B, C, H, W)
            return_rois: Whether to return ROI coordinates
            
        Returns:
            Multi-scale ROI features and optionally ROI coordinates
        """
        B, C, H, W = x.shape
        
        # Extract backbone features
        backbone_features = self.extract_backbone_features(x)
        
        # Detect ROIs
        roi_proposals, confidences = self.roi_detector(backbone_features)
        
        # Process ROIs at each scale
        scale_features = []
        all_rois = []
        
        for scale_idx, (rois, confs, feat) in enumerate(
            zip(roi_proposals, confidences, backbone_features)
        ):
            # Apply confidence thresholding and NMS
            valid_rois, valid_features = self._process_rois(
                rois, confs, feat, x, scale_idx
            )
            
            scale_features.append(valid_features)
            if return_rois:
                all_rois.append(valid_rois)
        
        # Combine features from all scales
        combined_features = torch.cat(scale_features, dim=1)  # (B, sum(roi_features), feature_dim)
        
        if return_rois:
            return combined_features, all_rois
        else:
            return combined_features
    
    def _process_rois(self, rois, confidences, features, original_img, scale_idx):
        """Process ROIs for a single scale."""
        B, _, fH, fW = features.shape
        _, _, H, W = original_img.shape
        
        # Reshape predictions
        rois = rois.permute(0, 2, 3, 1).reshape(B, -1, 4)  # (B, fH*fW, 4)
        confs = confidences.permute(0, 2, 3, 1).reshape(B, -1)  # (B, fH*fW)
        
        batch_roi_features = []
        batch_roi_coords = []
        
        for b in range(B):
            # Filter by confidence
            valid_mask = confs[b] > self.conf_threshold
            if not valid_mask.any():
                # If no valid ROIs, use top-k ROIs
                _, top_indices = torch.topk(confs[b], min(self.max_rois_per_scale, len(confs[b])))
                valid_mask = torch.zeros_like(confs[b], dtype=torch.bool)
                valid_mask[top_indices] = True
            
            valid_rois = rois[b][valid_mask]
            valid_confs = confs[b][valid_mask]
            
            # Apply NMS
            keep_indices = self._nms(valid_rois, valid_confs, self.nms_threshold)
            final_rois = valid_rois[keep_indices[:self.max_rois_per_scale]]
            
            # Convert to absolute coordinates
            final_rois[:, [0, 2]] *= W  # x coordinates
            final_rois[:, [1, 3]] *= H  # y coordinates
            
            # Extract ROI features
            roi_feats = self._extract_roi_features(
                final_rois, features[b:b+1], scale_idx
            )
            
            batch_roi_features.append(roi_feats)
            batch_roi_coords.append(final_rois)
        
        # Pad to consistent size
        max_rois = max(len(rf) for rf in batch_roi_features)
        padded_features = []
        
        for roi_feats in batch_roi_features:
            if len(roi_feats) < max_rois:
                # Pad with zeros
                padding = torch.zeros(
                    max_rois - len(roi_feats), roi_feats.shape[1],
                    device=roi_feats.device, dtype=roi_feats.dtype
                )
                roi_feats = torch.cat([roi_feats, padding], dim=0)
            padded_features.append(roi_feats)
        
        return torch.stack(padded_features), batch_roi_coords
    
    def _extract_roi_features(self, rois, features, scale_idx):
        """Extract features for ROIs using ROI pooling."""
        roi_features = []
        
        for roi in rois:
            x1, y1, x2, y2 = roi.int()
            
            # Clamp coordinates
            x1 = torch.clamp(x1, 0, features.shape[3] - 1)
            y1 = torch.clamp(y1, 0, features.shape[2] - 1)
            x2 = torch.clamp(x2, x1 + 1, features.shape[3])
            y2 = torch.clamp(y2, y1 + 1, features.shape[2])
            
            # Extract ROI from features
            roi_feat = features[:, :, y1:y2, x1:x2]
            
            # Apply ROI pooling
            pooled_feat = self.roi_pool(roi_feat)
            pooled_feat = pooled_feat.flatten(1)
            
            # Project to final feature dimension
            projected_feat = self.roi_projectors[scale_idx](pooled_feat)
            roi_features.append(projected_feat)
        
        return torch.stack(roi_features).squeeze(1)
    
    def _nms(self, boxes, scores, threshold):
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        
        # Convert to corner format if needed
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Compute areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by scores
        _, order = scores.sort(0, descending=True)
        
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            
            if order.numel() == 1:
                break
                
            # Compute IoU with remaining boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU less than threshold
            inds = torch.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def visualize_rois(self, image, rois_list, save_path=None):
        """
        Visualize detected ROIs on the image.
        
        Args:
            image: Input image tensor (C, H, W)
            rois_list: List of ROI coordinates for each scale
            save_path: Path to save visualization
            
        Returns:
            Visualization image as numpy array
        """
        # Convert to numpy
        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = image.copy()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB colors for different scales
        
        for scale_idx, rois in enumerate(rois_list):
            color = colors[scale_idx % len(colors)]
            
            for roi in rois:
                x1, y1, x2, y2 = roi.int().cpu().numpy()
                cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
                
                # Add scale label
                cv2.putText(
                    img_np, f'S{scale_idx}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        
        return img_np