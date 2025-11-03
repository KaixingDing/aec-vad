"""
Model Architecture B: Hierarchical/Progressive Multi-Task Model.

This architecture first performs VAD to identify speech regions, then uses
VAD information to guide the AEC task through attention mechanisms.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from models.base import (
    MultiTaskModel,
    FeatureExtractor,
    CRNNBackbone,
    AECHead,
    VADHead,
)


class AttentionGate(nn.Module):
    """
    Attention gate that uses VAD information to modulate AEC features.
    """
    
    def __init__(self, feature_dim: int):
        """
        Initialize attention gate.
        
        Args:
            feature_dim: Dimension of features to modulate
        """
        super(AttentionGate, self).__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(1, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, features: torch.Tensor, vad_probs: torch.Tensor) -> torch.Tensor:
        """
        Apply attention gating.
        
        Args:
            features: Feature tensor, shape (batch, time, feature_dim)
            vad_probs: VAD probabilities, shape (batch, time)
        
        Returns:
            Modulated features, shape (batch, time, feature_dim)
        """
        # Expand VAD probs to match feature dimension
        vad_probs = vad_probs.unsqueeze(-1)  # (batch, time, 1)
        
        # Compute attention weights
        attention_weights = self.gate(vad_probs)  # (batch, time, feature_dim)
        
        # Apply gating
        return features * attention_weights


class HierarchicalModel(MultiTaskModel):
    """
    Hierarchical multi-task model.
    
    Architecture:
        Input -> Feature Extractor -> VAD Branch (early) -> VAD Head
                                    |
                                    -> AEC Branch (with VAD guidance) -> AEC Head
    
    The model first performs coarse-grained VAD, then uses VAD predictions
    to guide the AEC task via attention mechanisms.
    """
    
    def __init__(self,
                 n_freqs: int = 257,
                 aec_channels: int = 2,
                 vad_channels: int = 1,
                 feature_channels: int = 64,
                 conv_channels: int = 128,
                 lstm_hidden: int = 256,
                 num_conv_layers: int = 2,
                 num_lstm_layers: int = 2):
        """
        Initialize hierarchical model.
        
        Args:
            n_freqs: Number of frequency bins (STFT)
            aec_channels: Number of input channels for AEC
            vad_channels: Number of input channels for VAD
            feature_channels: Output channels of feature extractor
            conv_channels: Channels in CRNN conv layers
            lstm_hidden: LSTM hidden size
            num_conv_layers: Number of conv layers
            num_lstm_layers: Number of LSTM layers
        """
        super(HierarchicalModel, self).__init__()
        
        self.n_freqs = n_freqs
        
        # Feature extractors
        self.aec_feature_extractor = FeatureExtractor(
            in_channels=aec_channels,
            out_channels=feature_channels,
            num_layers=3,
        )
        
        self.vad_feature_extractor = FeatureExtractor(
            in_channels=vad_channels,
            out_channels=feature_channels,
            num_layers=3,
        )
        
        # VAD branch (processes features early)
        self.vad_backbone = CRNNBackbone(
            in_channels=feature_channels,
            conv_channels=conv_channels // 2,  # Smaller for VAD
            lstm_hidden=lstm_hidden // 2,
            num_conv_layers=num_conv_layers,
            num_lstm_layers=num_lstm_layers,
            n_freqs=n_freqs,
        )
        
        vad_backbone_output_dim = lstm_hidden  # Bidirectional LSTM
        
        self.vad_head = VADHead(
            in_features=vad_backbone_output_dim,
            hidden_dim=128,
        )
        
        # AEC branch (processes features with VAD guidance)
        self.aec_backbone = CRNNBackbone(
            in_channels=feature_channels,
            conv_channels=conv_channels,
            lstm_hidden=lstm_hidden,
            num_conv_layers=num_conv_layers,
            num_lstm_layers=num_lstm_layers,
            n_freqs=n_freqs,
        )
        
        aec_backbone_output_dim = lstm_hidden * 2  # Bidirectional LSTM
        
        # Attention gate to modulate AEC features with VAD
        self.attention_gate = AttentionGate(feature_dim=aec_backbone_output_dim)
        
        self.aec_head = AECHead(
            in_features=aec_backbone_output_dim,
            n_freqs=n_freqs,
            hidden_dim=256,
        )
    
    def forward(self, x: torch.Tensor, task: str = 'aec',
                vad_guidance: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, channels, freq, time)
            task: Task identifier ('aec' or 'vad')
            vad_guidance: Optional pre-computed VAD probabilities for AEC task
        
        Returns:
            Dictionary containing task-specific outputs
        """
        if task == 'vad':
            # VAD forward pass
            features = self.vad_feature_extractor(x)
            backbone_output = self.vad_backbone(features)
            output_logits = self.vad_head(backbone_output)
            return {'logits': output_logits}
        
        elif task == 'aec':
            # AEC forward pass (potentially with VAD guidance)
            features = self.aec_feature_extractor(x)
            backbone_output = self.aec_backbone(features)
            
            # Apply VAD-guided attention if available
            if vad_guidance is not None:
                # vad_guidance should be probabilities in [0, 1]
                backbone_output = self.attention_gate(backbone_output, vad_guidance)
            
            output_mag = self.aec_head(backbone_output)
            return {'magnitude': output_mag}
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward_with_vad_guidance(self,
                                  aec_input: torch.Tensor,
                                  vad_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for AEC with VAD guidance.
        
        First computes VAD, then uses it to guide AEC.
        
        Args:
            aec_input: AEC input tensor, shape (batch, 2, freq, time)
            vad_input: VAD input tensor, shape (batch, 1, freq, time)
        
        Returns:
            Dictionary with both task outputs
        """
        # First, get VAD predictions
        vad_output = self.forward(vad_input, task='vad')
        vad_logits = vad_output['logits']
        vad_probs = torch.sigmoid(vad_logits)
        
        # Then, use VAD to guide AEC
        aec_output = self.forward(aec_input, task='aec', vad_guidance=vad_probs)
        
        return {
            'aec': aec_output,
            'vad': vad_output,
        }
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, int]:
        """
        Get detailed model information.
        
        Returns:
            Dictionary with parameter counts for each component
        """
        return {
            'total': self.get_num_parameters(),
            'aec_feature_extractor': sum(p.numel() for p in self.aec_feature_extractor.parameters()),
            'vad_feature_extractor': sum(p.numel() for p in self.vad_feature_extractor.parameters()),
            'vad_backbone': sum(p.numel() for p in self.vad_backbone.parameters()),
            'aec_backbone': sum(p.numel() for p in self.aec_backbone.parameters()),
            'attention_gate': sum(p.numel() for p in self.attention_gate.parameters()),
            'aec_head': sum(p.numel() for p in self.aec_head.parameters()),
            'vad_head': sum(p.numel() for p in self.vad_head.parameters()),
        }
