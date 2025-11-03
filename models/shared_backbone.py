"""
Model Architecture A: Shared Backbone with Separate Task Heads.

This architecture uses a shared CRNN backbone to extract common features,
followed by separate task-specific heads for AEC and VAD.
"""

import torch
import torch.nn as nn
from typing import Dict

from models.base import (
    MultiTaskModel,
    FeatureExtractor,
    CRNNBackbone,
    AECHead,
    VADHead,
)


class SharedBackboneModel(MultiTaskModel):
    """
    Multi-task model with shared backbone.
    
    Architecture:
        Input -> Feature Extractor -> Shared CRNN Backbone -> Task Heads
                                                            |-> AEC Head
                                                            |-> VAD Head
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
        Initialize shared backbone model.
        
        Args:
            n_freqs: Number of frequency bins (STFT)
            aec_channels: Number of input channels for AEC (mic + far-end)
            vad_channels: Number of input channels for VAD
            feature_channels: Output channels of feature extractor
            conv_channels: Channels in CRNN conv layers
            lstm_hidden: LSTM hidden size
            num_conv_layers: Number of conv layers in backbone
            num_lstm_layers: Number of LSTM layers in backbone
        """
        super(SharedBackboneModel, self).__init__()
        
        self.n_freqs = n_freqs
        
        # Separate feature extractors for each task (since input channels differ)
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
        
        # Shared CRNN backbone
        self.shared_backbone = CRNNBackbone(
            in_channels=feature_channels,
            conv_channels=conv_channels,
            lstm_hidden=lstm_hidden,
            num_conv_layers=num_conv_layers,
            num_lstm_layers=num_lstm_layers,
            n_freqs=n_freqs,
        )
        
        # Task-specific heads
        backbone_output_dim = lstm_hidden * 2  # Bidirectional LSTM
        
        self.aec_head = AECHead(
            in_features=backbone_output_dim,
            n_freqs=n_freqs,
            hidden_dim=256,
        )
        
        self.vad_head = VADHead(
            in_features=backbone_output_dim,
            hidden_dim=128,
        )
    
    def forward(self, x: torch.Tensor, task: str = 'aec') -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, channels, freq, time)
            task: Task identifier ('aec' or 'vad')
        
        Returns:
            Dictionary containing task-specific outputs
        """
        # Extract features based on task
        if task == 'aec':
            features = self.aec_feature_extractor(x)
        elif task == 'vad':
            features = self.vad_feature_extractor(x)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Shared backbone processing
        backbone_output = self.shared_backbone(features)
        
        # Task-specific head
        if task == 'aec':
            output_mag = self.aec_head(backbone_output)
            return {'magnitude': output_mag}
        else:  # task == 'vad'
            output_logits = self.vad_head(backbone_output)
            return {'logits': output_logits}
    
    def forward_both_tasks(self, 
                           aec_input: torch.Tensor,
                           vad_input: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass for both tasks simultaneously (for evaluation).
        
        Args:
            aec_input: AEC input tensor, shape (batch, 2, freq, time)
            vad_input: VAD input tensor, shape (batch, 1, freq, time)
        
        Returns:
            Dictionary with both task outputs
        """
        return {
            'aec': self.forward(aec_input, task='aec'),
            'vad': self.forward(vad_input, task='vad'),
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
            'shared_backbone': sum(p.numel() for p in self.shared_backbone.parameters()),
            'aec_head': sum(p.numel() for p in self.aec_head.parameters()),
            'vad_head': sum(p.numel() for p in self.vad_head.parameters()),
        }
