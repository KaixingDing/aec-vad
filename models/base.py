"""Base classes for multi-task learning models."""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from abc import ABC, abstractmethod


class MultiTaskModel(nn.Module, ABC):
    """
    Abstract base class for multi-task models.
    
    All multi-task models should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self):
        super(MultiTaskModel, self).__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor, task: str) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a specific task.
        
        Args:
            x: Input tensor, shape (batch, channels, freq, time)
            task: Task identifier ('aec' or 'vad')
        
        Returns:
            Dictionary containing task-specific outputs
        """
        pass
    
    @abstractmethod
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        pass


class FeatureExtractor(nn.Module):
    """
    Feature extraction module using convolutional layers.
    
    Extracts time-frequency features from STFT input.
    """
    
    def __init__(self, 
                 in_channels: int = 2,
                 out_channels: int = 64,
                 num_layers: int = 3):
        """
        Initialize feature extractor.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_layers: Number of convolutional layers
        """
        super(FeatureExtractor, self).__init__()
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            next_channels = out_channels if i == num_layers - 1 else out_channels // 2
            layers.extend([
                nn.Conv2d(current_channels, next_channels, 
                         kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True),
            ])
            current_channels = next_channels
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features.
        
        Args:
            x: Input tensor, shape (batch, in_channels, freq, time)
        
        Returns:
            Feature tensor, shape (batch, out_channels, freq, time)
        """
        return self.layers(x)


class CRNNBackbone(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) backbone.
    
    Combines CNN for local feature extraction and LSTM for temporal modeling.
    """
    
    def __init__(self,
                 in_channels: int = 64,
                 conv_channels: int = 128,
                 lstm_hidden: int = 256,
                 num_conv_layers: int = 2,
                 num_lstm_layers: int = 2,
                 n_freqs: int = 257):
        """
        Initialize CRNN backbone.
        
        Args:
            in_channels: Number of input channels from feature extractor
            conv_channels: Number of channels in conv layers
            lstm_hidden: LSTM hidden size
            num_conv_layers: Number of convolutional layers
            num_lstm_layers: Number of LSTM layers
            n_freqs: Number of frequency bins (for LSTM input size calculation)
        """
        super(CRNNBackbone, self).__init__()
        
        # Convolutional layers
        conv_layers = []
        current_channels = in_channels
        
        for i in range(num_conv_layers):
            next_channels = conv_channels
            conv_layers.extend([
                nn.Conv2d(current_channels, next_channels,
                         kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True),
            ])
            current_channels = next_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # LSTM layers (applied along time dimension)
        # Input will be freq * channels after flattening
        lstm_input_size = n_freqs * conv_channels
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        
        self.conv_channels = conv_channels
        self.lstm_hidden = lstm_hidden
        self.n_freqs = n_freqs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, in_channels, freq, time)
        
        Returns:
            Output tensor with temporal features
        """
        # Apply conv layers
        x = self.conv_layers(x)  # (batch, conv_channels, freq, time)
        
        batch, channels, freq, time = x.shape
        
        # Reshape for LSTM: (batch, time, freq * channels)
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, time, channels, freq)
        x = x.view(batch, time, -1)  # (batch, time, freq * channels)
        
        # Apply LSTM
        x, _ = self.lstm(x)  # (batch, time, lstm_hidden * 2)
        
        return x


class AECHead(nn.Module):
    """
    Task head for Acoustic Echo Cancellation.
    
    Outputs magnitude spectrum of enhanced near-end speech.
    """
    
    def __init__(self,
                 in_features: int,
                 n_freqs: int = 257,
                 hidden_dim: int = 256):
        """
        Initialize AEC head.
        
        Args:
            in_features: Number of input features from backbone
            n_freqs: Number of frequency bins
            hidden_dim: Hidden layer dimension
        """
        super(AECHead, self).__init__()
        
        self.n_freqs = n_freqs
        
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_freqs),
            nn.Sigmoid(),  # Output in [0, 1] range (can be scaled later)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict enhanced magnitude spectrum.
        
        Args:
            x: Input features, shape (batch, time, features)
        
        Returns:
            Magnitude spectrum, shape (batch, n_freqs, time)
        """
        batch, time, features = x.shape
        
        # Apply FC layers to each time frame
        x = x.reshape(batch * time, features)
        x = self.fc_layers(x)  # (batch * time, n_freqs)
        x = x.reshape(batch, time, self.n_freqs)
        
        # Transpose to (batch, n_freqs, time)
        x = x.permute(0, 2, 1)
        
        return x


class VADHead(nn.Module):
    """
    Task head for Voice Activity Detection.
    
    Outputs frame-wise speech probability.
    """
    
    def __init__(self,
                 in_features: int,
                 hidden_dim: int = 128):
        """
        Initialize VAD head.
        
        Args:
            in_features: Number of input features from backbone
            hidden_dim: Hidden layer dimension
        """
        super(VADHead, self).__init__()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            # No sigmoid here, will use BCEWithLogitsLoss
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict frame-wise speech probability.
        
        Args:
            x: Input features, shape (batch, time, features)
        
        Returns:
            Logits, shape (batch, time)
        """
        batch, time, features = x.shape
        
        # Apply FC layers to each time frame
        x = x.reshape(batch * time, features)
        x = self.fc_layers(x)  # (batch * time, 1)
        x = x.reshape(batch, time)
        
        return x
