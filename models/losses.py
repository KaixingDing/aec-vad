"""Loss functions for multi-task learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss.
    
    Used for time-domain or magnitude-domain speech enhancement.
    """
    
    def __init__(self):
        super(SISDRLoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute SI-SDR loss.
        
        Args:
            pred: Predicted signal, shape (batch, ...)
            target: Target signal, shape (batch, ...)
            eps: Small constant for numerical stability
        
        Returns:
            SI-SDR loss (negative SI-SDR)
        """
        # Flatten to (batch, -1)
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        
        # Zero-mean normalization
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        
        # Compute projection
        target_norm = torch.sum(target ** 2, dim=1, keepdim=True) + eps
        proj = (torch.sum(pred * target, dim=1, keepdim=True) / target_norm) * target
        
        # Compute SI-SDR
        noise = pred - proj
        si_sdr = 10 * torch.log10(
            (torch.sum(proj ** 2, dim=1) + eps) / (torch.sum(noise ** 2, dim=1) + eps)
        )
        
        # Return negative SI-SDR as loss (we want to maximize SI-SDR)
        return -si_sdr.mean()


class MagnitudeLoss(nn.Module):
    """
    Magnitude spectrum loss (MSE on magnitude).
    """
    
    def __init__(self):
        super(MagnitudeLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        """
        Compute magnitude loss.
        
        Args:
            pred_mag: Predicted magnitude, shape (batch, freq, time)
            target_mag: Target magnitude, shape (batch, freq, time)
        
        Returns:
            MSE loss on magnitude
        """
        return self.mse(pred_mag, target_mag)


class AECLoss(nn.Module):
    """
    Combined loss for AEC task.
    
    Combines magnitude loss and SI-SDR loss.
    """
    
    def __init__(self, mag_weight: float = 1.0, sisdr_weight: float = 1.0):
        """
        Initialize AEC loss.
        
        Args:
            mag_weight: Weight for magnitude loss
            sisdr_weight: Weight for SI-SDR loss
        """
        super(AECLoss, self).__init__()
        self.mag_loss = MagnitudeLoss()
        self.sisdr_loss = SISDRLoss()
        self.mag_weight = mag_weight
        self.sisdr_weight = sisdr_weight
    
    def forward(self, pred_mag: torch.Tensor, target_mag: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute AEC loss.
        
        Args:
            pred_mag: Predicted magnitude, shape (batch, freq, time)
            target_mag: Target magnitude, shape (batch, freq, time)
        
        Returns:
            Dictionary with loss components and total loss
        """
        mag_loss_val = self.mag_loss(pred_mag, target_mag)
        sisdr_loss_val = self.sisdr_loss(pred_mag, target_mag)
        
        total_loss = self.mag_weight * mag_loss_val + self.sisdr_weight * sisdr_loss_val
        
        return {
            'magnitude_loss': mag_loss_val,
            'sisdr_loss': sisdr_loss_val,
            'total': total_loss,
        }


class VADLoss(nn.Module):
    """
    Loss for VAD task using binary cross-entropy.
    """
    
    def __init__(self):
        super(VADLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute VAD loss.
        
        Args:
            logits: Predicted logits, shape (batch, time)
            labels: Target labels (0 or 1), shape (batch, time)
        
        Returns:
            Binary cross-entropy loss
        """
        # Convert labels to float
        labels = labels.float()
        
        return self.bce_loss(logits, labels)


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss.
    
    L_total = λ_AEC * L_AEC + λ_VAD * L_VAD
    """
    
    def __init__(self,
                 aec_weight: float = 1.0,
                 vad_weight: float = 1.0,
                 mag_weight: float = 1.0,
                 sisdr_weight: float = 1.0):
        """
        Initialize multi-task loss.
        
        Args:
            aec_weight: Weight for AEC task (λ_AEC)
            vad_weight: Weight for VAD task (λ_VAD)
            mag_weight: Weight for magnitude loss within AEC
            sisdr_weight: Weight for SI-SDR loss within AEC
        """
        super(MultiTaskLoss, self).__init__()
        
        self.aec_loss = AECLoss(mag_weight=mag_weight, sisdr_weight=sisdr_weight)
        self.vad_loss = VADLoss()
        
        self.aec_weight = aec_weight
        self.vad_weight = vad_weight
    
    def forward(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            outputs: Dictionary with task outputs
                - 'aec': {'magnitude': predicted magnitude} (optional)
                - 'vad': {'logits': predicted logits} (optional)
            targets: Dictionary with task targets
                - 'aec': {'target_mag': target magnitude} (optional)
                - 'vad': {'target_labels': target labels} (optional)
        
        Returns:
            Dictionary with loss components and total loss
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Compute AEC loss if present
        if 'aec' in outputs and 'aec' in targets:
            aec_losses = self.aec_loss(
                outputs['aec']['magnitude'],
                targets['aec']['target_mag']
            )
            loss_dict['aec_magnitude_loss'] = aec_losses['magnitude_loss']
            loss_dict['aec_sisdr_loss'] = aec_losses['sisdr_loss']
            loss_dict['aec_loss'] = aec_losses['total']
            total_loss += self.aec_weight * aec_losses['total']
        
        # Compute VAD loss if present
        if 'vad' in outputs and 'vad' in targets:
            vad_loss_val = self.vad_loss(
                outputs['vad']['logits'],
                targets['vad']['target_labels']
            )
            loss_dict['vad_loss'] = vad_loss_val
            total_loss += self.vad_weight * vad_loss_val
        
        loss_dict['total'] = total_loss
        
        return loss_dict
