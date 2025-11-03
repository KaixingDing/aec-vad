"""Evaluation metrics for AEC and VAD tasks."""

import torch
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class VADMetrics:
    """
    Metrics for Voice Activity Detection task.
    
    Computes: Accuracy, Precision, Recall, F1-Score
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Predicted logits or probabilities, shape (batch, time)
            targets: Target labels (0 or 1), shape (batch, time)
        """
        # Convert to binary predictions
        if predictions.dtype == torch.float:
            # Assume logits, apply sigmoid and threshold
            preds = (torch.sigmoid(predictions) > 0.5).long()
        else:
            preds = predictions
        
        # Flatten and move to CPU
        preds = preds.reshape(-1).cpu().numpy()
        targets = targets.reshape(-1).cpu().numpy()
        
        self.all_predictions.append(preds)
        self.all_targets.append(targets)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary with accuracy, precision, recall, and F1-score
        """
        if not self.all_predictions:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
            }
        
        predictions = np.concatenate(self.all_predictions)
        targets = np.concatenate(self.all_targets)
        
        return {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, zero_division=0),
            'recall': recall_score(targets, predictions, zero_division=0),
            'f1_score': f1_score(targets, predictions, zero_division=0),
        }


class AECMetrics:
    """
    Metrics for Acoustic Echo Cancellation task.
    
    Note: Full metrics like PESQ, STOI require time-domain reconstruction
    and external libraries. Here we provide SI-SDR as a basic metric.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.si_sdr_scores = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Predicted magnitude, shape (batch, freq, time)
            targets: Target magnitude, shape (batch, freq, time)
        """
        si_sdr = self.compute_si_sdr(predictions, targets)
        self.si_sdr_scores.append(si_sdr.cpu().numpy())
    
    @staticmethod
    def compute_si_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute Scale-Invariant SDR.
        
        Args:
            pred: Predicted signal, shape (batch, ...)
            target: Target signal, shape (batch, ...)
            eps: Small constant for numerical stability
        
        Returns:
            SI-SDR values, shape (batch,)
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
        
        return si_sdr
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary with SI-SDR
        """
        if not self.si_sdr_scores:
            return {
                'si_sdr': 0.0,
            }
        
        si_sdr_all = np.concatenate(self.si_sdr_scores)
        
        return {
            'si_sdr': float(si_sdr_all.mean()),
        }


class MultiTaskMetrics:
    """
    Combined metrics for multi-task learning.
    """
    
    def __init__(self):
        self.aec_metrics = AECMetrics()
        self.vad_metrics = VADMetrics()
    
    def reset(self):
        """Reset all metrics."""
        self.aec_metrics.reset()
        self.vad_metrics.reset()
    
    def update(self, outputs: Dict, targets: Dict):
        """
        Update metrics with new batch.
        
        Args:
            outputs: Dictionary with task outputs
            targets: Dictionary with task targets
        """
        if 'aec' in outputs and 'aec' in targets:
            self.aec_metrics.update(
                outputs['aec']['magnitude'],
                targets['aec']['target_mag']
            )
        
        if 'vad' in outputs and 'vad' in targets:
            self.vad_metrics.update(
                outputs['vad']['logits'],
                targets['vad']['target_labels']
            )
    
    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with metrics for each task
        """
        return {
            'aec': self.aec_metrics.compute(),
            'vad': self.vad_metrics.compute(),
        }
