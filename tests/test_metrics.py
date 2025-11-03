"""Unit tests for evaluation metrics using mock data."""

import pytest
import torch
import numpy as np

from models.metrics import VADMetrics, AECMetrics, MultiTaskMetrics


def test_vad_metrics_basic():
    """Test basic VAD metrics computation."""
    metrics = VADMetrics()
    
    # Create perfect predictions
    logits = torch.tensor([[2.0, -2.0, 2.0, -2.0],
                           [2.0, 2.0, -2.0, -2.0]])
    targets = torch.tensor([[1, 0, 1, 0],
                           [1, 1, 0, 0]])
    
    metrics.update(logits, targets)
    results = metrics.compute()
    
    # Perfect predictions should have accuracy=1.0
    assert results['accuracy'] == 1.0
    assert results['precision'] == 1.0
    assert results['recall'] == 1.0
    assert results['f1_score'] == 1.0


def test_vad_metrics_imperfect():
    """Test VAD metrics with imperfect predictions."""
    metrics = VADMetrics()
    
    # Create imperfect predictions
    logits = torch.tensor([[2.0, -2.0, -2.0, 2.0]])  # Last two are wrong
    targets = torch.tensor([[1, 0, 1, 0]])
    
    metrics.update(logits, targets)
    results = metrics.compute()
    
    # Should have less than perfect accuracy
    assert results['accuracy'] < 1.0


def test_vad_metrics_reset():
    """Test VAD metrics reset."""
    metrics = VADMetrics()
    
    # Add some data
    logits = torch.randn(4, 100)
    targets = torch.randint(0, 2, (4, 100))
    
    metrics.update(logits, targets)
    
    # Reset
    metrics.reset()
    
    # Should be empty
    assert len(metrics.all_predictions) == 0
    assert len(metrics.all_targets) == 0


def test_vad_metrics_multiple_batches():
    """Test VAD metrics accumulation over multiple batches."""
    metrics = VADMetrics()
    
    # Add multiple batches
    for _ in range(3):
        logits = torch.tensor([[2.0, -2.0, 2.0, -2.0]])
        targets = torch.tensor([[1, 0, 1, 0]])
        metrics.update(logits, targets)
    
    results = metrics.compute()
    
    # Should still be perfect
    assert results['accuracy'] == 1.0


def test_aec_metrics_basic():
    """Test basic AEC metrics computation."""
    metrics = AECMetrics()
    
    # Create predictions and targets
    predictions = torch.randn(4, 257, 100)
    targets = predictions.clone()  # Perfect match
    
    metrics.update(predictions, targets)
    results = metrics.compute()
    
    # SI-SDR should be very high for perfect match
    assert 'si_sdr' in results
    assert results['si_sdr'] > 30  # Should be very high


def test_aec_metrics_imperfect():
    """Test AEC metrics with imperfect predictions."""
    metrics = AECMetrics()
    
    # Create different predictions and targets
    predictions = torch.randn(4, 257, 100)
    targets = torch.randn(4, 257, 100)
    
    metrics.update(predictions, targets)
    results = metrics.compute()
    
    # SI-SDR should be finite
    assert 'si_sdr' in results
    assert np.isfinite(results['si_sdr'])


def test_aec_metrics_reset():
    """Test AEC metrics reset."""
    metrics = AECMetrics()
    
    # Add some data
    predictions = torch.randn(4, 257, 100)
    targets = torch.randn(4, 257, 100)
    
    metrics.update(predictions, targets)
    
    # Reset
    metrics.reset()
    
    # Should be empty
    assert len(metrics.si_sdr_scores) == 0


def test_multitask_metrics_aec_only():
    """Test multi-task metrics with AEC only."""
    metrics = MultiTaskMetrics()
    
    outputs = {
        'aec': {
            'magnitude': torch.randn(4, 257, 100)
        }
    }
    
    targets = {
        'aec': {
            'target_mag': torch.randn(4, 257, 100)
        }
    }
    
    metrics.update(outputs, targets)
    results = metrics.compute()
    
    # Should have AEC metrics
    assert 'aec' in results
    assert 'si_sdr' in results['aec']


def test_multitask_metrics_vad_only():
    """Test multi-task metrics with VAD only."""
    metrics = MultiTaskMetrics()
    
    outputs = {
        'vad': {
            'logits': torch.tensor([[2.0, -2.0, 2.0, -2.0]])
        }
    }
    
    targets = {
        'vad': {
            'target_labels': torch.tensor([[1, 0, 1, 0]])
        }
    }
    
    metrics.update(outputs, targets)
    results = metrics.compute()
    
    # Should have VAD metrics
    assert 'vad' in results
    assert 'accuracy' in results['vad']
    assert 'f1_score' in results['vad']


def test_multitask_metrics_both_tasks():
    """Test multi-task metrics with both tasks."""
    metrics = MultiTaskMetrics()
    
    outputs = {
        'aec': {
            'magnitude': torch.randn(4, 257, 100)
        },
        'vad': {
            'logits': torch.tensor([[2.0, -2.0, 2.0, -2.0]])
        }
    }
    
    targets = {
        'aec': {
            'target_mag': torch.randn(4, 257, 100)
        },
        'vad': {
            'target_labels': torch.tensor([[1, 0, 1, 0]])
        }
    }
    
    metrics.update(outputs, targets)
    results = metrics.compute()
    
    # Should have both metrics
    assert 'aec' in results
    assert 'vad' in results


def test_multitask_metrics_reset():
    """Test multi-task metrics reset."""
    metrics = MultiTaskMetrics()
    
    outputs = {
        'aec': {
            'magnitude': torch.randn(4, 257, 100)
        },
        'vad': {
            'logits': torch.randn(4, 100)
        }
    }
    
    targets = {
        'aec': {
            'target_mag': torch.randn(4, 257, 100)
        },
        'vad': {
            'target_labels': torch.randint(0, 2, (4, 100))
        }
    }
    
    metrics.update(outputs, targets)
    
    # Reset
    metrics.reset()
    
    # Both should be empty
    assert len(metrics.aec_metrics.si_sdr_scores) == 0
    assert len(metrics.vad_metrics.all_predictions) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
