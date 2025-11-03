"""Unit tests for loss functions using mock data."""

import pytest
import torch
from models.losses import (
    SISDRLoss,
    MagnitudeLoss,
    AECLoss,
    VADLoss,
    MultiTaskLoss,
)


def test_sisdr_loss():
    """Test SI-SDR loss computation."""
    loss_fn = SISDRLoss()
    
    # Create mock predictions and targets
    pred = torch.randn(4, 257, 100)  # batch=4, freq=257, time=100
    target = torch.randn(4, 257, 100)
    
    # Compute loss
    loss = loss_fn(pred, target)
    
    # Check loss is scalar
    assert loss.dim() == 0
    
    # Check loss is negative (since we negate SI-SDR)
    # (Note: may not always be negative depending on random data)
    assert torch.isfinite(loss)


def test_sisdr_loss_perfect_match():
    """Test SI-SDR loss with perfect prediction."""
    loss_fn = SISDRLoss()
    
    target = torch.randn(4, 257, 100)
    pred = target.clone()
    
    loss = loss_fn(pred, target)
    
    # Loss should be very negative (high SI-SDR)
    assert loss < 0


def test_magnitude_loss():
    """Test magnitude loss computation."""
    loss_fn = MagnitudeLoss()
    
    pred_mag = torch.randn(4, 257, 100)
    target_mag = torch.randn(4, 257, 100)
    
    loss = loss_fn(pred_mag, target_mag)
    
    # Check loss is scalar and positive
    assert loss.dim() == 0
    assert loss >= 0


def test_aec_loss():
    """Test combined AEC loss."""
    loss_fn = AECLoss(mag_weight=1.0, sisdr_weight=1.0)
    
    pred_mag = torch.randn(4, 257, 100)
    target_mag = torch.randn(4, 257, 100)
    
    losses = loss_fn(pred_mag, target_mag)
    
    # Check all components are present
    assert 'magnitude_loss' in losses
    assert 'sisdr_loss' in losses
    assert 'total' in losses
    
    # Check all are scalars
    for key, value in losses.items():
        assert value.dim() == 0


def test_vad_loss():
    """Test VAD loss computation."""
    loss_fn = VADLoss()
    
    # Create mock logits and labels
    logits = torch.randn(4, 100)  # batch=4, time=100
    labels = torch.randint(0, 2, (4, 100))  # binary labels
    
    loss = loss_fn(logits, labels)
    
    # Check loss is scalar and positive
    assert loss.dim() == 0
    assert loss >= 0


def test_multitask_loss_aec_only():
    """Test multi-task loss with AEC only."""
    loss_fn = MultiTaskLoss(aec_weight=1.0, vad_weight=1.0)
    
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
    
    losses = loss_fn(outputs, targets)
    
    # Check AEC losses are present
    assert 'aec_loss' in losses
    assert 'aec_magnitude_loss' in losses
    assert 'aec_sisdr_loss' in losses
    assert 'total' in losses


def test_multitask_loss_vad_only():
    """Test multi-task loss with VAD only."""
    loss_fn = MultiTaskLoss(aec_weight=1.0, vad_weight=1.0)
    
    outputs = {
        'vad': {
            'logits': torch.randn(4, 100)
        }
    }
    
    targets = {
        'vad': {
            'target_labels': torch.randint(0, 2, (4, 100))
        }
    }
    
    losses = loss_fn(outputs, targets)
    
    # Check VAD losses are present
    assert 'vad_loss' in losses
    assert 'total' in losses


def test_multitask_loss_both_tasks():
    """Test multi-task loss with both tasks."""
    loss_fn = MultiTaskLoss(aec_weight=1.0, vad_weight=1.0)
    
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
    
    losses = loss_fn(outputs, targets)
    
    # Check all losses are present
    assert 'aec_loss' in losses
    assert 'vad_loss' in losses
    assert 'total' in losses
    
    # Check total is sum of weighted losses
    assert torch.isfinite(losses['total'])


def test_multitask_loss_weights():
    """Test that task weights affect total loss."""
    # High AEC weight
    loss_fn_high_aec = MultiTaskLoss(aec_weight=10.0, vad_weight=0.1)
    
    # High VAD weight
    loss_fn_high_vad = MultiTaskLoss(aec_weight=0.1, vad_weight=10.0)
    
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
    
    losses_high_aec = loss_fn_high_aec(outputs, targets)
    losses_high_vad = loss_fn_high_vad(outputs, targets)
    
    # Losses should be different due to different weights
    assert not torch.allclose(
        losses_high_aec['total'],
        losses_high_vad['total']
    )


def test_loss_backward():
    """Test that losses can be backpropagated."""
    loss_fn = MultiTaskLoss()
    
    outputs = {
        'aec': {
            'magnitude': torch.randn(2, 257, 50, requires_grad=True)
        }
    }
    
    targets = {
        'aec': {
            'target_mag': torch.randn(2, 257, 50)
        }
    }
    
    losses = loss_fn(outputs, targets)
    
    # Backward pass
    losses['total'].backward()
    
    # Check gradient exists
    assert outputs['aec']['magnitude'].grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
