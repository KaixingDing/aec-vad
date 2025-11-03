"""Unit tests for model architectures using mock data."""

import pytest
import torch
import numpy as np

from models import SharedBackboneModel, HierarchicalModel


def test_shared_backbone_model_aec():
    """Test shared backbone model with AEC task."""
    model = SharedBackboneModel(
        n_freqs=257,
        aec_channels=2,
        vad_channels=1,
    )
    
    # Create mock input (batch=2, channels=2, freq=257, time=100)
    batch_size = 2
    n_freqs = 257
    n_frames = 100
    
    aec_input = torch.randn(batch_size, 2, n_freqs, n_frames)
    
    # Forward pass
    output = model(aec_input, task='aec')
    
    # Check output
    assert 'magnitude' in output
    assert output['magnitude'].shape == (batch_size, n_freqs, n_frames)
    
    # Check output is in valid range (after sigmoid)
    assert output['magnitude'].min() >= 0
    assert output['magnitude'].max() <= 1


def test_shared_backbone_model_vad():
    """Test shared backbone model with VAD task."""
    model = SharedBackboneModel(
        n_freqs=257,
        aec_channels=2,
        vad_channels=1,
    )
    
    # Create mock input (batch=2, channels=1, freq=257, time=100)
    batch_size = 2
    n_freqs = 257
    n_frames = 100
    
    vad_input = torch.randn(batch_size, 1, n_freqs, n_frames)
    
    # Forward pass
    output = model(vad_input, task='vad')
    
    # Check output
    assert 'logits' in output
    assert output['logits'].shape == (batch_size, n_frames)


def test_hierarchical_model_aec():
    """Test hierarchical model with AEC task."""
    model = HierarchicalModel(
        n_freqs=257,
        aec_channels=2,
        vad_channels=1,
    )
    
    # Create mock input
    batch_size = 2
    n_freqs = 257
    n_frames = 100
    
    aec_input = torch.randn(batch_size, 2, n_freqs, n_frames)
    
    # Forward pass without VAD guidance
    output = model(aec_input, task='aec')
    
    # Check output
    assert 'magnitude' in output
    assert output['magnitude'].shape == (batch_size, n_freqs, n_frames)


def test_hierarchical_model_vad():
    """Test hierarchical model with VAD task."""
    model = HierarchicalModel(
        n_freqs=257,
        aec_channels=2,
        vad_channels=1,
    )
    
    # Create mock input
    batch_size = 2
    n_freqs = 257
    n_frames = 100
    
    vad_input = torch.randn(batch_size, 1, n_freqs, n_frames)
    
    # Forward pass
    output = model(vad_input, task='vad')
    
    # Check output
    assert 'logits' in output
    assert output['logits'].shape == (batch_size, n_frames)


def test_hierarchical_model_with_vad_guidance():
    """Test hierarchical model with VAD guidance for AEC."""
    model = HierarchicalModel(
        n_freqs=257,
        aec_channels=2,
        vad_channels=1,
    )
    
    # Create mock inputs
    batch_size = 2
    n_freqs = 257
    n_frames = 100
    
    aec_input = torch.randn(batch_size, 2, n_freqs, n_frames)
    vad_guidance = torch.rand(batch_size, n_frames)  # VAD probabilities
    
    # Forward pass with VAD guidance
    output = model(aec_input, task='aec', vad_guidance=vad_guidance)
    
    # Check output
    assert 'magnitude' in output
    assert output['magnitude'].shape == (batch_size, n_freqs, n_frames)


def test_hierarchical_model_forward_with_vad_guidance():
    """Test hierarchical model joint forward with VAD guidance."""
    model = HierarchicalModel(
        n_freqs=257,
        aec_channels=2,
        vad_channels=1,
    )
    
    # Create mock inputs
    batch_size = 2
    n_freqs = 257
    n_frames = 100
    
    aec_input = torch.randn(batch_size, 2, n_freqs, n_frames)
    vad_input = torch.randn(batch_size, 1, n_freqs, n_frames)
    
    # Joint forward pass
    outputs = model.forward_with_vad_guidance(aec_input, vad_input)
    
    # Check outputs
    assert 'aec' in outputs
    assert 'vad' in outputs
    assert outputs['aec']['magnitude'].shape == (batch_size, n_freqs, n_frames)
    assert outputs['vad']['logits'].shape == (batch_size, n_frames)


def test_model_parameter_count():
    """Test model parameter counting."""
    model = SharedBackboneModel()
    
    # Get parameter count
    num_params = model.get_num_parameters()
    
    # Check it's reasonable
    assert num_params > 0
    
    # Get detailed info
    info = model.get_model_info()
    
    assert 'total' in info
    assert info['total'] == num_params


def test_model_gradient_flow():
    """Test that gradients flow through the model."""
    model = SharedBackboneModel(n_freqs=257)
    
    # Create mock input and target
    aec_input = torch.randn(2, 2, 257, 100, requires_grad=True)
    target = torch.randn(2, 257, 100)
    
    # Forward pass
    output = model(aec_input, task='aec')
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(output['magnitude'], target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist on input
    assert aec_input.grad is not None
    
    # Check gradients exist on AEC feature extractor (which was used)
    for param in model.aec_feature_extractor.parameters():
        if param.requires_grad:
            assert param.grad is not None
    
    # Check gradients exist on shared backbone (which was used)
    for param in model.shared_backbone.parameters():
        if param.requires_grad:
            assert param.grad is not None
    
    # Check gradients exist on AEC head (which was used)
    for param in model.aec_head.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_model_device_transfer():
    """Test model can be moved to different devices."""
    model = SharedBackboneModel(n_freqs=257)
    
    # Move to CPU (already there, but test the method)
    model = model.cpu()
    
    # Create input on CPU
    aec_input = torch.randn(1, 2, 257, 50)
    
    # Forward pass
    output = model(aec_input, task='aec')
    
    # Check output is on CPU
    assert output['magnitude'].device.type == 'cpu'
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        aec_input = aec_input.cuda()
        
        output = model(aec_input, task='aec')
        assert output['magnitude'].device.type == 'cuda'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
