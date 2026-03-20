#!/usr/bin/env python
"""
Test if the model can be imported and initialized correctly
"""

import torch
from model.spiking_ssm_topo_transformer import SpikingStateSpaceTopologyTransformer

def test_model():
    """Test model initialization"""
    print("Testing model initialization...")
    
    # Create model
    model = SpikingStateSpaceTopologyTransformer(
        num_nodes=25,
        in_channels=3,
        embed_dim=384,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        num_classes=60,
        v_threshold=0.5,
        dropout=0.25,
        use_topology_bias=True,
        topology_alpha=0.5,
        num_person=2,
        use_temporal_gradient_qkv=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params/1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # Test forward pass
    print("\nTesting forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy input: (N, C, T, V, M)
    dummy_input = torch.randn(2, 3, 16, 25, 2).to(device)
    
    with torch.no_grad():
        from spikingjelly.activation_based import functional
        functional.reset_net(model)
        output = model(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    print("\n✓ All tests passed!")

if __name__ == '__main__':
    test_model()

