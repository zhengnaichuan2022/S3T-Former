#!/usr/bin/env python
"""
Smoke test: builds S3T-Former and every ablation variant of Sec. 4.5 and runs a
forward and backward pass with the prefix objective on random input (CPU or GPU).

    python test_model.py
"""
import torch
from spikingjelly.activation_based import functional

from model.spiking_ssm_topo_transformer import SpikingStateSpaceTopologyTransformer, prefix_loss

VARIANTS = {
    'full (hybrid LSTR, S3-Engine, U-Readout)': {},
    'w/o M-ASE': dict(use_mase=False),
    'w/o ATG-QKV': dict(use_temporal_gradient_qkv=False),
    'w/o LSTR (identity)': dict(lstr_mode='identity'),
    'LSTR learned only': dict(lstr_mode='learned'),
    'LSTR physical only': dict(lstr_mode='physical'),
    'w/o S3-Engine': dict(use_s3_engine=False),
    'S3-Engine fixed decay 0.5': dict(s3_decay_mode='fixed'),
    'w/o U-Readout (spike count)': dict(readout='spike_count'),
    'H=1': dict(num_heads=1),
}


def run(name, device, num_nodes=25, num_person=2, num_classes=60, **kwargs):
    model = SpikingStateSpaceTopologyTransformer(num_nodes=num_nodes, num_person=num_person,
                                                 num_classes=num_classes, embed_dim=64, **kwargs).to(device)
    x = torch.randn(2, 3, 16, num_nodes, num_person, device=device)
    y = torch.randint(0, num_classes, (2,), device=device)
    functional.reset_net(model)
    z = model(x, return_prefix=True).transpose(0, 1)  # (T, N, classes)
    loss = prefix_loss(z, y)
    loss.backward()
    assert z.shape == (16, 2, num_classes) and torch.isfinite(loss)
    print(f'  ok  {name:45s} loss {loss.item():.3f}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    for name, kwargs in VARIANTS.items():
        run(name, device, **kwargs)
    run('NW-UCLA (20 joints, 1 person)', device, num_nodes=20, num_person=1, num_classes=10)
    print('All tests passed.')


if __name__ == '__main__':
    main()
