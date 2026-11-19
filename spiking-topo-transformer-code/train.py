#!/usr/bin/env python
"""
Training and evaluation of S3T-Former (Sec. 4.2 of the paper).

* objective: prefix objective of Eq. (14), averaged over all prefix logits z[t];
* optimizer: AdamW, base learning rate 0.01, weight decay 0.0005, batch size 64;
* schedule: 10-epoch linear warmup, then cosine annealing to 1e-5, 250 epochs;
* automatic mixed precision on CUDA; inference uses the terminal logits z[T].

Examples:
    python train.py --config config/nturgbd-cross-subject-s3t-former.yaml --work-dir logs/ntu60-xsub-j-d256-t16
    python train.py --config config/nturgbd-cross-subject-s3t-former.yaml --work-dir logs/t32 \
        --opts train_feeder_args.window_size=32 test_feeder_args.window_size=32
    python train.py --config <cfg> --work-dir <dir> --eval-only --weights <dir>/best_model.pth
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.spiking_ssm_topo_transformer import prefix_loss  # noqa: E402


# ---------------------------------------------------------------- config
def deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    """YAML loader with `_base_` inheritance (path relative to the config file)."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop('_base_', None)
    if base is None:
        return cfg
    return deep_update(load_config(os.path.join(os.path.dirname(path), base)), cfg)


def apply_overrides(cfg, opts):
    """--opts a.b=value ... (values parsed as YAML)."""
    for item in opts or []:
        key, _, value = item.partition('=')
        node = cfg
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def import_class(name):
    mod_str, _, cls_str = name.rpartition('.')
    __import__(mod_str)
    return getattr(sys.modules[mod_str], cls_str)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- schedule
def warmup_cosine(base_lr, min_lr, warmup_iters, total_iters):
    """Linear warmup to base_lr, then cosine annealing to min_lr (per iteration)."""
    def factor(it):
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return lr / base_lr
    return factor


# ---------------------------------------------------------------- loops
def as_5d(data):
    return data.unsqueeze(-1) if data.dim() == 4 else data  # (N, C, T, V[, M])


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, max_grad_norm, log):
    model.train()
    losses, correct, total = [], 0, 0
    for data, label, _ in tqdm(loader, desc=f'Epoch {epoch}', leave=False):
        data, label = as_5d(data).float().to(device), label.long().to(device)
        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            z = model(data, return_prefix=True).transpose(0, 1)  # (T, N, num_classes)
            loss = prefix_loss(z.float(), label)
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        correct += (z[-1].argmax(1) == label).sum().item()
        total += label.numel()
    log(f'Epoch {epoch}: train loss {np.mean(losses):.4f}, train top-1 {100.0 * correct / total:.2f}%, '
        f'lr {optimizer.param_groups[0]["lr"]:.2e}')


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """Returns Top-1, Top-5 (%) on z[T], the terminal logits, and the labels."""
    model.eval()
    scores, labels = [], []
    for data, label, _ in tqdm(loader, desc='Testing', leave=False):
        data = as_5d(data).float().to(device)
        functional.reset_net(model)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            z = model(data)
        scores.append(z.float().cpu())
        labels.append(label.long())
    scores, labels = torch.cat(scores), torch.cat(labels)
    top5 = scores.topk(min(5, scores.shape[1]), dim=1).indices
    top1 = 100.0 * (top5[:, 0] == labels).float().mean().item()
    top5 = 100.0 * (top5 == labels[:, None]).any(1).float().mean().item()
    return top1, top5, scores.numpy(), labels.numpy()


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description='S3T-Former training / evaluation')
    parser.add_argument('--config', default='config/nturgbd-cross-subject-s3t-former.yaml')
    parser.add_argument('--work-dir', default='./logs/s3t-former')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opts', nargs='*', default=[], help='config overrides, e.g. model_args.embed_dim=384')
    parser.add_argument('--eval-only', action='store_true', help='evaluate --weights and save the test scores')
    parser.add_argument('--weights', default=None, help='checkpoint for --eval-only')
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.opts)
    init_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    log_file = open(os.path.join(args.work_dir, 'log.txt'), 'a', encoding='utf-8')

    def log(msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line)
        log_file.write(line + '\n')
        log_file.flush()

    with open(os.path.join(args.work_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device_ids = cfg.get('device', [0])
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    model = import_class(cfg['model'])(**cfg['model_args']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'Model: {cfg["model"]}, {n_params / 1e6:.2f}M parameters, device {device}')
    if device.type == 'cuda' and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    core = model.module if isinstance(model, nn.DataParallel) else model

    Feeder = import_class(cfg['feeder'])
    test_loader = DataLoader(Feeder(**cfg['test_feeder_args']), batch_size=cfg.get('test_batch_size', 128),
                             shuffle=False, num_workers=cfg.get('num_worker', 4),
                             pin_memory=cfg.get('pin_memory', True))
    use_amp = bool(cfg.get('use_amp', True)) and device.type == 'cuda'

    if args.eval_only:
        state = torch.load(args.weights, map_location=device)
        core.load_state_dict(state.get('model_state_dict', state))
        top1, top5, scores, labels = evaluate(model, test_loader, device, use_amp)
        np.savez(os.path.join(args.work_dir, 'test_scores.npz'), scores=scores, labels=labels)
        log(f'Eval {args.weights}: top-1 {top1:.2f}%, top-5 {top5:.2f}%')
        return

    train_loader = DataLoader(Feeder(**cfg['train_feeder_args']), batch_size=cfg.get('batch_size', 64),
                              shuffle=True, drop_last=True, num_workers=cfg.get('num_worker', 4),
                              pin_memory=cfg.get('pin_memory', True))
    base_lr, min_lr = float(cfg.get('base_lr', 0.01)), float(cfg.get('min_lr', 1e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=float(cfg.get('weight_decay', 5e-4)),
                                  betas=tuple(cfg.get('betas', (0.9, 0.999))))
    num_epoch = int(cfg.get('num_epoch', 250))
    iters = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(
        base_lr, min_lr, int(cfg.get('warmup_epochs', 10)) * iters, num_epoch * iters))
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best = 0.0
    for epoch in range(1, num_epoch + 1):
        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch,
                        cfg.get('max_grad_norm'), log)
        torch.save({'epoch': epoch, 'model_state_dict': core.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), 'best_top1': best},
                   os.path.join(args.work_dir, 'latest_model.pth'))
        if epoch % int(cfg.get('eval_interval', 5)) == 0 or epoch == num_epoch:
            top1, top5, scores, labels = evaluate(model, test_loader, device, use_amp)
            log(f'Epoch {epoch}: test top-1 {top1:.2f}%, top-5 {top5:.2f}%')
            if top1 > best:
                best = top1
                torch.save({'epoch': epoch, 'model_state_dict': core.state_dict()},
                           os.path.join(args.work_dir, 'best_model.pth'))
                np.savez(os.path.join(args.work_dir, 'test_scores.npz'), scores=scores, labels=labels)
                log(f'  saved best model (top-1 {best:.2f}%)')
    log(f'Training finished. Best top-1 {best:.2f}%')


if __name__ == '__main__':
    main()
