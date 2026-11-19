#!/usr/bin/env python
"""
Multi-stream ensemble (J+B+JM+BM, "All" in Table 1).

Each stream is evaluated with
    python train.py --config <stream config> --work-dir <dir> --eval-only --weights <dir>/best_model.pth
which writes <dir>/test_scores.npz (terminal logits z[T] and labels). This script
sums the per-stream scores (optionally weighted) and reports Top-1 / Top-5.

Example:
    python ensemble.py --scores logs/j/test_scores.npz logs/b/test_scores.npz \
        logs/jm/test_scores.npz logs/bm/test_scores.npz
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='S3T-Former multi-stream ensemble')
    parser.add_argument('--scores', nargs='+', required=True, help='test_scores.npz of each stream')
    parser.add_argument('--weights', nargs='*', type=float, default=None, help='per-stream weights (default 1)')
    args = parser.parse_args()

    weights = args.weights or [1.0] * len(args.scores)
    assert len(weights) == len(args.scores), 'one weight per score file'
    total, labels = None, None
    for path, w in zip(args.scores, weights):
        f = np.load(path)
        if labels is None:
            labels = f['labels']
        elif not np.array_equal(labels, f['labels']):
            raise ValueError(f'{path}: labels do not match the first score file')
        total = w * f['scores'] if total is None else total + w * f['scores']

    rank = np.argsort(-total, axis=1)
    top1 = 100.0 * np.mean(rank[:, 0] == labels)
    top5 = 100.0 * np.mean(np.any(rank[:, :5] == labels[:, None], axis=1))
    print(f'{len(args.scores)}-stream ensemble: top-1 {top1:.2f}%, top-5 {top5:.2f}%')


if __name__ == '__main__':
    main()
