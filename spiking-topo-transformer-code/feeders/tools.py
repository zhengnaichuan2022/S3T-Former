"""
Preprocessing used for all benchmarks (Sec. 4.2 of the paper):

* the 3D joint coordinates are translated relative to the spine-base joint;
* each sequence is uniformly sampled or zero-padded to T frames;
* no random rotation, scaling, shearing, or additional temporal augmentation.

All arrays use the layout (C, T, V, M).
"""
import numpy as np


def num_valid_frames(data):
    """Number of leading frames that contain any non-zero coordinate."""
    nonzero = np.any(data != 0, axis=(0, 2, 3))
    return int(np.nonzero(nonzero)[0][-1] + 1) if nonzero.any() else 0


def translate_to_spine_base(data, root=0):
    """Subtract the spine-base coordinates of the first valid frame of the first
    person from every non-padded frame of every person."""
    present = np.any(data != 0, axis=(0, 2))  # (T, M)
    if not present[:, 0].any():
        return data
    f0 = int(np.argmax(present[:, 0]))
    origin = data[:, f0, root, 0].copy()  # (C,)
    out = data - origin[:, None, None, None]
    return out * present[None, :, None, :]  # keep zero padding (absent frames / persons) at zero


def uniform_sample_or_pad(data, valid, window):
    """Uniformly sample `window` frames from the first `valid` frames, or keep
    all valid frames and zero-pad when the sequence is shorter than `window`."""
    C, _, V, M = data.shape
    if valid >= window:
        idx = np.linspace(0, valid - 1, window).round().astype(np.int64)
        return data[:, idx].copy(), window
    out = np.zeros((C, window, V, M), dtype=data.dtype)
    out[:, :valid] = data[:, :valid]
    return out, valid


def to_bone(data, pairs):
    """Bone vectors from 1-based (child, parent) pairs: B[child] = J[child] - J[parent]."""
    bone = np.zeros_like(data)
    for v1, v2 in pairs:
        bone[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
    return bone


def to_motion(data, valid):
    """Frame-to-frame differences X[t+1] - X[t] within the valid frames; zero elsewhere."""
    motion = np.zeros_like(data)
    if valid > 1:
        motion[:, :valid - 1] = data[:, 1:valid] - data[:, :valid - 1]
    return motion
