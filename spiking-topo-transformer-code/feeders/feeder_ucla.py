"""
NW-UCLA feeder (20 joints, single person).

Expected dataset layout (recommended):
    <data_path>/
        train/*.json
        val/*.json

Each JSON file follows common NW-UCLA format: {"skeletons": <list>} where
numpy.array(skeletons) has shape (T, V, 3) with V=20.

Class id: parsed from filename `aXX_...` -> label = XX - 1 (10 classes).

Alternatively set `label_path` to a JSON file: list of
{"file_name": "a01_..._v01", "label": 1}  (label 1-based, optional if filename parses).
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import re

import numpy as np
from torch.utils.data import Dataset

from feeders import tools


def _label_from_stem(stem: str) -> int:
    m = re.match(r"^a(\d+)_", stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse action id from filename: {stem}")
    return int(m.group(1)) - 1


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        split="train",
        label_path=None,
        p_interval=1,
        window_size=16,
        random_choose=False,
        random_shift=False,
        random_move=False,
        random_rot=False,
        normalization=False,
        debug=False,
        use_mmap=False,
        bone=False,
        vel=False,
        random_noise=False,
        noise_std=0.02,
        random_scale=False,
        scale_range=(0.9, 1.1),
        random_mask=False,
        mask_ratio=0.1,
        temporal_warp=False,
        warp_sigma=0.2,
        warp_knot=4,
        random_flip=False,
        flip_prob=0.5,
        gaussian_blur=False,
        blur_kernel_size=3,
        blur_sigma=1.0,
        random_dropout_frames=False,
        dropout_ratio=0.1,
    ):
        self.data_path = os.path.abspath(data_path)
        self.split = split
        self.label_path = label_path
        self.p_interval = p_interval
        self.window_size = window_size
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_rot = random_rot
        self.normalization = normalization
        self.debug = debug
        self.use_mmap = use_mmap
        self.bone = bone
        self.vel = vel
        self.random_noise = random_noise
        self.noise_std = noise_std
        self.random_scale = random_scale
        self.scale_range = list(scale_range) if scale_range is not None else [0.9, 1.1]
        self.random_mask = random_mask
        self.mask_ratio = mask_ratio
        self.temporal_warp = temporal_warp
        self.warp_sigma = warp_sigma
        self.warp_knot = warp_knot
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.gaussian_blur = gaussian_blur
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.random_dropout_frames = random_dropout_frames
        self.dropout_ratio = dropout_ratio

        self.samples = []  # list of dict: path, label (0-based)
        self._load_sample_list()

        if self.debug:
            self.samples = self.samples[:100]

        if normalization:
            self.get_mean_map()

    def _load_sample_list(self):
        if self.label_path is not None and str(self.label_path).lower() not in ("null", "none", ""):
            with open(self.label_path, "r") as f:
                meta = json.load(f)
            root = self.data_path
            for item in meta:
                stem = item["file_name"]
                if stem.endswith(".json"):
                    stem = stem[:-5]
                path = os.path.join(root, f"{stem}.json")
                lab = int(item.get("label", _label_from_stem(stem) + 1))
                self.samples.append({"path": path, "label": lab - 1})
            return

        # Subfolder split: train/ or val/
        sub = "train" if self.split == "train" else "val"
        split_dir = os.path.join(self.data_path, sub)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"NW-UCLA: expected directory {split_dir} or provide label_path JSON. "
                f"Put {sub}/*.json under {self.data_path}."
            )
        paths = sorted(glob.glob(os.path.join(split_dir, "*.json")))
        for path in paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            self.samples.append({"path": path, "label": _label_from_stem(stem)})

    def get_mean_map(self):
        # Optional; not used when normalization=False (default)
        pass

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _rand_view_transform(x, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
        ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
        ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
        x0 = np.dot(np.reshape(x, (-1, 3)), np.dot(ry, np.dot(rx, ss)))
        return np.reshape(x0, x.shape)

    def _load_skeleton(self, path):
        with open(path, "r") as f:
            obj = json.load(f)
        sk = np.array(obj["skeletons"], dtype=np.float32)
        if sk.ndim != 3 or sk.shape[1] != 20 or sk.shape[2] != 3:
            raise ValueError(f"Expected skeletons (T,20,3), got {sk.shape} in {path}")
        return sk

    def __getitem__(self, index):
        item = self.samples[index]
        path = item["path"]
        label = item["label"]
        value = np.array(self._load_skeleton(path), copy=True)

        is_train = self.split == "train"
        if is_train:
            random.random()
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)
        else:
            agx, agy, s = 0, 0, 1.0

        center = value[0, 1, :].copy()
        value = value - center
        value = self._rand_view_transform(value, agx, agy, s)

        flat = np.reshape(value, (-1, 3))
        fmin = np.min(flat, axis=0)
        fmax = np.max(flat, axis=0)
        denom = np.maximum(fmax - fmin, 1e-8)
        flat = (flat - fmin) / denom
        flat = flat * 2.0 - 1.0
        value = np.reshape(flat, (-1, 20, 3))

        # (C, T, V, M)
        data_numpy = np.transpose(value, (2, 0, 1))[:, :, :, None]

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        if self.random_choose and is_train:
            data_numpy = tools.random_choose(data_numpy, self.window_size if self.window_size > 0 else data_numpy.shape[1])
        if self.random_shift and is_train:
            data_numpy = tools.random_shift(data_numpy)
        if self.temporal_warp and is_train:
            data_numpy = tools.temporal_warp(data_numpy, sigma=self.warp_sigma, knot=self.warp_knot)
        if self.random_dropout_frames and is_train:
            data_numpy = tools.random_dropout_frames(data_numpy, dropout_ratio=self.dropout_ratio)
        if self.random_move and is_train:
            data_numpy = tools.random_move(data_numpy)
        if self.random_rot and is_train:
            data_numpy = tools.random_rot(data_numpy)
        if self.random_flip and is_train:
            data_numpy = tools.random_flip(data_numpy, flip_prob=self.flip_prob)
        if self.random_scale and is_train:
            data_numpy = tools.random_scale(data_numpy, scale_range=self.scale_range)
        if self.random_noise and is_train:
            data_numpy = tools.random_noise(data_numpy, noise_std=self.noise_std)
        if self.random_mask and is_train:
            data_numpy = tools.random_mask(data_numpy, mask_ratio=self.mask_ratio)
        if self.gaussian_blur and is_train:
            data_numpy = tools.gaussian_blur_temporal(
                data_numpy, kernel_size=self.blur_kernel_size, sigma=self.blur_sigma
            )

        if self.bone:
            from feeders.bone_pairs import ucla_pairs

            bone_data = np.zeros_like(data_numpy)
            for v1, v2 in ucla_pairs:
                bone_data[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
