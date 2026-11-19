"""
NW-UCLA feeder (20 joints, single person).

Expected dataset layout:
    <data_path>/
        train/*.json      (cameras V1 and V2)
        val/*.json        (camera V3)

Each JSON file follows the common NW-UCLA format {"skeletons": <list>} where
numpy.array(skeletons) has shape (T, 20, 3). The class id is parsed from the
file name prefix aXX_... (label = XX - 1, 10 classes). Alternatively set
label_path to a JSON list of {"file_name": "a01_..._v01", "label": 1} (1-based).
Preprocessing follows Sec. 4.2 (see feeders/tools.py).
"""
import glob
import json
import os
import re

import numpy as np
from torch.utils.data import Dataset

from feeders import tools
from feeders.bone_pairs import ucla_pairs


def _label_from_stem(stem):
    m = re.match(r'^a(\d+)_', stem, re.IGNORECASE)
    if not m:
        raise ValueError(f'Cannot parse action id from filename: {stem}')
    return int(m.group(1)) - 1


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, split='train', window_size=16,
                 bone=False, vel=False, debug=False, use_mmap=False):
        self.data_path = os.path.abspath(data_path)
        self.label_path, self.split, self.window_size = label_path, split, window_size
        self.bone, self.vel = bone, vel
        self.samples = []
        self._load_sample_list()
        if debug:
            self.samples = self.samples[:100]
        self.label = np.array([s['label'] for s in self.samples])

    def _load_sample_list(self):
        if self.label_path is not None and str(self.label_path).lower() not in ('null', 'none', ''):
            with open(self.label_path, 'r') as f:
                meta = json.load(f)
            for item in meta:
                stem = item['file_name'][:-5] if item['file_name'].endswith('.json') else item['file_name']
                label = int(item.get('label', _label_from_stem(stem) + 1)) - 1
                self.samples.append({'path': os.path.join(self.data_path, f'{stem}.json'), 'label': label})
            return
        sub = 'train' if self.split == 'train' else 'val'
        split_dir = os.path.join(self.data_path, sub)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f'NW-UCLA: expected directory {split_dir} or a label_path JSON.')
        for path in sorted(glob.glob(os.path.join(split_dir, '*.json'))):
            stem = os.path.splitext(os.path.basename(path))[0]
            self.samples.append({'path': path, 'label': _label_from_stem(stem)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        with open(item['path'], 'r') as f:
            sk = np.array(json.load(f)['skeletons'], dtype=np.float32)
        if sk.ndim != 3 or sk.shape[1:] != (20, 3):
            raise ValueError(f'Expected skeletons (T, 20, 3), got {sk.shape} in {item["path"]}')
        x = sk.transpose(2, 0, 1)[:, :, :, None]  # (C, T, V, M=1)
        valid = tools.num_valid_frames(x)
        x = tools.translate_to_spine_base(x, root=0)  # joint 1: hip centre / spine base
        x, valid = tools.uniform_sample_or_pad(x, valid, self.window_size)
        if self.bone:
            x = tools.to_bone(x, ucla_pairs)
        if self.vel:
            x = tools.to_motion(x, valid)
        return x.astype(np.float32), int(item['label']), index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit) * 1.0 / len(hit)
