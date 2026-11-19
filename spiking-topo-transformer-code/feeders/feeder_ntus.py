"""
NTU RGB+D 60 / 120 feeder.

Expects an .npz file with x_train / y_train / x_test / y_test, where x_* has shape
(N, T, 150) = (N, T, 2 persons x 25 joints x 3) and y_* is one-hot.
Preprocessing follows Sec. 4.2 (see feeders/tools.py); bone and motion
streams are derived after preprocessing.
"""
import numpy as np
from torch.utils.data import Dataset

from feeders import tools
from feeders.bone_pairs import ntu_pairs


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, split='train', window_size=16,
                 bone=False, vel=False, debug=False, use_mmap=False):
        """
        :param data_path: path to the .npz file
        :param label_path: unused, kept for config compatibility
        :param split: 'train' or 'test'
        :param window_size: simulation window T
        :param bone: use the bone modality
        :param vel: use the motion modality (applied after bone for bone-motion)
        :param debug: only use the first 100 samples
        :param use_mmap: memory-map the .npz file
        """
        self.data_path, self.split, self.window_size = data_path, split, window_size
        self.bone, self.vel, self.debug, self.use_mmap = bone, vel, debug, use_mmap
        self.load_data()

    def load_data(self):
        npz = np.load(self.data_path, mmap_mode='r' if self.use_mmap else None)
        if self.split == 'train':
            data, label = npz['x_train'], npz['y_train']
        elif self.split == 'test':
            data, label = npz['x_test'], npz['y_test']
        else:
            raise NotImplementedError('split must be train or test')
        if self.debug:
            data, label = data[:100], label[:100]
        self.data = data  # (N, T, 150), reshaped lazily per sample
        self.label = np.where(np.asarray(label) > 0)[1]
        self.sample_name = [f'{self.split}_{i}' for i in range(len(self.label))]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        x = np.array(self.data[index], dtype=np.float32)  # (T, 150)
        x = x.reshape(x.shape[0], 2, 25, 3).transpose(3, 0, 2, 1)  # (C, T, V, M)
        valid = tools.num_valid_frames(x)
        x = tools.translate_to_spine_base(x, root=0)
        x, valid = tools.uniform_sample_or_pad(x, valid, self.window_size)
        if self.bone:
            x = tools.to_bone(x, ntu_pairs)
        if self.vel:
            x = tools.to_motion(x, valid)
        return x.astype(np.float32), int(self.label[index]), index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit) * 1.0 / len(hit)
