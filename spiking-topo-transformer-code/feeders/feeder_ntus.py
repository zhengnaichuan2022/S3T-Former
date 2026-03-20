import numpy as np

from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False,
                 # New data augmentation parameters
                 random_noise=False, noise_std=0.01,
                 random_scale=False, scale_range=[0.9, 1.1],
                 random_mask=False, mask_ratio=0.1,
                 temporal_warp=False, warp_sigma=0.2, warp_knot=4,
                 random_flip=False, flip_prob=0.5,
                 gaussian_blur=False, blur_kernel_size=3, blur_sigma=1.0,
                 random_dropout_frames=False, dropout_ratio=0.1):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: If true, apply random movement transformation
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param random_noise: If true, add random Gaussian noise
        :param noise_std: Standard deviation of Gaussian noise
        :param random_scale: If true, apply random scaling
        :param scale_range: Range of scaling factors [min, max]
        :param random_mask: If true, randomly mask some joints
        :param mask_ratio: Ratio of joints to mask
        :param temporal_warp: If true, apply temporal warping
        :param warp_sigma: Strength of temporal warping
        :param warp_knot: Number of control points for temporal warping
        :param random_flip: If true, randomly flip horizontally
        :param flip_prob: Probability of flipping
        :param gaussian_blur: If true, apply Gaussian blur on temporal dimension
        :param blur_kernel_size: Kernel size for Gaussian blur
        :param blur_sigma: Sigma for Gaussian blur
        :param random_dropout_frames: If true, randomly drop some frames
        :param dropout_ratio: Ratio of frames to drop
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        
        # New data augmentation parameters
        self.random_noise = random_noise
        self.noise_std = noise_std
        self.random_scale = random_scale
        self.scale_range = scale_range
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
        
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        import time
        print(f"Loading data from {self.data_path} (split={self.split})...")
        start_time = time.time()
        
        # Optimization: Use mmap mode to load large files, save memory
        if self.use_mmap:
            print("  Using mmap mode (memory-efficient, but first access may be slower)...")
            npz_data = np.load(self.data_path, mmap_mode='r')  # Read-only mmap mode
        else:
            print("  Loading full data into memory (may take time for large files)...")
            npz_data = np.load(self.data_path)
        
        load_time = time.time() - start_time
        print(f"  ✓ NPZ file loaded in {load_time:.2f}s")
        
        start_time = time.time()
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        
        access_time = time.time() - start_time
        print(f"  ✓ Data accessed in {access_time:.2f}s (samples: {len(self.data)})")
        
        start_time = time.time()
        N, T, _ = self.data.shape
        # Optimization: If using mmap, reshape and transpose will create new arrays (need to copy)
        # If not using mmap, can operate directly
        if self.use_mmap:
            # In mmap mode, need to explicitly copy data to reshape
            # Using copy=False can avoid unnecessary copying, but reshape requires contiguous arrays, so still need to copy
            self.data = np.array(self.data, copy=True).reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        else:
            self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        
        reshape_time = time.time() - start_time
        print(f"  ✓ Data reshaped in {reshape_time:.2f}s")
        print(f"  Total loading time: {load_time + access_time + reshape_time:.2f}s")

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        # Optimization: Data augmentation will modify data, need to ensure it's an independent copy
        # If using mmap, indexing returns a view, need to copy; if not using mmap, also need to copy to avoid modifying original data
        data_numpy = np.array(data_numpy, copy=True)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        
        # Data augmentation (only applied during training, and applied in order)
        # Note: Some augmentation methods may change data shape, need to carefully apply in order
        
        # 1. Temporal-related augmentation (before spatial augmentation)
        if self.random_choose and self.split == 'train':
            data_numpy = tools.random_choose(data_numpy, self.window_size if self.window_size > 0 else data_numpy.shape[1])
        
        if self.random_shift and self.split == 'train':
            data_numpy = tools.random_shift(data_numpy)
        
        if self.temporal_warp and self.split == 'train':
            data_numpy = tools.temporal_warp(data_numpy, sigma=self.warp_sigma, knot=self.warp_knot)
        
        if self.random_dropout_frames and self.split == 'train':
            data_numpy = tools.random_dropout_frames(data_numpy, dropout_ratio=self.dropout_ratio)
        
        # 2. Spatial transformation augmentation
        if self.random_move and self.split == 'train':
            data_numpy = tools.random_move(data_numpy)
        
        if self.random_rot and self.split == 'train':
            data_numpy = tools.random_rot(data_numpy)
        
        if self.random_flip and self.split == 'train':
            data_numpy = tools.random_flip(data_numpy, flip_prob=self.flip_prob)
        
        if self.random_scale and self.split == 'train':
            data_numpy = tools.random_scale(data_numpy, scale_range=self.scale_range)
        
        # 3. Noise and masking augmentation
        if self.random_noise and self.split == 'train':
            data_numpy = tools.random_noise(data_numpy, noise_std=self.noise_std)
        
        if self.random_mask and self.split == 'train':
            data_numpy = tools.random_mask(data_numpy, mask_ratio=self.mask_ratio)
        
        # 4. Smoothing augmentation (applied last)
        if self.gaussian_blur and self.split == 'train':
            data_numpy = tools.gaussian_blur_temporal(data_numpy, kernel_size=self.blur_kernel_size, sigma=self.blur_sigma)
        
        # 5. Modality conversion (bone and vel)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
