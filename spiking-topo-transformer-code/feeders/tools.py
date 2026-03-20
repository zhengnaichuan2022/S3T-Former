import random
import matplotlib.pyplot as plt
import numpy as np
import pdb

import torch
import torch.nn.functional as F

def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    #crop
    # Handle case where p_interval may be integer or list
    if isinstance(p_interval, (int, float)):
        # If single number, convert to list
        p_interval = [p_interval]
    
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)# constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize - unify temporal window length
    # If window=-1, don't resize, directly return original data
    if window == -1:
        return data
    
    # Use F.interpolate to unify temporal dimension to window length
    # This is the key tool for unifying temporal window length in multi-attentionSGN
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data

def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M randomly choose a segment, not very reasonable because there are zeros
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    # Ensure zeros and ones have same dtype as rot
    zeros = torch.zeros(rot.shape[0], 1, dtype=rot.dtype, device=rot.device)  # T,1
    ones = torch.ones(rot.shape[0], 1, dtype=rot.dtype, device=rot.device)  # T,1

    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy, theta=0.3):
    """
    data_numpy: C,T,V,M (numpy array or torch tensor)
    Returns: numpy array (C,T,V,M)
    """
    # If numpy, convert to tensor
    if isinstance(data_numpy, np.ndarray):
        data_torch = torch.from_numpy(data_numpy).float()  # Ensure float32
    else:
        data_torch = data_numpy.float() if data_numpy.dtype != torch.float32 else data_numpy
    
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
    # Ensure rot is also float32 type
    rot = torch.zeros(3, dtype=torch.float32).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    # Return numpy array to maintain consistency
    if isinstance(data_torch, torch.Tensor):
        return data_torch.numpy()
    return data_torch

def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def random_noise(data_numpy, noise_std=0.01):
    """
    Add random Gaussian noise
    :param data_numpy: C,T,V,M
    :param noise_std: Noise standard deviation
    """
    C, T, V, M = data_numpy.shape
    noise = np.random.normal(0, noise_std, data_numpy.shape).astype(data_numpy.dtype)
    return data_numpy + noise


def random_scale(data_numpy, scale_range=[0.9, 1.1]):
    """
    Randomly scale skeleton data
    :param data_numpy: C,T,V,M
    :param scale_range: Scaling range [min, max]
    """
    C, T, V, M = data_numpy.shape
    scale = np.random.uniform(scale_range[0], scale_range[1])
    # Compute skeleton center
    center = data_numpy.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True)  # C,T,1,1
    data_numpy = (data_numpy - center) * scale + center
    return data_numpy


def random_mask(data_numpy, mask_ratio=0.1, mask_value=0.0):
    """
    Randomly mask some joints
    :param data_numpy: C,T,V,M
    :param mask_ratio: Masking ratio
    :param mask_value: Mask value (usually 0)
    """
    C, T, V, M = data_numpy.shape
    data_numpy = data_numpy.copy()
    num_mask = int(V * mask_ratio)
    if num_mask > 0:
        mask_indices = np.random.choice(V, num_mask, replace=False)
        data_numpy[:, :, mask_indices, :] = mask_value
    return data_numpy


def temporal_warp(data_numpy, sigma=0.2, knot=4):
    """
    Temporal warping: Apply nonlinear warping to temporal dimension
    :param data_numpy: C,T,V,M
    :param sigma: Warping strength
    :param knot: Number of control points
    """
    C, T, V, M = data_numpy.shape
    orig_steps = np.arange(T)
    
    # Generate random control points
    random_steps = (1.0 + sigma * np.random.randn(knot)).cumsum()
    random_steps = (random_steps - random_steps.min()) / (random_steps.max() - random_steps.min() + 1e-8)
    random_steps = (T - 1) * random_steps
    
    # Interpolate to generate warped temporal indices
    warp_steps = np.interp(orig_steps, np.linspace(0, T-1, knot), random_steps)
    warp_steps = np.clip(warp_steps, 0, T-1).astype(int)
    
    # Apply temporal warping
    data_numpy = data_numpy[:, warp_steps, :, :]
    return data_numpy


def random_flip(data_numpy, flip_prob=0.5):
    """
    Randomly flip horizontally (left-right flip)
    :param data_numpy: C,T,V,M
    :param flip_prob: Flip probability
    """
    if np.random.rand() < flip_prob:
        C, T, V, M = data_numpy.shape
        data_numpy = data_numpy.copy()
        # Flip x coordinate (usually the 0th channel)
        data_numpy[0, :, :, :] = -data_numpy[0, :, :, :]
    return data_numpy


def random_joint_swap(data_numpy, swap_prob=0.1):
    """
    Randomly swap symmetric joints (left-right symmetric)
    :param data_numpy: C,T,V,M
    :param swap_prob: Swap probability
    """
    # NTU skeleton symmetric joint pairs (left-right symmetric)
    # Note: This needs to be defined according to actual NTU skeleton structure
    # Here provides a general framework, actual use needs adjustment according to specific skeleton
    C, T, V, M = data_numpy.shape
    
    # NTU-25 skeleton symmetric joint pairs (example, needs adjustment according to actual skeleton)
    # If V=25, common symmetric pairs might be:
    # Left shoulder-right shoulder, left elbow-right elbow, left hand-right hand, left hip-right hip, left knee-right knee, left foot-right foot, etc.
    # Here provides a general implementation framework
    if np.random.rand() < swap_prob and V >= 2:
        data_numpy = data_numpy.copy()
        # Simple example: swap first half and second half joints (needs adjustment according to actual skeleton structure)
        # Actual use should define specific symmetric joint pairs
        pass  # Not implemented yet, needs specific skeleton structure information
    
    return data_numpy


def gaussian_blur_temporal(data_numpy, kernel_size=3, sigma=1.0):
    """
    Gaussian blur on temporal dimension (smoothing)
    :param data_numpy: C,T,V,M
    :param kernel_size: Kernel size (must be odd)
    :param sigma: Gaussian kernel standard deviation
    """
    try:
        from scipy import ndimage
    except ImportError:
        # If scipy not available, use numpy to implement simple Gaussian smoothing
        import warnings
        warnings.warn("scipy not available, using numpy-based Gaussian blur (slower)")
        return _gaussian_blur_numpy(data_numpy, sigma=sigma)
    
    C, T, V, M = data_numpy.shape
    data_numpy = data_numpy.copy()
    
    # Smooth temporal dimension for each channel, each node, each person separately
    for c in range(C):
        for v in range(V):
            for m in range(M):
                data_numpy[c, :, v, m] = ndimage.gaussian_filter1d(
                    data_numpy[c, :, v, m], sigma=sigma, mode='nearest'
                )
    
    return data_numpy


def _gaussian_blur_numpy(data_numpy, sigma=1.0):
    """
    Gaussian blur implemented using numpy (fallback solution)
    """
    C, T, V, M = data_numpy.shape
    data_numpy = data_numpy.copy()
    
    # Generate Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    center = kernel_size // 2
    kernel = np.exp(-0.5 * ((np.arange(kernel_size) - center) / sigma) ** 2)
    kernel = kernel / kernel.sum()
    
    # Smooth temporal dimension for each channel, each node, each person separately
    for c in range(C):
        for v in range(V):
            for m in range(M):
                signal = data_numpy[c, :, v, m]
                # Use convolution for smoothing (use mode='same' to maintain length)
                padded = np.pad(signal, (center, center), mode='edge')
                smoothed = np.convolve(padded, kernel, mode='valid')
                data_numpy[c, :, v, m] = smoothed
    
    return data_numpy


def random_dropout_frames(data_numpy, dropout_ratio=0.1):
    """
    Randomly drop some frames
    :param data_numpy: C,T,V,M
    :param dropout_ratio: Dropout ratio
    """
    C, T, V, M = data_numpy.shape
    num_drop = int(T * dropout_ratio)
    if num_drop > 0:
        drop_indices = np.random.choice(T, num_drop, replace=False)
        # Fill with previous or next frame (here use zero padding)
        data_numpy = data_numpy.copy()
        for idx in drop_indices:
            if idx > 0:
                data_numpy[:, idx, :, :] = data_numpy[:, idx-1, :, :]
            else:
                data_numpy[:, idx, :, :] = 0
    return data_numpy
