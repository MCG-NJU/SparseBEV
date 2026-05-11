import math
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
from mmdet3d.core.bbox.structures.utils import limit_period

def create_encoder(input_dim, embed_dims):
    return nn.Sequential(
        nn.Linear(input_dim, embed_dims),
        nn.ReLU(inplace=True),
        nn.LayerNorm(embed_dims),
        nn.Linear(embed_dims, embed_dims),
        nn.ReLU(inplace=True),
        nn.LayerNorm(embed_dims)
    )

def history_bbox_warp(bbox, T_temp2cur, time_diff, pc_range=None):

    T_temp2cur = torch.unsqueeze(T_temp2cur,  dim=1)  # [B, 1, 4, 4]
    
    pc_range_min = pc_range[:3]
    pc_range_max = pc_range[3:6]
    pc_range_scale = pc_range_max - pc_range_min
    
    out_bbox = bbox.clone()
    center = out_bbox[..., :3] * pc_range_scale + pc_range_min  # [B, K, 3]
    
    # there is no vel_dim for box in waymo
    if bbox.shape[-1] > 8:
        # warp center based on velocity
        vel = bbox[..., -2:]  # [B, K, 2]
        dist = vel * time_diff[..., None, None].to(dtype=vel.dtype)  # [B, 1, 1]
        center = torch.cat([
            center[..., 0:2] + dist,
            center[..., 2:3]
        ], dim=-1)

    center = torch.cat([center, torch.ones_like(center[..., :1])], dim=-1)  # [B, K, 4]
    
    yaw = torch.cat([out_bbox[..., 6:8].flip(-1), torch.zeros_like(center[..., :2])], dim=-1)  # [B, K, 4]
    vel_mat = torch.cat([out_bbox[..., -2:], torch.zeros_like(center[..., :2])], dim=-1)  # [B, K, 4]
    box_mat = torch.stack([center, yaw, vel_mat], dim=-1)  # [B, K, 4, 3]

    warpped_box = T_temp2cur @ box_mat  # [B, 1, 4, 4] @ [B, K, 4, 3] -> [B, K, 4, 3]
    
    transformed_center = warpped_box[..., :3, 0]  # [B, K, 3]
    
    out_bbox[..., :3] = (transformed_center - pc_range_min) / pc_range_scale
    out_bbox[..., 6:8] = warpped_box[..., :2, 1].flip(-1)  # [B, K, 2]
    if bbox.shape[-1] > 8:
        out_bbox[..., -2:] = warpped_box[..., :2, 2]  # [B, K, 2]
    
    return out_bbox

def batch_indexing(batched_data: torch.Tensor, batched_indices: torch.Tensor, layout='channel_first'):
    def batch_indexing_channel_first(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, C, N]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, C, I1, I2, ..., Im]
        """
        def product(arr):
            p = 1
            for i in arr:
                p *= i
            return p
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size, n_channels = batched_data.shape[:2]
        indices_shape = list(batched_indices.shape[1:])
        batched_indices = batched_indices.reshape([batch_size, 1, -1])
        batched_indices = batched_indices.expand([batch_size, n_channels, product(indices_shape)])
        result = torch.gather(batched_data, dim=2, index=batched_indices.to(torch.int64))
        result = result.view([batch_size, n_channels] + indices_shape)
        return result

    def batch_indexing_channel_last(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, N, C]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, I1, I2, ..., Im, C]
        """
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size = batched_data.shape[0]
        view_shape = [batch_size] + [1] * (len(batched_indices.shape) - 1)
        expand_shape = [batch_size] + list(batched_indices.shape)[1:]
        indices_of_batch = torch.arange(batch_size, dtype=torch.long, device=batched_data.device)
        indices_of_batch = indices_of_batch.view(view_shape).expand(expand_shape)  # [bs, I1, I2, ..., Im]
        if len(batched_data.shape) == 2:
            return batched_data[indices_of_batch, batched_indices.to(torch.long)]
        else:
            return batched_data[indices_of_batch, batched_indices.to(torch.long), :]

    if layout == 'channel_first':
        return batch_indexing_channel_first(batched_data, batched_indices)
    elif layout == 'channel_last':
        return batch_indexing_channel_last(batched_data, batched_indices)
    else:
        raise ValueError

# https://github.com/exiawsh/StreamPETR/blob/2315cf9f077817ec7089c87094ba8a63f76c2acf/projects/mmdet3d_plugin/models/utils/misc.py#L7
def memory_refresh(memory, prev_exist):
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:]) 
    return memory * prev_exist


def transform_reference_points(reference_points, egopose, reverse=False, translation=True):
    reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
    if reverse:
        matrix = egopose.inverse()
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0
    reference_points = (matrix.unsqueeze(1) @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]
    return reference_points

def transform_angle(angle, rot_mat):
    rot_mat_T = rot_mat.T
    angle_delta = torch.atan2(rot_mat_T[0, 1], rot_mat_T[0, 0]).unsqueeze(-1)  # [B, 1]
    angle += angle_delta
    angle = limit_period(angle, period=np.pi * 2)

    return angle
    

class GridMask(nn.Module):
    def __init__(self, ratio=0.5, prob=0.7):
        super(GridMask, self).__init__()
        self.ratio = ratio
        self.prob = prob

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x

        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)

        d = np.random.randint(2, h)
        l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.uint8)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        for i in range(hh // d):
            s = d*i + st_h
            t = min(s + l, hh)
            mask[s:t, :] = 0

        for i in range(ww // d):
            s = d*i + st_w
            t = min(s + l, ww)
            mask[:, s:t] = 0

        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]
        mask = torch.tensor(mask, dtype=x.dtype, device=x.device)
        mask = 1 - mask
        mask = mask.expand_as(x)
        x = x * mask 

        return x.view(n, c, h, w)


def rotation_3d_in_axis(points, angles):
    assert points.shape[-1] == 3
    assert angles.shape[-1] == 1
    angles = angles[..., 0]

    n_points = points.shape[-2]
    input_dims = angles.shape

    if len(input_dims) > 1:
        points = points.reshape(-1, n_points, 3)
        angles = angles.reshape(-1)

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if VERSION.name == 'v0.17.1':
        rot_mat_T = torch.stack([
            rot_cos, -rot_sin, zeros,
            rot_sin, rot_cos, zeros,
            zeros, zeros, ones,
        ]).transpose(0, 1).reshape(-1, 3, 3)
    else:
        rot_mat_T = torch.stack([
            rot_cos, rot_sin, zeros,
            -rot_sin, rot_cos, zeros,
            zeros, zeros, ones,
        ]).transpose(0, 1).reshape(-1, 3, 3)

    points = torch.bmm(points, rot_mat_T)

    if len(input_dims) > 1:
        points = points.reshape(*input_dims, n_points, 3)
    
    return points


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pad_multiple(inputs, img_metas, size_divisor=32, training=False):
    _, _, img_h, img_w = inputs.shape

    pad_h = 0 if img_h % size_divisor == 0 else size_divisor - (img_h % size_divisor)
    pad_w = 0 if img_w % size_divisor == 0 else size_divisor - (img_w % size_divisor)

    if training:
        T = len(img_metas)
        B = len(img_metas[0])
        N = len(img_metas[0][0]['ori_shape'])

        for t in range(T):
            for b in range(B):
                img_metas[t][b]['img_shape'] = [(img_h + pad_h, img_w + pad_w, 3) for _ in range(N)]
                img_metas[t][b]['pad_shape'] = [(img_h + pad_h, img_w + pad_w, 3) for _ in range(N)]

    else:
        B = len(img_metas)
        N = len(img_metas[0]['ori_shape'])

        for b in range(B):
            img_metas[b]['img_shape'] = [(img_h + pad_h, img_w + pad_w, 3) for _ in range(N)]
            img_metas[b]['pad_shape'] = [(img_h + pad_h, img_w + pad_w, 3) for _ in range(N)]

    if pad_h == 0 and pad_w == 0:
        return inputs
    else:
        return F.pad(inputs, [0, pad_w, 0, pad_h], value=0)


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    .. image:: _static/img/rgb_to_hsv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    image = image / 255.0

    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0

    h = h * 360.0
    v = v * 255.0

    return torch.stack((h, s, v), dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :] / 360.0
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :] / 255.0

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)
    out = out * 255.0

    return out


class GpuPhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, imgs):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = imgs[:, [2, 1, 0], :, :]  # BGR to RGB

        contrast_modes = []
        for _ in range(imgs.shape[0]):
            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            contrast_modes.append(random.randint(2))

        for idx in range(imgs.shape[0]):
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                imgs[idx] += delta

            if contrast_modes[idx] == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    imgs[idx] *= alpha

        # convert color from BGR to HSV
        imgs = rgb_to_hsv(imgs)

        for idx in range(imgs.shape[0]):
            # random saturation
            if random.randint(2):
                imgs[idx, 1] *= random.uniform(self.saturation_lower, self.saturation_upper)

            # random hue
            if random.randint(2):
                imgs[idx, 0] += random.uniform(-self.hue_delta, self.hue_delta)

        imgs[:, 0][imgs[:, 0] > 360] -= 360
        imgs[:, 0][imgs[:, 0] < 0] += 360

        # convert color from HSV to BGR
        imgs = hsv_to_rgb(imgs)

        for idx in range(imgs.shape[0]):
            # random contrast
            if contrast_modes[idx] == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    imgs[idx] *= alpha

            # randomly swap channels
            if random.randint(2):
                imgs[idx] = imgs[idx, random.permutation(3)]

        imgs = imgs[:, [2, 1, 0], :, :]  # RGB to BGR

        return imgs


class DumpConfig:
    def __init__(self):
        self.enabled = False
        self.out_dir = tempfile.mkdtemp()
        self.stage_count = 0
        self.frame_count = 0


DUMP = DumpConfig()


# for backward compatibility
class Version:
    def __init__(self):
        self.name = 'v1.0.0'

VERSION = Version()
