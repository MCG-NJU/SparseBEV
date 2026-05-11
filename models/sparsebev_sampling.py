import torch
import torch.nn.functional as F
from .bbox.utils import decode_bbox
from .utils import rotation_3d_in_axis, DUMP
from .csrc.wrapper import msmv_sampling, msmv_sampling_pytorch


def make_sample_points(query_bbox, offset, pc_range):
    '''
    query_bbox: [B, Q, 10]
    offset: [B, Q, num_points, 4], normalized by stride
    '''
    query_bbox = decode_bbox(query_bbox, pc_range)  # [B, Q, 9]

    xyz = query_bbox[..., 0:3]  # [B, Q, 3]
    wlh = query_bbox[..., 3:6]  # [B, Q, 3]
    ang = query_bbox[..., 6:7]  # [B, Q, 1]

    delta_xyz = offset[..., 0:3]  # [B, Q, P, 3]
    delta_xyz = wlh[:, :, None, :] * delta_xyz  # [B, Q, P, 3]
    delta_xyz = rotation_3d_in_axis(delta_xyz, ang)  # [B, Q, P, 3]
    sample_xyz = xyz[:, :, None, :] + delta_xyz  # [B, Q, P, 3]

    return sample_xyz  # [B, Q, P, 3]


def sampling_4d(sample_points, mlvl_feats, scale_weights, lidar2img, image_h, image_w, num_views=6, eps=1e-5):
    """
    Args:
        sample_points: 3D sampling points in shape [B, Q, G*P, 3]
        mlvl_feats: list of multi-scale features from neck, each in shape [B*G, C, N, H, W]
        scale_weights: weights for multi-scale aggregation, [B, Q, G, P, L]
        lidar2img: 4x4 projection matrix in shape [B, N, 4, 4]
    """

    
    B, Q, G, P, _ = scale_weights.shape  # [B, Q, G, P, L]
    N = num_views

    #sample_points = sample_points.reshape(B, Q, G * P, 3)

    # get the projection matrix
    lidar2img = lidar2img[:, None, None, :, :, :]  # [B, 1, 1, N, 4, 4]
    lidar2img = lidar2img.expand(B, Q, G * P, N, 4, 4)
    #lidar2img = lidar2img.reshape(B, Q, G * P, N, 4, 4)

    # expand the points
    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)  # [B, Q, GP, 4]
    sample_points = sample_points[..., None, :, None]  # [B, Q, GP, 1, 4, 1]
    sample_points = sample_points.expand(B, Q, G * P, N, 4, 1)

    # project 3d sampling points to N views
    sample_points_cam = torch.matmul(lidar2img, sample_points).squeeze(-1)  # [B, Q, GP, N, 4]

    # homo coord -> pixel coord
    homo = sample_points_cam[..., 2:3]
    homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, Q, GP, N, 2]

    # normalize
    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h

    # check if out of image
    valid_mask = ((homo > eps) \
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0)
    ).squeeze(-1).float()  # [B, Q, GP, N]

    #valid_mask = valid_mask.permute(0, 1, 2, 3)  # [B, Q, GP, N]
    #sample_points_cam = sample_points_cam.permute(0, 1, 2, 3, 4)  # [B, Q, GP, N, 2]

    # prepare batched indexing
    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1).expand(B, Q, G * P, 1)
    i_query = i_query.view(1, Q, 1, 1).expand(B, Q, G * P, 1)
    i_point = i_point.view(1, 1, G * P, 1).expand(B, Q, G * P, 1)

    # we only keep at most one valid sampling point, see https://zhuanlan.zhihu.com/p/654821380
    i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, Q, GP, 1]

    # index the only one sampling point and its valid flag
    sample_points_cam = sample_points_cam[i_batch, i_query, i_point, i_view, :]  # [B, Q, GP, 1, 2]
    valid_mask = valid_mask[i_batch, i_query, i_point, i_view]  # [B, Q, GP, 1]

    # treat the view index as a new axis for grid_sample and normalize the view index to [0, 1]
    sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / (N - 1)], dim=-1)

    # reorganize the tensor to stack G to the batch dim for better parallelism
    sample_points_cam = sample_points_cam.reshape(B, Q, G, P, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 2, 1, 3, 4, 5)  # [B, G, Q, P, 1, 3]
    sample_points_cam = sample_points_cam.reshape(B*G, Q, P, 3)

    # reorganize the tensor to stack G to the batch dim for better parallelism
    #scale_weights = scale_weights.reshape(B, Q, G, P, -1)
    scale_weights = scale_weights.permute(0, 2, 1, 3, 4)
    scale_weights = scale_weights.reshape(B*G, Q, P, -1)

    sample_points_cam = sample_points_cam.contiguous()
    scale_weights = scale_weights.contiguous()
    # multi-scale multi-view grid sample
    final = msmv_sampling(mlvl_feats, sample_points_cam, scale_weights)

    # reorganize the sampled features
    C = final.shape[2]  # [BG, Q, C, P]
    final = final.reshape(B, G, Q, C, P)
    final = final.permute(0, 2, 1, 4, 3)  # [B, Q, G*P, C]
    #final = final.flatten(2, 3)  # [B, Q, G*P, C]

    return final
