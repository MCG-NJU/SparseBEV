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


def sampling_4d(sample_points, mlvl_feats, scale_weights, lidar2img, image_h, image_w, eps=1e-5):
    """
    Args:
        sample_points: 3D sampling points in shape [B, Q, T, G, P, 3]
        mlvl_feats: list of multi-scale features from neck, each in shape [B*T*G, C, N, H, W]
        scale_weights: weights for multi-scale aggregation, [B, Q, G, T, P, L]
        lidar2img: 4x4 projection matrix in shape [B, TN, 4, 4]
    Symbol meaning:
        B: batch size
        Q: num of queries
        T: num of frames
        G: num of groups (we follow the group sampling mechanism of AdaMixer)
        P: num of sampling points per frame per group
        N: num of views (six for nuScenes)
        L: num of layers of feature pyramid (typically it is 4: C2, C3, C4, C5)
    """

    B, Q, T, G, P, _ = sample_points.shape  # [B, Q, T, G, P, 3]
    N = 6
    
    sample_points = sample_points.reshape(B, Q, T, G * P, 3)

    # get the projection matrix
    lidar2img = lidar2img[:, :, None, None, :, :]  # [B, TN, 1, 1, 4, 4]
    lidar2img = lidar2img.expand(B, T*N, Q, G * P, 4, 4)
    lidar2img = lidar2img.reshape(B, T, N, Q, G*P, 4, 4)

    # expand the points
    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)  # [B, Q, GP, 4]
    sample_points = sample_points[:, :, None, ..., None]     # [B, Q, T, GP, 4]
    sample_points = sample_points.expand(B, Q, N, T, G * P, 4, 1)
    sample_points = sample_points.transpose(1, 3)   # [B, T, N, Q, GP, 4, 1]

    # project 3d sampling points to N views
    sample_points_cam = torch.matmul(lidar2img, sample_points).squeeze(-1)  # [B, T, N, Q, GP, 4]

    # homo coord -> pixel coord
    homo = sample_points_cam[..., 2:3]
    homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, T, N, Q, GP, 2]

    # normalize
    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h

    # check if out of image
    valid_mask = ((homo > eps) \
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0)
    ).squeeze(-1).float()  # [B, T, N, Q, GP]

    # for visualization only
    if DUMP.enabled:
        torch.save(torch.cat([sample_points_cam, homo_nonzero], dim=-1).cpu(),
                   '{}/sample_points_cam_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
        torch.save(valid_mask.cpu(),
                   '{}/sample_points_cam_valid_mask_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

    valid_mask = valid_mask.permute(0, 1, 3, 4, 2)  # [B, T, Q, GP, N]
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 4, 2, 5)  # [B, T, Q, GP, N, 2]

    # prepare batched indexing
    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_time = torch.arange(T, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_time = i_time.view(1, T, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_query = i_query.view(1, 1, Q, 1, 1).expand(B, T, Q, G * P, 1)
    i_point = i_point.view(1, 1, 1, G * P, 1).expand(B, T, Q, G * P, 1)
    
    # we only keep at most one valid sampling point, see https://zhuanlan.zhihu.com/p/654821380
    i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, T, Q, GP, 1]

    # index the only one sampling point and its valid flag
    sample_points_cam = sample_points_cam[i_batch, i_time, i_query, i_point, i_view, :]  # [B, Q, GP, 1, 2]
    valid_mask = valid_mask[i_batch, i_time, i_query, i_point, i_view]  # [B, Q, GP, 1]

    # treat the view index as a new axis for grid_sample and normalize the view index to [0, 1]
    sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / (N - 1)], dim=-1)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    sample_points_cam = sample_points_cam.reshape(B, T, Q, G, P, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 2, 4, 5, 6)  # [B, T, G, Q, P, 1, 3]
    sample_points_cam = sample_points_cam.reshape(B*T*G, Q, P, 3)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    scale_weights = scale_weights.reshape(B, Q, G, T, P, -1)
    scale_weights = scale_weights.permute(0, 2, 3, 1, 4, 5)
    scale_weights = scale_weights.reshape(B*G*T, Q, P, -1)

    # multi-scale multi-view grid sample
    final = msmv_sampling(mlvl_feats, sample_points_cam, scale_weights)

    # reorganize the sampled features
    C = final.shape[2]  # [BTG, Q, C, P]
    final = final.reshape(B, T, G, Q, C, P)
    final = final.permute(0, 3, 2, 1, 5, 4)
    final = final.flatten(3, 4)  # [B, Q, G, FP, C]

    return final
