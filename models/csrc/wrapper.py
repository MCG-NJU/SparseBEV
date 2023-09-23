import torch
import torch.nn.functional as F

try:
    from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c2345_forward, _ms_deform_attn_cuda_c2345_backward
    from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c23456_forward, _ms_deform_attn_cuda_c23456_backward
    MSMV_CUDA = True
except ImportError as e:
    print('Warning: failed to load one or more CUDA extensions, performance may be hurt.')
    print('Error message:', e)
    MSMV_CUDA = False


def msmv_sampling_pytorch(mlvl_feats, sampling_locations, scale_weights):
    """
    value: [B, N, H1W1 + H2W2..., C]
    sampling_locations: [B, Q, P, 3]
    scale_weights: [B, Q, P, 4]
    """
    assert scale_weights.shape[-1] == len(mlvl_feats)

    B, C, _, _, _ = mlvl_feats[0].shape
    _, Q, P, _ = sampling_locations.shape

    sampling_locations = sampling_locations * 2 - 1
    sampling_locations = sampling_locations[:, :, :, None, :]  # [B, Q, P, 1, 3]

    final = torch.zeros([B, C, Q, P], device=mlvl_feats[0].device)

    for lvl, feat in enumerate(mlvl_feats):
        out = F.grid_sample(
            feat, sampling_locations, mode='bilinear',
            padding_mode='zeros', align_corners=True,
        )[..., 0]  # [B, C, Q, P]
        out = out * scale_weights[..., lvl].reshape(B, 1, Q, P)
        final += out

    return final.permute(0, 2, 1, 3)


class MSMVSamplingC2345(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights):
        ctx.save_for_backward(feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights)
        
        assert callable(_ms_deform_attn_cuda_c2345_forward)
        return _ms_deform_attn_cuda_c2345_forward(
            feat_c2, feat_c3, feat_c4, feat_c5,
            sampling_locations, scale_weights)

    @staticmethod
    def backward(ctx, grad_output):
        feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights = ctx.saved_tensors

        assert callable(_ms_deform_attn_cuda_c2345_backward)
        grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight = _ms_deform_attn_cuda_c2345_backward(grad_output.contiguous(), 
            feat_c2, feat_c3, feat_c4, feat_c5,
            sampling_locations, scale_weights
        )
        
        return grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight


class MSMVSamplingC23456(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights):
        ctx.save_for_backward(feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights)
        
        assert callable(_ms_deform_attn_cuda_c23456_forward)
        return _ms_deform_attn_cuda_c23456_forward(
            feat_c2, feat_c3, feat_c4, feat_c5, feat_c6,
            sampling_locations, scale_weights)

    @staticmethod
    def backward(ctx, grad_output):
        feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights = ctx.saved_tensors

        assert callable(_ms_deform_attn_cuda_c23456_backward)
        grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_value_c6, grad_sampling_loc, grad_attn_weight = _ms_deform_attn_cuda_c23456_backward(grad_output.contiguous(), 
            feat_c2, feat_c3, feat_c4, feat_c5, feat_c6,
            sampling_locations, scale_weights
        )
        
        return grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_value_c6, grad_sampling_loc, grad_attn_weight


def msmv_sampling(mlvl_feats, sampling_locations, scale_weights):
    if len(mlvl_feats) == 4 and MSMV_CUDA:
        return MSMVSamplingC2345.apply(*mlvl_feats, sampling_locations, scale_weights)
    elif len(mlvl_feats) == 5 and MSMV_CUDA:
        return MSMVSamplingC23456.apply(*mlvl_feats, sampling_locations, scale_weights)
    else:
        return msmv_sampling_pytorch(mlvl_feats, sampling_locations, scale_weights)
