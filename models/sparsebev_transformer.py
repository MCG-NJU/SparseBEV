import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import bias_init_with_prob
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmdet.models.utils.builder import TRANSFORMER
from .bbox.utils import decode_bbox
from .utils import create_encoder, inverse_sigmoid, DUMP
from .sparsebev_sampling import sampling_4d, make_sample_points
from .checkpoint import checkpoint as cp


@TRANSFORMER.register_module()
class SparseBEVTransformer(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_layers=6, num_levels=4, num_classes=10, num_views=6, code_size=10, pc_range=[], return_intermediate=True, init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super(SparseBEVTransformer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_views = num_views
        self.return_intermediate = return_intermediate

        self.decoder = SparseBEVTransformerDecoder(embed_dims, num_frames, num_points, num_layers, num_levels, num_classes, num_views, code_size, pc_range=pc_range, return_intermediate=return_intermediate)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_bbox, query_feat, mlvl_feats, temp_query_feat, temp_query_bbox, attn_mask, img_metas, **data):
        cls_scores, bbox_preds, outs_dec = self.decoder(query_bbox, query_feat, mlvl_feats, temp_query_feat, temp_query_bbox, attn_mask, img_metas, **data)

        cls_scores = torch.nan_to_num(cls_scores)
        bbox_preds = torch.nan_to_num(bbox_preds)
        outs_dec = torch.nan_to_num(outs_dec)

        return cls_scores, bbox_preds, outs_dec


class SparseBEVTransformerDecoder(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_layers=6, num_levels=4, num_classes=10, num_views=6, code_size=10, pc_range=[], return_intermediate=True, init_cfg=None):
        super(SparseBEVTransformerDecoder, self).__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.num_views = num_views
        self.return_intermediate = return_intermediate
        # params are shared across all decoder layers
        self.decoder_layer = SparseBEVTransformerDecoderLayer(
            embed_dims, num_frames, num_points, num_levels, num_classes, num_views, code_size, pc_range=pc_range
        )

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layer.init_weights()

    def forward(self, query_bbox, query_feat, mlvl_feats, temp_query_feat, temp_query_bbox, attn_mask, img_metas):
        
        cls_scores, bbox_preds, outs_dec = [], [], []
        temp_pos = None
        self.num_layers = 4 if not self.training else 6
        for i in range(self.num_layers):
            DUMP.stage_count = i

            query_feat, cls_score, bbox_pred, temp_pos = self.decoder_layer(
                query_bbox, query_feat, mlvl_feats, temp_query_feat, temp_query_bbox, temp_pos, attn_mask, img_metas
            )
            query_bbox = bbox_pred.clone().detach()

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            outs_dec.append(query_feat)

        cls_scores = torch.stack(cls_scores)
        bbox_preds = torch.stack(bbox_preds)
        outs_dec = torch.stack(outs_dec)

        return cls_scores, bbox_preds, outs_dec


class SparseBEVTransformerDecoderLayer(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_levels=4, num_classes=10, num_views=6, code_size=10, num_cls_fcs=2, num_reg_fcs=2, pc_range=[], init_cfg=None):
        super(SparseBEVTransformerDecoderLayer, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_views = num_views
        self.code_size = code_size
        self.pc_range = pc_range

        self.position_encoder = QueryBoxEncoder(embed_dims, code_size=self.code_size)
        
        self.self_attn = SparseBEVSelfAttention(embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        #self.self_attn = SelfAdaptiveAttention(embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range, num_per_frame=256)

        self.cross_attn = TimeAdaptiveAttention(embed_dims, num_heads=8, dropout=0.1, frames=4, num_per_frame=256)
        #self.cross_attn = MultiheadAttention(embed_dims, num_heads=8, dropout=0.1, batch_first=True)

        self.sampling = SparseBEVSampling(embed_dims, num_frames=num_frames, num_groups=4, num_points=num_points, num_levels=num_levels, num_views=num_views, pc_range=pc_range)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points, n_groups=4, out_points=128)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.norm_temp = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        self.reg_branch = nn.Sequential(*reg_branch)

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def refine_bbox(self, bbox_proposal, bbox_delta):
        xyz = inverse_sigmoid(bbox_proposal[..., 0:3])
        xyz_delta = bbox_delta[..., 0:3]
        xyz_new = torch.sigmoid(xyz_delta + xyz)

        return torch.cat([xyz_new, bbox_delta[..., 3:]], dim=-1)

    def forward(self, query_bbox, query_feat, mlvl_feats, temp_query_feat, temp_query_bbox, temp_pos, attn_mask, img_metas):
        """
        query_bbox: [B, Q, 10] [cx, cy, cz, w, h, d, rot.sin, rot.cos, vx, vy]
        """
        query_pos = self.position_encoder(query_bbox)

        # temporal attn
        if self.training:
            if temp_query_bbox is not None:
                temp_pos = self.position_encoder(temp_query_bbox)
            else:
                temp_pos = None
        else:
            if DUMP.stage_count == 0 and temp_query_bbox is not None:
                temp_pos = self.position_encoder(temp_query_bbox)

        query_feat = self.norm_temp(self.cross_attn(query_feat, temp_query_feat,
                                                    query_pos=query_pos, key_pos=temp_pos))
        
        query_feat = query_feat + query_pos
        query_feat = self.norm1(self.self_attn(query_bbox, query_feat, attn_mask))

        sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas)
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        cls_score = self.cls_branch(query_feat)  # [B, Q, num_classes]
        bbox_pred = self.reg_branch(query_feat)  # [B, Q, code_size]
        bbox_pred = self.refine_bbox(query_bbox, bbox_pred)

        # calculate absolute velocity according to time difference
        time_diff = img_metas[0]['time_diff']  # [B, F]
        if time_diff.shape[1] > 1:
            time_diff = time_diff.clone()
            time_diff[time_diff < 1e-5] = 1.0
            bbox_pred[..., 8:] = bbox_pred[..., 8:] / time_diff[:, 1:2, None]

        if DUMP.enabled:
            query_bbox_dec = decode_bbox(query_bbox, self.pc_range)
            bbox_pred_dec = decode_bbox(bbox_pred, self.pc_range)
            cls_score_sig = torch.sigmoid(cls_score)
            torch.save(query_bbox_dec.cpu(), '{}/query_bbox_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
            torch.save(bbox_pred_dec.cpu(), '{}/bbox_pred_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
            torch.save(cls_score_sig.cpu(), '{}/cls_score_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        return query_feat, cls_score, bbox_pred, temp_pos

class QueryBoxEncoder(BaseModule):
    def __init__(self, embed_dims=256, code_size=10):
        super().__init__()
        self.embed_dims = embed_dims
        self.code_size = code_size
        self.xyz_encoder = create_encoder(3, embed_dims)
        self.whl_encoder = create_encoder(3, embed_dims)
        self.yaw_encoder = create_encoder(2, embed_dims)
        if code_size > 8:
            self.vel_encoder = create_encoder(2, embed_dims)

        self.output_fc = create_encoder(embed_dims, embed_dims)

    def forward(self, query_bbox):
        xyz_feat = self.xyz_encoder(query_bbox[..., :3])
        whl_feat = self.whl_encoder(query_bbox[..., 3:6])
        yaw_feat = self.yaw_encoder(query_bbox[..., 6:8])
        output = xyz_feat + whl_feat + yaw_feat
        if self.code_size > 8:
            vel_feat = self.vel_encoder(query_bbox[..., 8:10])
            output = output + vel_feat
        
        output = self.output_fc(output)
        return output

class SparseBEVSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=[], init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_bbox, query_feat, pre_attn_mask):
        """
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        """
        dist = self.calc_bbox_dists(query_bbox)
        tau = self.gen_tau(query_feat)  # [B, Q, 8]

        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]

        if pre_attn_mask is not None:  # for query denoising
            attn_mask[:, :, pre_attn_mask] = float('-inf')

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_bbox, query_feat, pre_attn_mask):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_bbox, query_feat, pre_attn_mask, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, pre_attn_mask)

    @torch.no_grad()
    def calc_bbox_dists(self, bboxes):
        centers = decode_bbox(bboxes, self.pc_range)[..., :2]  # [B, Q, 2]

        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)  # [B, Q, Q]
        dist = -dist

        return dist

class TimeAdaptiveAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, frames=4, num_per_frame=256, init_cfg=None):
        super().__init__(init_cfg)

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

        self.num_per_frame = num_per_frame

        cross_dist = self.pre_calc_time_interval(frames)
        self.register_buffer('cross_dist', cross_dist)
        self.register_buffer('query_dist', torch.ones(1, dtype=torch.float32))

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def pre_calc_time_interval(self, frames):
        """
        query_time_stamp: 0  
        key_time_stamp: [1, 2, ..., T]
        """
        key_time_stamp = torch.arange(frames, dtype=torch.float32) + 1  #[T]
        dist = key_time_stamp[None, None, :].repeat_interleave(self.num_per_frame, dim=-1)  # [1, 1, K]

        dist = -dist  # [1, 1, K]

        return dist
    
    def inner_forward(self, query_feat, temp_feat, query_pos, key_pos):
        """
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        """
        if temp_feat is not None:
            B, Q = query_feat.shape[:2]
            dist = self.cross_dist[..., :temp_feat.shape[1]].expand(B, Q, -1)  # [1, 1, K]
        else:
            B, Q = query_feat.shape[:2]
            dist = self.query_dist.expand(B, Q, Q)  # [B, Q, Q]

        tau = self.gen_tau(query_feat)  # [B, Q, 8]

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, K]

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, K]
        
        return self.attention(query_feat, temp_feat, temp_feat, query_pos=query_pos, key_pos=key_pos, attn_mask=attn_mask)

    def forward(self, query_feat, temp_feat, query_pos, key_pos):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward,  query_feat, temp_feat, query_pos, key_pos, use_reentrant=False)
        else:
            return self.inner_forward(query_feat, temp_feat, query_pos, key_pos)

    @torch.no_grad()
    def calc_time_interval(self, query_time_stamp, key_time_stamp, bs, num_query):
        """
        query_time_stamp: 0  
        key_time_stamp: [1, 2, ..., T]
        """
        dist = key_time_stamp[None, :] - query_time_stamp[:, None] # [1, T]
        dist = dist[None, :, :].expand(bs, num_query, -1).repeat_interleave(self.num_per_frame, dim=-1)  # [B, Q, K]

        dist = -dist  # [B, Q, K]

        return dist

class SparseBEVSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims=256, num_frames=4, num_groups=4, num_points=8, num_levels=4, num_views=6, pc_range=[], init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.num_views = num_views
        self.pc_range = pc_range

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_bbox, query_feat, mlvl_feats, img_metas):
        '''
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        '''
        #assert mlvl_feats[0].shape[1] % self.num_views == 0
        B, Q = query_bbox.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, self.num_groups * self.num_points, 3)
        sampling_points = make_sample_points(query_bbox, sampling_offset, self.pc_range)  # [B, Q, GP, 3]

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        #scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            img_metas[0]['lidar2img'],
            image_h, image_w,
            self.num_views
        )  # [B, Q, G, FP, C]

        return sampled_feats

    def forward(self, query_bbox, query_feat, mlvl_feats, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_bbox, query_feat, mlvl_feats, img_metas, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, mlvl_feats, img_metas)


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super(AdaptiveMixing, self).__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)