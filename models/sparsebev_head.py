import math
import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from .bbox.utils import normalize_bbox, encode_bbox
from .utils import VERSION


@HEADS.register_module()
class SparseBEVHead(DETRHead):
    def __init__(self,
                 *args,
                 num_classes,
                 in_channels,
                 query_denoising=True,
                 query_denoising_groups=10,
                 bbox_coder=None,
                 code_size=10,
                 code_weights=[1.0] * 10,
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 **kwargs):
        self.code_size = code_size
        self.code_weights = code_weights
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = in_channels

        super(SparseBEVHead, self).__init__(num_classes, in_channels, train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)

        self.code_weights = nn.Parameter(torch.tensor(self.code_weights), requires_grad=False)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range

        self.dn_enabled = query_denoising
        self.dn_group_num = query_denoising_groups
        self.dn_weight = 1.0
        self.dn_bbox_noise_scale = 0.5
        self.dn_label_noise_scale = 0.5

    def _init_layers(self):
        self.init_query_bbox = nn.Embedding(self.num_query, 10)  # (x, y, z, w, l, h, sin, cos, vx, vy)
        self.label_enc = nn.Embedding(self.num_classes + 1, self.embed_dims - 1)  # DAB-DETR

        nn.init.zeros_(self.init_query_bbox.weight[:, 2:3])
        nn.init.zeros_(self.init_query_bbox.weight[:, 8:10])
        nn.init.constant_(self.init_query_bbox.weight[:, 5:6], 1.5)

        grid_size = int(math.sqrt(self.num_query))
        assert grid_size * grid_size == self.num_query
        x = y = torch.arange(grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')  # [0, grid_size - 1]
        xy = torch.cat([xx[..., None], yy[..., None]], dim=-1)
        xy = (xy + 0.5) / grid_size  # [0.5, grid_size - 0.5] / grid_size ~= (0, 1)
        with torch.no_grad():
            self.init_query_bbox.weight[:, :2] = xy.reshape(-1, 2)  # [Q, 2]

    def init_weights(self):
        self.transformer.init_weights()

    def forward(self, mlvl_feats, img_metas):
        query_bbox = self.init_query_bbox.weight.clone()  # [Q, 10]
        #query_bbox[..., :3] = query_bbox[..., :3].sigmoid()

        # query denoising
        B = mlvl_feats[0].shape[0]
        query_bbox, query_feat, attn_mask, mask_dict = self.prepare_for_dn_input(B, query_bbox, self.label_enc, img_metas)

        cls_scores, bbox_preds = self.transformer(
            query_bbox,
            query_feat,
            mlvl_feats,
            attn_mask=attn_mask,
            img_metas=img_metas,
        )

        bbox_preds[..., 0] = bbox_preds[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        bbox_preds[..., 1] = bbox_preds[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        bbox_preds[..., 2] = bbox_preds[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        bbox_preds = torch.cat([
            bbox_preds[..., 0:2],
            bbox_preds[..., 3:5],
            bbox_preds[..., 2:3],
            bbox_preds[..., 5:10],
        ], dim=-1)  # [cx, cy, w, l, cz, h, sin, cos, vx, vy]

        if mask_dict is not None and mask_dict['pad_size'] > 0:  # if using query denoising
            output_known_cls_scores = cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_bbox_preds = bbox_preds[:, :, :mask_dict['pad_size'], :]
            output_cls_scores = cls_scores[:, :, mask_dict['pad_size']:, :]
            output_bbox_preds = bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes'] = (output_known_cls_scores, output_known_bbox_preds)
            outs = {
                'all_cls_scores': output_cls_scores,
                'all_bbox_preds': output_bbox_preds,
                'enc_cls_scores': None,
                'enc_bbox_preds': None, 
                'dn_mask_dict': mask_dict,
            }
        else:
            outs = {
                'all_cls_scores': cls_scores,
                'all_bbox_preds': bbox_preds,
                'enc_cls_scores': None,
                'enc_bbox_preds': None, 
            }

        return outs

    def prepare_for_dn_input(self, batch_size, init_query_bbox, label_enc, img_metas):
        # mostly borrowed from:
        #  - https://github.com/IDEA-Research/DN-DETR/blob/main/models/DN_DAB_DETR/dn_components.py
        #  - https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/models/dense_heads/petrv2_dnhead.py

        device = init_query_bbox.device
        indicator0 = torch.zeros([self.num_query, 1], device=device)
        init_query_feat = label_enc.weight[self.num_classes].repeat(self.num_query, 1)
        init_query_feat = torch.cat([init_query_feat, indicator0], dim=1)

        if self.training and self.dn_enabled:
            targets = [{
                'bboxes': torch.cat([m['gt_bboxes_3d'].gravity_center,
                                     m['gt_bboxes_3d'].tensor[:, 3:]], dim=1).cuda(),
                'labels': m['gt_labels_3d'].cuda().long()
            } for m in img_metas]

            known = [torch.ones_like(t['labels'], device=device) for t in targets]
            known_num = [sum(k) for k in known]

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets]).clone()
            bboxes = torch.cat([t['bboxes'] for t in targets]).clone()
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # add noise
            known_indice = known_indice.repeat(self.dn_group_num, 1).view(-1)
            known_labels = labels.repeat(self.dn_group_num, 1).view(-1)
            known_bid = batch_idx.repeat(self.dn_group_num, 1).view(-1)
            known_bboxs = bboxes.repeat(self.dn_group_num, 1) # 9
            known_labels_expand = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            # noise on the box
            if self.dn_bbox_noise_scale > 0:
                wlh = known_bbox_expand[..., 3:6].clone()
                rand_prob = torch.rand_like(known_bbox_expand) * 2 - 1.0
                known_bbox_expand[..., 0:3] += torch.mul(rand_prob[..., 0:3], wlh / 2) * self.dn_bbox_noise_scale
                # known_bbox_expand[..., 3:6] += torch.mul(rand_prob[..., 3:6], wlh) * self.dn_bbox_noise_scale
                # known_bbox_expand[..., 6:7] += torch.mul(rand_prob[..., 6:7], 3.14159) * self.dn_bbox_noise_scale

            known_bbox_expand = encode_bbox(known_bbox_expand, self.pc_range)
            known_bbox_expand[..., 0:3].clamp_(min=0.0, max=1.0)
            # nn.init.constant(known_bbox_expand[..., 8:10], 0.0)

            # noise on the label
            if self.dn_label_noise_scale > 0:
                p = torch.rand_like(known_labels_expand.float())
                chosen_indice = torch.nonzero(p < self.dn_label_noise_scale).view(-1)  # usually half of bbox noise
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                known_labels_expand.scatter_(0, chosen_indice, new_label)

            known_feat_expand = label_enc(known_labels_expand)
            indicator1 = torch.ones([known_feat_expand.shape[0], 1], device=device)  # add dn part indicator
            known_feat_expand = torch.cat([known_feat_expand, indicator1], dim=1)

            # construct final query
            dn_single_pad = int(max(known_num))
            dn_pad_size = int(dn_single_pad * self.dn_group_num)
            dn_query_bbox = torch.zeros([dn_pad_size, init_query_bbox.shape[-1]], device=device)
            dn_query_feat = torch.zeros([dn_pad_size, self.embed_dims], device=device)
            input_query_bbox = torch.cat([dn_query_bbox, init_query_bbox], dim=0).repeat(batch_size, 1, 1)
            input_query_feat = torch.cat([dn_query_feat, init_query_feat], dim=0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + dn_single_pad * i for i in range(self.dn_group_num)]).long()

            if len(known_bid):
                input_query_bbox[known_bid.long(), map_known_indice] = known_bbox_expand
                input_query_feat[(known_bid.long(), map_known_indice)] = known_feat_expand

            total_size = dn_pad_size + self.num_query
            attn_mask = torch.ones([total_size, total_size], device=device) < 0

            # match query cannot see the reconstruct
            attn_mask[dn_pad_size:, :dn_pad_size] = True
            for i in range(self.dn_group_num):
                if i == 0:
                    attn_mask[dn_single_pad * i:dn_single_pad * (i + 1), dn_single_pad * (i + 1):dn_pad_size] = True
                if i == self.dn_group_num - 1:
                    attn_mask[dn_single_pad * i:dn_single_pad * (i + 1), :dn_single_pad * i] = True
                else:
                    attn_mask[dn_single_pad * i:dn_single_pad * (i + 1), dn_single_pad * (i + 1):dn_pad_size] = True
                    attn_mask[dn_single_pad * i:dn_single_pad * (i + 1), :dn_single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'pad_size': dn_pad_size
            }
        else:
            input_query_bbox = init_query_bbox.repeat(batch_size, 1, 1)
            input_query_feat = init_query_feat.repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return input_query_bbox, input_query_feat, attn_mask, mask_dict

    def prepare_for_dn_loss(self, mask_dict):
        cls_scores, bbox_preds = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        num_tgt = known_indice.numel()

        if len(cls_scores) > 0:
            cls_scores = cls_scores.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            bbox_preds = bbox_preds.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)

        return known_labels, known_bboxs, cls_scores, bbox_preds, num_tgt

    def dn_loss_single(self,
                       cls_scores,
                       bbox_preds,
                       known_bboxs,
                       known_labels,
                       num_total_pos=None):        
        # Compute the average number of gt boxes accross all gpus
        num_total_pos = cls_scores.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1.0).item()

        # cls loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        loss_cls = self.loss_cls(
            cls_scores,
            known_labels.long(),
            label_weights,
            avg_factor=num_total_pos
        )

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = self.dn_weight * torch.nan_to_num(loss_cls)
        loss_bbox = self.dn_weight * torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def calc_dn_loss(self, loss_dict, preds_dicts, num_dec_layers):
        known_labels, known_bboxs, cls_scores, bbox_preds, num_tgt = \
            self.prepare_for_dn_loss(preds_dicts['dn_mask_dict'])

        all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
        all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
        all_num_tgts_list = [num_tgt for _ in range(num_dec_layers)]

        dn_losses_cls, dn_losses_bbox = multi_apply(
            self.dn_loss_single, cls_scores, bbox_preds,
            all_known_bboxs_list, all_known_labels_list, all_num_tgts_list)

        loss_dict['loss_cls_dn'] = dn_losses_cls[-1]
        loss_dict['loss_bbox_dn'] = dn_losses_bbox[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1], dn_losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_dn'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_dn'] = loss_bbox_i
            num_dec_layer += 1

        return loss_dict

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        num_bboxes = bbox_pred.size(0)

        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore, self.code_weights, True)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        if 'dn_mask_dict' in preds_dicts and preds_dicts['dn_mask_dict'] is not None:
            loss_dict = self.calc_dn_loss(loss_dict, preds_dicts, num_dec_layers)

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            if VERSION.name == 'v0.17.1':
                import copy
                w, l = copy.deepcopy(bboxes[:, 3]), copy.deepcopy(bboxes[:, 4])
                bboxes[:, 3], bboxes[:, 4] = l, w
                bboxes[:, 6] = -bboxes[:, 6] - math.pi / 2

            bboxes = LiDARInstance3DBoxes(bboxes, 9)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
