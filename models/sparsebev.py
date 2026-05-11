import queue
import torch
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .utils import GridMask, pad_multiple, GpuPhotoMetricDistortion
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS


@DETECTORS.register_module()
class SparseBEV(MVXTwoStageDetector):
    def __init__(self,
                 data_aug=None,
                 stop_prev_grad=0,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=16,
                 aux_2d_loss=True,
                 depth_branch=None,
                 pretrained=None):
        super(SparseBEV, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = True

        self.memory = {}
        #self.queue = queue.Queue()

        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.prev_scene_token = None
        self.test_flag = False
        self.aux_2d_loss = aux_2d_loss
        self.stride = stride

        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_feat(self, img, img_metas, return_depth=False):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        # move some augmentations to GPU
        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'], training=self.training)

        input_shape = img.shape[-2:]

        if self.training and self.stop_prev_grad > 0:
            assert False
            H, W = input_shape
            img = img.reshape(B, -1, 6, C, H, W)

            img_grad = img[:, -self.num_frame_backbone_grads:]  
            img_nograd = img[:, :-self.num_frame_backbone_grads]  

            all_img_feats = [self.extract_img_feat(img_grad.reshape(-1, C, H, W))]

            with torch.no_grad():
                self.eval()
                for k in range(img_nograd.shape[1]):
                    all_img_feats.append(self.extract_img_feat(img_nograd[:, k].reshape(-1, C, H, W)))
                self.train()

            img_feats = []
            for lvl in range(len(all_img_feats[0])):
                C, H, W = all_img_feats[0][lvl].shape[1:]
                img_feat = torch.cat([feat[lvl].reshape(B, -1, 6, C, H, W) for feat in all_img_feats], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)
        else:
            img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped
    
    def obtain_history_memory(self,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            img_metas=None,
                            gt_bboxes_ignore=None,
                            **data):
        losses = dict()
        B, T, N = data['img'].shape[:3]
        
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                if key != 'img_feats':
                    data_t[key] = data[key][:, i] 

            # img_feats[lvl] : [B, T*N, C, H, W]
            mlvl_feats = []
            for lvl in range(len(data['img_feats'])):
                img_feat_t = data['img_feats'][lvl][:, i*N: i*N + N, ...]  # [B, N, C, H, W]
                mlvl_feats.append(img_feat_t)

            for b in range(B):
                img_metas[i][b]['gt_bboxes_3d'] = gt_bboxes_3d[i][b]
                img_metas[i][b]['gt_labels_3d'] = gt_labels_3d[i][b]
           
            data_t['img_feats'] = mlvl_feats
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True

            loss = self.forward_pts_train(gt_bboxes_3d[i], gt_labels_3d[i], img_metas[i], gt_bboxes_ignore,
                                        requires_grad=requires_grad, return_losses=return_losses,
                                        **data_t)
            if loss is not None:
                if T > 1:  # sliding window training
                    for key, value in loss.items():
                        losses['frame_'+str(i)+"_"+key] = value
                else:
                    losses = loss
        return losses

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(img_metas, **data)
            self.train()
        else:
            outs = self.pts_bbox_head(img_metas, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)

            return losses
        else:
            return None

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'img_metas']:
                kwargs[key] = list(zip(*kwargs[key]))
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img=None,
                      gt_bboxes_ignore=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: # for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
            self.prev_scene_token = None  # NOTE important when use different bs and custom_gpu_test in interval eval

        B, queue_len, N, C, H, W = img.size()
        data['img'] = img
        img = img.reshape(B, queue_len*N, C, H, W)
        img_feats = self.extract_feat(img, img_metas)
        data['img_feats'] = img_feats
        
        losses = self.obtain_history_memory(gt_bboxes_3d, gt_labels_3d,
                                            img_metas, gt_bboxes_ignore, **data)

        return losses

    def forward_test(self, img_metas, img=None, rescale=False, **data):
        if not self.test_flag: # for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = True
        
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
    
        for key in data.keys():
            data[key] = data[key][0][0].unsqueeze(0)
            
        data['img'] = img[0]
        return self.simple_test(img_metas[0], rescale, **data)

    def simple_test_pts(self, img_metas, rescale=False, **data):
        outs = self.pts_bbox_head(img_metas, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas[0], rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
    
    def simple_test(self, img_metas, rescale=False, **data):
        img_feats = self.extract_feat(img=data['img'], img_metas=img_metas)
        data['img_feats'] = img_feats

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_metas, rescale=rescale, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            
            if pts_bbox['boxes_3d'].box_dim < 9:  # waymo bbox
                result_dict['sample_idx'] = img_metas[0]['sample_idx']
                result_dict['context_name'] = img_metas[0]['context_name']
                #result_dict['timestamp'] = str(img_metas[0]['timestamp'].data.item()*1e6)
                result_dict['timestamp'] = format(img_metas[0]['timestamp'].data.item()*1e6, ".0f")

        return bbox_list
