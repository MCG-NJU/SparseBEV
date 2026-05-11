# Copyright (c) OpenMMLab. All rights reserved.
# Copy from https://github.com/Tai-Wang/Depth-from-Motion
import copy
import math
import os
import random
import tempfile
from os import path as osp
from typing import List, Optional, Sequence, Tuple, Union
import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes,
                         LiDARInstance3DBoxes, points_cam2img)
from mmdet.datasets import DATASETS

from loaders.nuscenes_dataset import invert_matrix_egopose_numpy
from .kitti_dataset import KittiDataset
from mmcv.parallel import DataContainer as DC

# @DATASETS.register_module()
@DATASETS.register_module(force=True)
class CustomWaymoDataset(KittiDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list(float), optional): The range of point cloud used
            to filter invalid predicted boxes.
            Default: [-85, -85, -5, 85, 85, 5].
        load_mode: Loading mode for different settings. Supported choices
            include: 'lidar_frame', 'cam_frame', 'cam_mono'. 'lidar_frame'
            supports loading frame-based data for 3D detection based on LiDAR
            coordinate system; 'cam_mono' means only loading image-based
            front-view data; 'cam_frame' means loading multi-view images based
            on perspective views. Defaults to 'lidarpcd_limit_range_frame'.
    """

    CLASSES = ('Car', 'Cyclist', 'Pedestrian')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 data_prefix: dict = dict(
                     pts='velodyne',
                     CAM_FRONT='image_0',
                     CAM_FRONT_LEFT='image_1',
                     CAM_FRONT_RIGHT='image_2',
                     CAM_SIDE_LEFT='image_3',
                     CAM_SIDE_RIGHT='image_4'),
                 pipeline=None,
                 classes=None,
                 modality=None, 
                 seq_mode=False, 
                 seq_split_num=1, 
                 num_frame_losses=1, 
                 queue_length=1,
                 collect_keys=None, 
                 random_length=0,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 load_mode='lidar_frame',
                 cam_sync=False,
                 multiview_indices=[
                     'image_0', 'image_1', 'image_2', 'image_3', 'image_4'
                 ],
                 max_sweeps=0,
                 file_client_args=dict(backend='disk')):
        self.load_interval = load_interval
        # set loading mode for different task settings
        self.load_mode = load_mode
        assert load_mode in ['lidar_frame', 'cam_frame', 'cam_mono']
        self.cam_sync = cam_sync
        # construct self.cat_ids for vision-only anns parsing
        self.cat_ids = range(len(self.CLASSES))
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.bbox_code_size = 7
        self.multiview_indices = multiview_indices
        self.default_view_index = 'image_0'
        self.max_sweeps = max_sweeps
        self.file_client_args = file_client_args
        # we do not provide file_client_args to custom_3d init
        # because we want disk loading for info
        # while ceph loading for KITTI2Waymo
        # TODO: support this case with different file_client_args
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range)
        
        self.data_prefix = data_prefix
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode

        self.results = []
        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        prev_seg = '000'
        for idx in range(len(self.data_infos)):
            # NOTE in '{a}{bbb}{ccc}.bin', bbb means idx of segments
            if idx != 0:
                curr_seg = self.data_infos[idx]['lidar_points']['lidar_path'].split('.')[0][1:4]
                if curr_seg != prev_seg:
                    curr_sequence += 1
                    prev_seg = curr_seg
            res.append(curr_sequence)
        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)
        prev_scene_token = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            queue.append(example)

        for k in range(self.num_frame_losses):
            if self.filter_empty_gt and \
                (queue[-k-1] is None or ~(queue[-k-1]['gt_labels_3d']._data != -1).any()):
                return None
        return self.union2one(queue)

    def union2one(self, queue):
        for key in self.collect_keys:
            if key != 'img_metas':
                queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
            else:
                queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
        if not self.test_mode:
            for key in ['gt_bboxes_3d', 'gt_labels_3d']:
                if key == 'gt_bboxes_3d':
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                else:
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)

        queue = queue[-1]
        return queue

    def load_annotations(self, ann_file):
        # re-write load_annotations for different tasks
        # sometimes need to re-organize self.data_infos
        # e.g.: frame-based -> cam-based
        
        annotations = mmcv.load(ann_file, file_format='pkl')
        
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']
        raw_data_list = raw_data_list[::self.load_interval]
        if self.load_interval > 1:
            print_log(
                f'Sample size will be reduced to 1/{self.load_interval} of'
                ' the original data sample',
                logger='current')

        if self.cam_sync and ('instances' in raw_data_list[0]):
            for raw_data_info in raw_data_list:
                #raw_data_info['annos'] = raw_data_info['cam_sync_annos']
                raw_data_info['instances'] = raw_data_info['cam_sync_instances']

        # avoid empty_gt and random_another loop
        remove_list = ['453', '416', '589']
        for raw_data_info in raw_data_list[:]:
            if raw_data_info['lidar_points']['lidar_path'].split('.')[0][1:4] in remove_list:
                raw_data_list.remove(raw_data_info)

        if self.load_mode == 'lidar_frame':
            self.raw_data_infos = raw_data_list
            return raw_data_list
        else:
            raise NotImplementedError
    
    def _get_pts_filename(self, idx):
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx:07d}.bin')
        return pts_filename

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]

        # NOTE modified
        assert self.load_mode == 'lidar_frame'
        sample_idx = info['sample_idx']
        pts_filename = self._get_pts_filename(sample_idx)
        # scene_idx = str(info['lidar_points']['lidar_path'].split('.')[0][1:4])
        # print(scene_idx)
        ego_pose = np.array(info['ego2global'])
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        input_dict = dict(
            sample_idx=sample_idx,
            context_name=info['context_name'],
            #pts_filename=info['lidar_points']['lidar_path'],
            pts_filename=pts_filename,
            timestamp=info['timestamp'] / 1e6,
            ego_pose=ego_pose,
            ego_pose_inv=ego_pose_inv
        )

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            cam_intrinsic = []

            for (cam_key, img_info) in info['images'].items():
                
                # TODO remove if
                if 'img_path' in img_info:
                    cam_prefix = self.data_prefix.get(cam_key, '')
                    img_path = osp.join(self.root_split,
                        cam_prefix, img_info['img_path'])
                if 'lidar2cam' in img_info:
                    lidar2cam = np.array(img_info['lidar2cam'])
                if 'cam2img' in img_info:
                    cam2img = np.array(img_info['cam2img'])
                if 'lidar2img' in img_info:
                    lidar2img = np.array(img_info['lidar2img'])
                else:
                    lidar2img = cam2img @ lidar2cam

                lidar2img_pad = np.eye(4)
                lidar2img_pad[:lidar2img.shape[0], :lidar2img.shape[1]] = lidar2img

                img_paths.append(img_path)
                img_timestamps.append(info['timestamp'] / 1e6)
                lidar2img_rts.append(lidar2img_pad)
                cam_intrinsic.append(cam2img)
            
            if not self.test_mode: # for seq_mode
                prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                prev_exists=prev_exists,
                cam_intrinsic=cam_intrinsic,
            ))

        else:
            raise NotImplementedError
        
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        # TODO
        # if self.test_mode and self.load_eval_anns:
        #     info['eval_ann_info'] = self.parse_ann_info(info)

        return input_dict
    
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                    0, 1, 2 represent xxxxx respectively.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        
        # TODO
        # ['bbox_3d', 'bbox_label_3d'] -> ['gt_bboxes_3d', 'gt_labels_3d']
        ann_info = super().parse_ann_info(info)

        if ann_info is None:
            # empty instance
            ann_info = {}
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
        
        # we need other objects to avoid collision when sample
        ann_info = self._remove_dontcare(ann_info)

        if self.load_mode == 'lidar_frame':
            gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        else:
            gt_bboxes_3d = CameraInstance3DBoxes(ann_info['gt_bboxes_3d'])

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=ann_info['gt_labels_3d']
            )
        
        return anns_results
        
    def _remove_dontcare(self, ann_info: dict) -> dict:
        """Remove annotations that do not need to be cared.

        -1 indicates dontcare in MMDet3d.

        Args:
            ann_info (dict): Dict of annotation infos. The
                instance with label `-1` will be removed.

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        filter_mask = ann_info['gt_labels_3d'] > -1
        for key in ann_info.keys():
            if key != 'instances':
                img_filtered_annotations[key] = (ann_info[key][filter_mask])
            else:
                img_filtered_annotations[key] = ann_info[key]
        return img_filtered_annotations
    
    def _parse_cam_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                # Note that the categories are defined as
                # ('Pedestrian', 'Cyclist', 'Car') in the converter
                # TODO: unify the settings
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                # 3D annotations in camera coordinates
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(-1, )
                gt_bboxes_cam3d.append(bbox_cam3d)
                # 2.5D annotations in camera coordinates
                center2d = ann['center2d'][:2]
                depth = ann['center2d'][2]
                centers2d.append(center2d)
                depths.append(depth)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size),
                                       dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)

        gt_bboxes_cam3d = CameraInstance3DBoxes(
            gt_bboxes_cam3d,
            box_dim=gt_bboxes_cam3d.shape[-1],
            origin=(0.5, 0.5, 0.5))
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_bboxes_3d=gt_bboxes_cam3d,
            gt_labels_3d=gt_labels_3d,
            centers2d=centers2d,
            depths=depths,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def format_results(
        self,
        #results: List[dict],
        result_prefix: Optional[str] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the results to bin file.

        Args:
            results (List[dict]): Testing results of the dataset.
            result_prefix (str, optional): The prefix of result file. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
        """
        waymo_results_final_path = f'{result_prefix}.bin'

        from core.evaluation.waymo_utils.prediction_to_waymo import \
            Prediction2Waymo
        # NOTE add classes
        classes = ['Car', 'Pedestrian', 'Cyclist']
        converter = Prediction2Waymo(self.results, waymo_results_final_path,
                                     classes)
        converter.convert()

    def evaluate(self,
                 results,
                 metric='waymo',
                 logger=None,
                 pklfile_prefix=None,
                 result_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            result_prefix (str, optional): The prefix of result '*.bin' file,
                including the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        assert ('waymo' in metric), \
            f'invalid metric {metric}'
        if 'waymo' in metric:
            
            waymo_root = osp.join(
                self.data_root.split('kitti_format')[0], 'waymo_format')
            if pklfile_prefix is None:
                eval_tmp_dir = tempfile.TemporaryDirectory()
                pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
            else:
                eval_tmp_dir = None
            # cy modified
            self.process(results)
            self.format_results(result_prefix=result_prefix)
            # result_files, tmp_dir = self.format_results(
            #     results,
            #     pklfile_prefix,
            #     submission_prefix,
            #     data_format='waymo')
            import subprocess
            
            eval_script = 'core/evaluation/waymo_utils/' + \
                f'compute_detection_metrics_main {result_prefix}.bin '
            # parse the text to get ap_dict
            ap_dict = {
                'Vehicle/L1 mAP': 0,
                'Vehicle/L1 mAPH': 0,
                'Vehicle/L2 mAP': 0,
                'Vehicle/L2 mAPH': 0,
                'Pedestrian/L1 mAP': 0,
                'Pedestrian/L1 mAPH': 0,
                'Pedestrian/L2 mAP': 0,
                'Pedestrian/L2 mAPH': 0,
                'Sign/L1 mAP': 0,
                'Sign/L1 mAPH': 0,
                'Sign/L2 mAP': 0,
                'Sign/L2 mAPH': 0,
                'Cyclist/L1 mAP': 0,
                'Cyclist/L1 mAPH': 0,
                'Cyclist/L2 mAP': 0,
                'Cyclist/L2 mAPH': 0,
                'Overall/L1 mAP': 0,
                'Overall/L1 mAPH': 0,
                'Overall/L2 mAP': 0,
                'Overall/L2 mAPH': 0
            }
            if self.load_mode == 'lidar_frame':
                if self.modality['use_lidar']:
                    eval_script += f'{waymo_root}/gt.bin'
                else:
                    eval_script += f'{waymo_root}/cam_gt.bin'
            elif self.load_mode == 'cam_mono':
                eval_script += f'{waymo_root}/fov_gt.bin'
            elif self.load_mode == 'cam_frame':
                eval_script += f'{waymo_root}/cam_gt.bin'
            if self.cam_sync:  # use let metric when using cam_sync
                eval_script = eval_script.replace(
                    'compute_detection_metrics_main',
                    'compute_detection_let_metrics_main')
                ap_dict = {
                    'Vehicle mAPL': 0,
                    'Vehicle mAP': 0,
                    'Vehicle mAPH': 0,
                    'Pedestrian mAPL': 0,
                    'Pedestrian mAP': 0,
                    'Pedestrian mAPH': 0,
                    'Sign mAPL': 0,
                    'Sign mAP': 0,
                    'Sign mAPH': 0,
                    'Cyclist mAPL': 0,
                    'Cyclist mAP': 0,
                    'Cyclist mAPH': 0,
                    'Overall mAPL': 0,
                    'Overall mAP': 0,
                    'Overall mAPH': 0
                }
            ret_bytes = subprocess.check_output(eval_script, shell=True)
            ret_texts = ret_bytes.decode('utf-8')
            print_log(ret_texts)
            if not self.cam_sync:
                mAP_splits = ret_texts.split('mAP ')
                mAPH_splits = ret_texts.split('mAPH ')
                for idx, key in enumerate(ap_dict.keys()):
                    split_idx = int(idx / 2) + 1
                    if idx % 2 == 0:  # mAP
                        ap_dict[key] = float(
                            mAP_splits[split_idx].split(']')[0])
                    else:  # mAPH
                        ap_dict[key] = float(
                            mAPH_splits[split_idx].split(']')[0])
                ap_dict['Overall/L1 mAP'] = \
                    (ap_dict['Vehicle/L1 mAP'] +
                     ap_dict['Pedestrian/L1 mAP'] +
                     ap_dict['Cyclist/L1 mAP']) / 3
                ap_dict['Overall/L1 mAPH'] = \
                    (ap_dict['Vehicle/L1 mAPH'] +
                     ap_dict['Pedestrian/L1 mAPH'] +
                     ap_dict['Cyclist/L1 mAPH']) / 3
                ap_dict['Overall/L2 mAP'] = \
                    (ap_dict['Vehicle/L2 mAP'] +
                     ap_dict['Pedestrian/L2 mAP'] +
                     ap_dict['Cyclist/L2 mAP']) / 3
                ap_dict['Overall/L2 mAPH'] = \
                    (ap_dict['Vehicle/L2 mAPH'] +
                     ap_dict['Pedestrian/L2 mAPH'] +
                     ap_dict['Cyclist/L2 mAPH']) / 3
            else:
                mAPL_splits = ret_texts.split('mAPL ')
                mAP_splits = ret_texts.split('mAP ')
                mAPH_splits = ret_texts.split('mAPH ')
                for idx, key in enumerate(ap_dict.keys()):
                    split_idx = int(idx / 3) + 1
                    if idx % 3 == 0:  # mAPL
                        ap_dict[key] = float(
                            mAPL_splits[split_idx].split(']')[0])
                    elif idx % 3 == 1:  # mAP
                        ap_dict[key] = float(
                            mAP_splits[split_idx].split(']')[0])
                    else:  # mAPH
                        ap_dict[key] = float(
                            mAPH_splits[split_idx].split(']')[0])
                ap_dict['Overall mAPL'] = \
                    (ap_dict['Vehicle mAPL'] + ap_dict['Pedestrian mAPL'] +
                     ap_dict['Cyclist mAPL']) / 3
                ap_dict['Overall mAP'] = \
                    (ap_dict['Vehicle mAP'] + ap_dict['Pedestrian mAP'] +
                     ap_dict['Cyclist mAP']) / 3
                ap_dict['Overall mAPH'] = \
                    (ap_dict['Vehicle mAPH'] + ap_dict['Pedestrian mAPH'] +
                     ap_dict['Cyclist mAPH']) / 3
            if eval_tmp_dir is not None:
                eval_tmp_dir.cleanup()

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict

    def process(self, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            bboxes_3d = data_sample['pts_bbox']['boxes_3d']
            bboxes_3d.limit_yaw(offset=0.5, period=np.pi * 2)
            scores_3d = data_sample['pts_bbox']['scores_3d']
            labels_3d = data_sample['pts_bbox']['labels_3d']
            # TODO: check lidar post-processing
            if isinstance(bboxes_3d, CameraInstance3DBoxes):
                pass
            result['bboxes_3d'] = bboxes_3d.tensor.cpu().numpy()
            result['scores_3d'] = scores_3d.cpu().numpy()
            result['labels_3d'] = labels_3d.cpu().numpy()
            result['sample_idx'] = data_sample['sample_idx']
            result['context_name'] = data_sample['context_name']
            result['timestamp'] = data_sample['timestamp']
            self.results.append(result)