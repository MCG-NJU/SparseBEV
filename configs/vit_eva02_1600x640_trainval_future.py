_base_ = ['./r50_nuimg_704x256.py']

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_backbone = dict(
    _delete_=True,
    type='EVA02',
    img_size=1536,
    real_img_size=(640, 1600),
    patch_size=16,
    in_chans=3,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4*2/3,
    qkv_bias=True,
    drop_path_rate=0.3,
    use_abs_pos=True,
    window_size=16,
    window_block_indexes=(
        list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
    ),
    residual_block_indexes=(),
    use_act_checkpoint=True,
    # args for simple FPN
    fpn_out_channels=256,
    fpn_scale_factors=(4.0, 2.0, 1.0, 0.5),
    fpn_top_block=True,
    fpn_norm="LN",
    fpn_square_pad=1600,
    pretrained='pretrain/eva02_L_coco_seg_sys_o365.pth',
    frozen_blocks=3,
)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True
)

model = dict(
    img_backbone=img_backbone,
    img_neck=None,
    stop_prev_grad=4,
    pts_bbox_head=dict(
        num_query=1600,
        transformer=dict(
            num_levels=5,
            num_points=8,
            num_frames=15))
)

ida_aug_conf = {
    'resize_lim': (0.94, 1.25),
    'final_dim': (640, 1600),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweepsFutureInterleave', prev_sweeps_num=7, next_sweeps_num=7),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'], meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweepsFutureInterleave', prev_sweeps_num=7, next_sweeps_num=7, test_mode=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img'], meta_keys=(
                'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                'lidar2img', 'img_timestamp'))
        ])
]

data = dict(
    train=dict(
        ann_file=['data/nuscenes/nuscenes_infos_train_sweep.pkl',
                  'data/nuscenes/nuscenes_infos_val_sweep.pkl'],
        pipeline=train_pipeline),
    val=dict(
        ann_file='data/nuscenes/nuscenes_infos_val_sweep.pkl',  # use nuscenes_infos_test_sweep.pkl for submission
        pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)

load_from = None
revise_keys = None
