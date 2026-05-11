_base_ = ['./waymo_r50_nuimg_1440x960_seq.py']
file_client_args = dict(backend="disk")
# For nuScenes we usually do 10-class detection
class_names = ['Car', 'Pedestrian', 'Cyclist']

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-35.0, -75.0, -2, 75.0, 75.0, 4]

# voxel_size = [0.2, 0.2, 8]
# [0.5, 0.5, 0.5]
voxel_size = [0.5, 0.5, 6]

collect_keys=['lidar2img', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']

batch_size = 16
num_iters_per_epoch = 31499 // (batch_size)  # 1/5 of training set
num_epochs = 48
checkpoint_epoch_interval = 8

img_backbone = dict(
    type='ResNet',
    depth=101,
    with_cp=True,
)

img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5,
)

model = dict(
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_bbox_head=dict(transformer=dict(num_levels=5)),
)

ida_aug_conf = {
    'resize_lim': (0.7, 0.8),
    'final_dim': (960, 1440),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 1280, 'W': 1920,
    'rand_flip': True,
}

train_pipeline = [
    dict(type='LoadMultiViewDifferentShapeImage', to_float32=False, color_type='color'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'prev_exists'] + collect_keys, meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 'timestamp', 'ego_pose', 'ego_pose_inv'))
]

test_pipeline = [
    dict(type='LoadMultiViewDifferentShapeImage', to_float32=False, color_type='color'),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1280),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys, with_label=False),
            dict(type='Collect3D', keys=['img']+  collect_keys, meta_keys=(
                'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                'lidar2img', 'img_timestamp', 'timestamp', 'ego_pose', 'ego_pose_inv',
                'sample_idx', 'context_name'))
        ])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)

optimizer = dict(
    type='AdamW',
    lr=3e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.001
)

total_epochs = 48

# load pretrained weights
load_from = 'pretrain/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth'
revise_keys = [('backbone', 'img_backbone')]

# checkpointing
checkpoint_config = dict(interval=num_iters_per_epoch*checkpoint_epoch_interval, max_keep_ckpts=1, by_epoch=False)
# evaluation
eval_config = dict(interval=num_iters_per_epoch*num_epochs, by_epoch=False)  # TODO code for interval eval 