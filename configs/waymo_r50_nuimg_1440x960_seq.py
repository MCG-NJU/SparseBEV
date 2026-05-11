dataset_type = 'CustomWaymoDataset'
dataset_root = 'data/waymo/kitti_format/'
#dataset_root = 'data/Waymo_mini/kitti_format/'
file_client_args = dict(backend="disk")

# Waymo input modality
input_modality = dict(use_lidar=False, use_camera=True)

# For Waymo 3-class
# class_names = ['Pedestrian', 'Cyclist', 'Car']
class_names = ['Car', 'Pedestrian', 'Cyclist']

batch_size = 16
#num_iters_per_epoch = 31617 // (batch_size)  # 1/5 of training set
num_iters_per_epoch = 31499 // (batch_size)  # 1/5 of training set
num_epochs = 60
checkpoint_epoch_interval = 10

queue_length = 1
num_frame_losses = 1

collect_keys=['lidar2img', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
# Waymo point_cloud_range
point_cloud_range = [-35.0, -75.0, -2, 75.0, 75.0, 4]

# voxel_size = [0.2, 0.2, 8]
# [0.5, 0.5, 0.5]
voxel_size = [0.5, 0.5, 6]

# arch config
embed_dims = 256
num_layers = 6
num_query = 400
num_frames = 1
num_levels = 4
num_points = 4

img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=True)
img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)

model = dict(
    type='SparseBEV',
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    aux_2d_loss=False,
    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_bbox_head=dict(
        type='SparseBEVHead',     
        # num_classes=10,
        num_classes=3,
        in_channels=embed_dims,
        num_query=num_query,
        num_views=5,
        query_denoising=True,
        query_denoising_groups=10,
        code_size=8,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        sync_cls_avg_factor=True,
        memory_config=dict(
            num_history=500,
            max_time_interval=2,
            memory_len=1024,
            len_per_frame=256,
            interval=1
        ),
        transformer=dict(
            type='SparseBEVTransformer',
            return_intermediate=True,
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_views=5,
            num_classes=3,
            code_size=8,
            pc_range=point_cloud_range),
        bbox_coder=dict(
            type='NMSFreeCoder',
            # TODO
            # nuscene got point_cloud_range = [-50, -50, -5, 50, 50, 3]
            # but waymo is point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
            # transfusion post_center_range=[-80, -80, -10.0, 80, 80, 10.0]
            # post_center_range=[-40, -80, -10.0, 80, 80, 10.0],
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            score_threshold=0.05,
            num_classes=3),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=embed_dims // 2,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
        )
    ))
)

ida_aug_conf = {
    'resize_lim': (0.62, 0.68),
    'final_dim': (832, 1248),
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
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'waymo_infos_train.pkl',
        split='training',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        num_frame_losses=num_frame_losses,
        seq_split_num=2, # streaming video training
        seq_mode=True, # streaming video training
        test_mode=False,
        load_interval=5,  # 1/5 of training set
        box_type_3d='LiDAR',
        load_mode='lidar_frame',
        cam_sync=True),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'waymo_infos_val.pkl',
        split='validation',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'img_metas'],
        queue_length=queue_length,
        test_mode=True,
        box_type_3d='LiDAR',
        load_mode='lidar_frame',
        cam_sync=True),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'waymo_infos_val_10f_5intvl.pkl',
        split='training',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        load_mode='lidar_frame',
        cam_sync=True),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.001
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2)
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    by_epoch=False
)
total_epochs = 60

# load pretrained weights
load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]

# resume the last training
resume_from = None

# checkpointing
checkpoint_config = dict(interval=num_iters_per_epoch*num_epochs, max_keep_ckpts=1, by_epoch=False)

# logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='MyTextLoggerHook', by_epoch=False, interval=51),
        #dict(type='MyTensorboardLoggerHook', by_epoch=False, interval=500, reset_flag=True)
        dict(type='MyWandbLoggerHook', by_epoch=False, interval=500, reset_flag=True, commit=True, project_name='SparseBev', team_name='liuhs-team')
    ]
)

# evaluation
eval_config = dict(interval=num_iters_per_epoch*num_epochs, by_epoch=False)  # TODO code for interval eval 

runner = dict(type='IterBasedRunner')

# other flags
debug = False
# no_validate = False
no_validate = True
