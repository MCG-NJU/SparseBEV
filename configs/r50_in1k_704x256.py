_base_ = ['./r50_nuimg_704x256.py']

img_backbone = dict(pretrained='torchvision://resnet50')

model = dict(
    img_backbone=img_backbone,
    pts_bbox_head=dict(num_query=900)
)

optimizer = dict(
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.4),
        'sampling_offset': dict(lr_mult=0.1),
    })
)

load_from = None
revise_keys = None

total_epochs = 36
eval_config = dict(interval=total_epochs)
