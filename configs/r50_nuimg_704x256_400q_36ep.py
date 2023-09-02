_base_ = ['./r50_nuimg_704x256.py']

model = dict(
    pts_bbox_head=dict(num_query=400)
)

total_epochs = 36
eval_config = dict(interval=total_epochs)

data = dict(workers_per_gpu=12)
