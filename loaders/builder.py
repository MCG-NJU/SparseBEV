from functools import partial
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader
from mmdet.datasets.samplers import DistributedGroupSampler, DistributedSampler, GroupSampler
from loaders.samplers.group_sampler import InfiniteGroupEachSampleInBatchSampler
from loaders.samplers.sampler import build_sampler
from mmdet.datasets.builder import worker_init_fn

def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     shuffler_sampler=None,
                     nonshuffler_sampler=None,
                     runner_type=dict(type='EpochBasedRunner'),
                     **kwargs):

    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = build_sampler(shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                                     dict(
                                         dataset=dataset,
                                         samples_per_gpu=samples_per_gpu,
                                         num_replicas=world_size,
                                         rank=rank,
                                         seed=seed)
                                     )
        else:
            sampler = build_sampler(nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                                     dict(
                                         dataset=dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=shuffle,
                                         seed=seed)
                                     )
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    batch_sampler = None

    if runner_type['type'] == 'IterBasedRunner' and shuffler_sampler['type'] =='InfiniteGroupEachSampleInBatchSampler':
        # NOTE: original has more options 
        # https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/datasets/builder.py#L145-L157
        batch_sampler = InfiniteGroupEachSampleInBatchSampler(
            dataset,
            samples_per_gpu,
            world_size,
            rank,
            seed=seed)
        batch_size = 1
        sampler = None

    # pass in init_fn to dataloader to force the worker_init_fn to know the rank and distinguish workers in different rank
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader
