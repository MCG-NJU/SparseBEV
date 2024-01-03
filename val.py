import os
import utils
import logging
import argparse
import importlib
import torch
import torch.distributed
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed, multi_gpu_test, single_gpu_test
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from models.utils import VERSION


def evaluate(dataset, results, epoch):
    metrics = dataset.evaluate(results, jsonfile_prefix='submission')

    mAP = metrics['pts_bbox_NuScenes/mAP']
    mATE = metrics['pts_bbox_NuScenes/mATE']
    mASE = metrics['pts_bbox_NuScenes/mASE']
    mAOE = metrics['pts_bbox_NuScenes/mAOE']
    mAVE = metrics['pts_bbox_NuScenes/mAVE']
    mAAE = metrics['pts_bbox_NuScenes/mAAE']
    NDS = metrics['pts_bbox_NuScenes/NDS']

    logging.info('--- Evaluation Results (Epoch %d) ---' % epoch)
    logging.info('mAP: %.4f' % metrics['pts_bbox_NuScenes/mAP'])
    logging.info('mATE: %.4f' % metrics['pts_bbox_NuScenes/mATE'])
    logging.info('mASE: %.4f' % metrics['pts_bbox_NuScenes/mASE'])
    logging.info('mAOE: %.4f' % metrics['pts_bbox_NuScenes/mAOE'])
    logging.info('mAVE: %.4f' % metrics['pts_bbox_NuScenes/mAVE'])
    logging.info('mAAE: %.4f' % metrics['pts_bbox_NuScenes/mAAE'])
    logging.info('NDS: %.4f' % metrics['pts_bbox_NuScenes/NDS'])

    return {
        'mAP': mAP,
        'mATE': mATE,
        'mASE': mASE,
        'mAOE': mAOE,
        'mAVE': mAVE,
        'mAAE': mAAE,
        'NDS': NDS,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    # you need GPUs
    assert torch.cuda.is_available()

    # determine local_rank and world_size
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if local_rank == 0:
        utils.init_logging(None, cfgs.debug)
    else:
        logging.root.disabled = True

    logging.info('Using GPU: %s' % torch.cuda.get_device_name(local_rank))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        logging.info('Initializing DDP with %d GPUs...' % world_size)
        dist.init_process_group('nccl', init_method='env://')

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=False,
        seed=0,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()
    model.fp16_enabled = True

    if world_size > 1:
        model = MMDistributedDataParallel(model, [local_rank], broadcast_buffers=False)
    else:
        model = MMDataParallel(model, [0])

    logging.info('Loading checkpoint from %s' % args.weights)
    checkpoint = load_checkpoint(
        model, args.weights, map_location='cuda', strict=True,
        logger=logging.Logger(__name__, logging.ERROR)
    )

    if 'version' in checkpoint:
        VERSION.name = checkpoint['version']

    if world_size > 1:
        results = multi_gpu_test(model, val_loader, gpu_collect=True)
    else:
        results = single_gpu_test(model, val_loader)

    if local_rank == 0:
        evaluate(val_dataset, results, -1)


if __name__ == '__main__':
    main()
