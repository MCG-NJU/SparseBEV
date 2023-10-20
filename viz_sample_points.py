import utils
import logging
import argparse
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from models.utils import DUMP, VERSION


def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument('--score_threshold', default=0.3)
    parser.add_argument('--stage_id', default=5)
    parser.add_argument('--num_frames', default=3)
    parser.add_argument('--num_views', default=6)
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    # use val-mini for visualization
    cfgs.data.val.ann_file = cfgs.data.val.ann_file.replace('val', 'val_mini')

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    # you need one GPU
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1

    utils.init_logging(None, cfgs.debug)

    logging.info('Using GPU: %s' % torch.cuda.get_device_name(0))
    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()
    model = MMDataParallel(model, [0])

    logging.info('Loading checkpoint from %s' % args.weights)
    checkpoint = load_checkpoint(
        model, args.weights, map_location='cuda', strict=True,
        logger=logging.Logger(__name__, logging.ERROR)
    )

    if 'version' in checkpoint:
        VERSION.name = checkpoint['version']

    for idx, data in enumerate(val_loader):
        DUMP.enabled = True
        model.eval()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        cls_scores = torch.load('{}/cls_score_stage{}.pth'.format(DUMP.out_dir, args.stage_id))[0]
        cls_scores, cls_ids = torch.max(cls_scores, dim=-1)

        # only select queries with high confidence
        query_ids = torch.where(cls_scores > args.score_threshold)[0]
        cls_scores, cls_ids = cls_scores[query_ids], cls_ids[query_ids]

        plt.figure(figsize=(240, 49))
        view_mapping = [1, 2, 0, 4, 5, 3]

        for frame_id in range(args.num_frames):
            sample_points_cam = torch.load(
                '{}/sample_points_cam_stage{}.pth'.format(DUMP.out_dir, args.stage_id)
            )  # [1, 8f, 6view, 900, 32, 2]
            valid_mask = torch.load(
                '{}/sample_points_cam_valid_mask_stage{}.pth'.format(DUMP.out_dir, args.stage_id)
            )  # [1, 8f, 6view, 900, 32]

            for view_id in range(args.num_views):
                filenames = data['img_metas'][0].data[0][0]['filename']
                filename = filenames[frame_id * 6 + view_id]

                # crop 1600x640 area
                img = Image.open(filename)
                img = img.crop((0, 260, 1600, 900))

                # plot image
                plot_id = frame_id * args.num_views + view_mapping[view_id] + 1
                ax = plt.subplot(args.num_frames, args.num_views, plot_id)
                ax.imshow(img)
                ax.axis('off')
                ax.set_xlim(0, 1600)
                ax.set_ylim(640, 0)

                # plot the sampling points for each query
                for query_id in query_ids:
                    xyz = sample_points_cam[0, frame_id, view_id, query_id].numpy()  # [32, 3]
                    mask = valid_mask[0, frame_id, view_id, query_id].numpy()  # [32]
                    mask = np.round(mask).astype(bool)

                    cx = xyz[:, 0] * 1600
                    cy = xyz[:, 1] * 640
                    cz = xyz[:, 2]

                    cz[np.where(cz <= 0)] = 1e8
                    cz = np.log(60 / cz ** 0.8) * 2.4
                    cx, cy, cz = cx[mask], cy[mask], cz[mask]

                    if len(cz) == 0:
                        continue

                    ax.scatter(cx, cy, s=4**(cz + 1), alpha=0.7, color='C%d' % (query_id % 5))

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.01, wspace=0.01)
        plt.savefig('outputs/sp_%04d.jpg' % idx, dpi=20)
        plt.close()

        logging.info('Visualized result is dumped to outputs/sp_%04d.jpg' % idx)


if __name__ == '__main__':
    main()
