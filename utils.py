import os
import pdb
import pickle
import socket
import wandb
import sys
import glob
import torch
import shutil
import logging
import datetime
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.hooks.logger import LoggerHook, TextLoggerHook
from mmcv.runner.dist_utils import master_only
#from torch.utils.tensorboard import SummaryWriter
import mmcv
import os.path as osp
import tempfile
import time
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results
import bisect
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.core.evaluation.eval_hooks import _calc_dynamic_intervals

from mmcv.runner import load_state_dict
import torch.nn as nn
import math
import os
from copy import deepcopy

from models.utils import rotation_3d_in_axis

def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)

class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def set_trace():
    ForkedPdb().set_trace(sys._getframe().f_back)

def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def backup_code(work_dir, verbose=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for pattern in ['*.py', 'configs/*.py', 'models/*.py', 'loaders/*.py', 'loaders/pipelines/*.py']:
        for file in glob.glob(pattern):
            src = os.path.join(base_dir, file)
            dst = os.path.join(work_dir, 'backup', os.path.dirname(file))

            if verbose:
                logging.info('Copying %s -> %s' % (os.path.relpath(src), os.path.relpath(dst)))
            
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src, dst)


@HOOKS.register_module()
class MyTextLoggerHook(TextLoggerHook):
    def _log_info(self, log_dict, runner):
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)

        # by epoch: Epoch [4][100/1000]
        # by iter:  Iter [100/100000]
        if log_dict['mode'] == 'train':
            if self.by_epoch:
                log_str = f'Epoch [{log_dict["epoch"]}/{runner.max_epochs}]' \
                            f'[{log_dict["iter"]}/{len(runner.data_loader)}] '
            else:
                log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}] '

            log_str += 'loss: %.2f, ' % log_dict['loss']

        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: Epoch[val] [4][1000]
            # by iter: Iter[val] [1000]
            if self.by_epoch:
                log_str = f'Epoch({log_dict["mode"]}) ' \
                    f'[{log_dict["epoch"]}][{log_dict["iter"]}]\t'
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

            log_items = []
            for name, val in log_dict.items():
                # TODO support for nuscenes and waymo 
                # if name not in [
                #     'pts_bbox_NuScenes/NDS', 'pts_bbox_NuScenes/mAP',
                #     'pts_bbox_NuScenes/mATE', 'pts_bbox_NuScenes/mASE',
                #     'pts_bbox_NuScenes/mAOE', 'pts_bbox_NuScenes/mAVE', 
                #     'pts_bbox_NuScenes/mAAE'
                # ]:
                #     continue
                if isinstance(val, float):
                    val = f'{val:.4f}'
                log_items.append(f'{name}: {val}')
            log_str += ', '.join(log_items)

        #log_str += 'loss: %.2f, ' % log_dict['loss']

        if 'time' in log_dict.keys():
            # MOD: skip the first iteration since it's not accurate
            if runner.iter == self.start_iter:
                time_sec_avg = log_dict['time']
            else:
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter)

            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_str += f'eta: {eta_str}, '
            log_str += f'time: {log_dict["time"]:.2f}s, ' \
                        f'data: {log_dict["data_time"] * 1000:.0f}ms, '
            # statistic memory
            if torch.cuda.is_available():
                log_str += f'mem: {log_dict["memory"]}M'

        runner.logger.info(log_str)

    def log(self, runner):
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = {
            'mode': self.get_mode(runner),
            'epoch': self.get_epoch(runner),
            'iter': cur_iter
        }

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        if 'time' in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)

        log_dict = dict(log_dict, **runner.log_buffer.output)

        # MOD: disable writing to files
        # self._dump_log(log_dict, runner)
        self._log_info(log_dict, runner)

        return log_dict

    def after_train_epoch(self, runner):
        if runner.log_buffer.ready:
            metrics = self.get_loggable_tags(runner)
            
            if 'val/pts_bbox_NuScenes/NDS' in metrics.keys():
                runner.logger.info('--- Evaluation Results ---')
                runner.logger.info('mAP: %.4f' % metrics['val/pts_bbox_NuScenes/mAP'])
                runner.logger.info('mATE: %.4f' % metrics['val/pts_bbox_NuScenes/mATE'])
                runner.logger.info('mASE: %.4f' % metrics['val/pts_bbox_NuScenes/mASE'])
                runner.logger.info('mAOE: %.4f' % metrics['val/pts_bbox_NuScenes/mAOE'])
                runner.logger.info('mAVE: %.4f' % metrics['val/pts_bbox_NuScenes/mAVE'])
                runner.logger.info('mAAE: %.4f' % metrics['val/pts_bbox_NuScenes/mAAE'])
                runner.logger.info('NDS: %.4f' % metrics['val/pts_bbox_NuScenes/NDS'])
            else:
                runner.logger.info('--- Evaluation Results ---')
                runner.logger.info('Vehicle mAPL: %.4f' % metrics['val/Vehicle mAPL'])
                runner.logger.info('Vehicle mAP: %.4f' % metrics['val/Vehicle mAP'])
                runner.logger.info('Vehicle mAPH: %.4f' % metrics['val/Vehicle mAPH'])
                runner.logger.info('Pedestrian mAPL: %.4f' % metrics['val/Pedestrian mAPL'])
                runner.logger.info('Pedestrian mAP: %.4f' % metrics['val/Pedestrian mAP'])
                runner.logger.info('Pedestrian mAPH: %.4f' % metrics['val/Pedestrian mAPH'])
                runner.logger.info('Cyclist mAPL: %.4f' % metrics['val/Cyclist mAPL'])
                runner.logger.info('Cyclist mAP: %.4f' % metrics['val/Cyclist mAP'])
                runner.logger.info('Cyclist mAPH: %.4f' % metrics['val/Cyclist mAPH'])
                runner.logger.info('Overall mAPL: %.4f' % metrics['val/Overall mAPL'])
                runner.logger.info('Overall mAP: %.4f' % metrics['val/Overall mAP'])
                runner.logger.info('Overall mAPH: %.4f' % metrics['val/Overall mAPH'])

def center_to_corner_box3d(centers, sizes, angles):
    """Convert centers, sizes and angles to corners.

    Args:
        centers (torch.tensor): (N, 3).
        dims (torch.tensor): (N, 3).
        angles (torch.tensor): (N, 1).

    Returns:
        torch.tensor: Corners with the shape of (N, 8, 3).
    """
    assert centers.shape[-1] == 3
    assert sizes.shape[-1] == 3
    assert angles.shape[-1] == 1
    input_dims = angles[..., 0].shape

    if len(input_dims) > 1:
        centers = centers.reshape(-1, 3)
        sizes = sizes.reshape(-1, 3)
        angles = angles.reshape(-1, 1)

    '''
    corners: x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1

                                               up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
            left y<-------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)

    '''
    corners_norm = torch.tensor([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ], dtype=torch.float32, device=sizes.device)
    corners_norm[:, :2] -= 0.5

    # corners: [N, 8, 3]
    corners = sizes.reshape(-1, 1, 3) * corners_norm.reshape(1, 8, 3)
    corners = rotation_3d_in_axis(corners, angles)

    corners += centers.reshape([-1, 1, 3])
    corners[..., 2:3] -= sizes[:, None, 2:3] / 2.0

    if len(input_dims) > 1:
        corners = corners.reshape(*input_dims, 8, 3)

    return corners

# @HOOKS.register_module()
# class MyTensorboardLoggerHook(LoggerHook):
#     def __init__(self, log_dir=None, interval=10, ignore_last=True, reset_flag=False, by_epoch=True):
#         super(MyTensorboardLoggerHook, self).__init__(
#             interval, ignore_last, reset_flag, by_epoch)
#         self.log_dir = log_dir

#     @master_only
#     def before_run(self, runner):
#         super(MyTensorboardLoggerHook, self).before_run(runner)
#         if self.log_dir is None:
#             self.log_dir = runner.work_dir
#         self.writer = SummaryWriter(self.log_dir)

#     @master_only
#     def log(self, runner):
#         tags = self.get_loggable_tags(runner)

#         for key, value in tags.items():
#             # MOD: merge into the 'train' group
#             if key == 'learning_rate':
#                 key = 'train/learning_rate'

#             # MOD: skip momentum
#             ignore = False
#             if key == 'momentum':
#                 ignore = True

#             # MOD: skip intermediate losses
#             for i in range(5):
#                 if key[:13] == 'train/d%d.loss' % i:
#                     ignore = True

#             if key[:3] == 'val':
#                 metric_name = key[22:]
#                 if metric_name in ['mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE', 'NDS']:
#                     key = 'val/' + metric_name
#                 else:
#                     ignore = True

#             if self.get_mode(runner) == 'train' and key[:5] != 'train':
#                 ignore = True

#             if self.get_mode(runner) != 'train' and key[:3] != 'val':
#                 ignore = True

#             if ignore:
#                 continue

#             if key[:5] == 'train':
#                 self.writer.add_scalar(key, value, self.get_iter(runner))
#             elif key[:3] == 'val':
#                 if not self.by_epoch:
#                     self.writer.add_scalar(key, value, self.get_iter(runner))
#                 else:
#                     self.writer.add_scalar(key, value, self.get_epoch(runner))

#     @master_only
#     def after_run(self, runner):
#         self.writer.close()

# modified from mmcv.runner.hooks.logger.wandb
@HOOKS.register_module()
class MyWandbLoggerHook(LoggerHook):
    """Class to log metrics with wandb.

    It requires `wandb`_ to be installed.


    Args:
        log_dir (str): directory for saving logs
            Default None.
        project_name (str): name for your project (mainly used to specify saving path on wandb server)
            Default None.
        team_name (str): name for your team (mainly used to specify saving path on wandb server)
            Default None.
        experiment_name (str): name for your run, if not specified, use the last part of log_dir
            Default None.
        interval (int): Logging interval (every k iterations).
            Default 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
            Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        commit (bool): Save the metrics dict to the wandb server and increment
            the step. If false ``wandb.log`` just updates the current metrics
            dict with the row argument and metrics won't be saved until
            ``wandb.log`` is called with ``commit=True``.
            Default: True.
        by_epoch (bool): Whether EpochBasedRunner is used.
            Default: True.
        with_step (bool): If True, the step will be logged from
            ``self.get_iters``. Otherwise, step will not be logged.
            Default: True.
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be uploaded to wandb.
            Default: ('.log.json', '.log', '.py').
            `New in version 1.4.3.`

    .. _wandb:
        https://docs.wandb.ai
    """
    def __init__(self, log_dir=None, project_name=None, team_name=None, experiment_name=None, 
                 interval=10, ignore_last=True, reset_flag=False, by_epoch=True, commit=True, 
                 with_step=True, out_suffix = ('.log.json', '.log', '.py')):
        
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.import_wandb()
        self.commit = commit
        self.with_step = with_step
        self.out_suffix = out_suffix
        
        self.log_dir = log_dir
        self.project_name = project_name
        self.team_name = team_name
        self.experiment_name = experiment_name
        if commit:
            os.system('wandb online')
        else:
            os.system('wandb offline')
            
    def import_wandb(self) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb
        
    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        if self.log_dir is None:
            self.log_dir = runner.work_dir
        if self.experiment_name is None:
            self.experiment_name = os.path.basename(self.log_dir)
        init_kwargs = dict(
            project=self.project_name,
            entity=self.team_name,
            notes=socket.gethostname(),
            name=self.experiment_name,
            dir=self.log_dir,
            reinit=True
        )
            
        if self.wandb is None:
            self.import_wandb()
        if init_kwargs:
            self.wandb.init(**init_kwargs)  # type: ignore
        else:
            self.wandb.init()  # type: ignore
    
    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        mode = self.get_mode(runner)
        if not tags:
            return
        if 'learning_rate' in tags.keys():
            tags['train/learning_rate'] = tags['learning_rate']
            del tags['learning_rate']
        if 'momentum' in tags.keys():
            del tags['momentum']

        # TODO support for nuscenes and waymo 
        # remove some keys in val
        # for k in list(tags.keys()):
        #     if k.startswith('val') and k.split('/')[-1] not in [
        #         'NDS', 'mAP',
        #         'mATE', 'mASE',
        #         'mAOE', 'mAVE', 
        #         'mAAE'
        #     ]:
        #         del tags[k]

        tags = {k: v for k, v in tags.items() if k.startswith(mode)}
        
        if self.with_step:
            self.wandb.log(
                tags, step=self.get_iter(runner), commit=self.commit)
        else:
            tags['global_step'] = self.get_iter(runner)
            self.wandb.log(tags, commit=self.commit)

    @master_only
    def after_run(self, runner) -> None:
        self.wandb.join()

class ModelEMA:
    """Model Exponential Moving Average from https://github.com/rwightman/
    pytorch-image-models Keep a moving average of everything in the model
    state_dict (parameters and buffers).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/
    ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training
    schemes to perform well.
    This class is sensitive where it is initialized in the sequence
    of model init, GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema_model = deepcopy(model).eval()
        self.ema = self.ema_model.module.module if is_parallel(
            self.ema_model.module) else self.ema_model.module
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, trainer, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(
                model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()


@HOOKS.register_module()
class MEGVIIEMAHook(Hook):
    """EMAHook used in BEVDepth.

    Modified from https://github.com/Megvii-Base
    Detection/BEVDepth/blob/main/callbacks/ema.py.
    """

    def __init__(self, init_updates=0, decay=0.9990, resume=None, interval=-1):
        super().__init__()
        self.init_updates = init_updates
        self.resume = resume
        self.decay = decay
        self.interval = interval

    def before_run(self, runner):
        from torch.nn.modules.batchnorm import SyncBatchNorm

        bn_model_list = list()
        bn_model_dist_group_list = list()
        for model_ref in runner.model.modules():
            if isinstance(model_ref, SyncBatchNorm):
                bn_model_list.append(model_ref)
                bn_model_dist_group_list.append(model_ref.process_group)
                model_ref.process_group = None
        runner.ema_model = ModelEMA(runner.model, self.decay)

        for bn_model, dist_group in zip(bn_model_list,
                                        bn_model_dist_group_list):
            bn_model.process_group = dist_group
        runner.ema_model.updates = self.init_updates

        if self.resume is not None:
            runner.logger.info(f'resume ema checkpoint from {self.resume}')
            cpt = torch.load(self.resume, map_location='cpu')
            load_state_dict(runner.ema_model.ema, cpt['state_dict'])
            runner.ema_model.updates = cpt['updates']

    def after_train_iter(self, runner):
        runner.ema_model.update(runner, runner.model.module)
        curr_step = runner.iter
        if self.interval>0:
            if curr_step % self.interval==0 and curr_step>0:
                self.save_checkpoint_iter(runner)
            

    def after_train_epoch(self, runner):
        self.save_checkpoint(runner)

    def after_run(self, runner):
        self.save_checkpoint_iter(runner)

    @master_only
    def save_checkpoint(self, runner):
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            'epoch': runner.epoch,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = f'epoch_{runner.epoch+1}_ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')
    
    @master_only
    def save_checkpoint_iter(self, runner):
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            'iter': runner.iter,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = f'iter_{runner.iter}_ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)

    return results

def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        NOTE bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        '''
        NOTE bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

# Modify from DistEvalHook
class CustomDistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        # Changed results to self.results so that MMDetWandbHook can access
        # the evaluation results and log them to wandb.

        # NOTE  use custom_multi_gpu_test, since sampler of val_loader is changed, 
        # result collect should also be changed 
        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        self.latest_results = results
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            # the key_score may be `None` so it needs to skip
            # the action to save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)