# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import random
import time

import _init_paths
import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist

from mmpt.apis import set_random_seed, train_model
from mmpt.datasets import build_dataset
from mmpt.models import build_model
from mmpt.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train an editor')
    parser.add_argument('--config', help='train config file path', default='/home/lr/project/mmpt/configs/train/local/pt/pips_rep_kubric_dino.py')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        default=False,
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--auto-seed', type=bool, default=True, help='auto generate random seed')

    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():    
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    torch.autograd.set_detect_anomaly(True)
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # cp code to work_dir
    if distributed and cfg.get('cp_project', True):
        file_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
        os.system(f"rsync -a --exclude 'output' --exclude '.git' {file_path} {cfg.work_dir}/")

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log env info
    env_info_dict = collect_env.collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if args.auto_seed:
        args.seed = random.randint(1, 1e5)
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    # build model and test model
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.init_weights()

    if cfg.get('model_test', False) and cfg.get('evaluation', False):
        if cfg.get('val_cfg', False):
            test_cfg = cfg.val_cfg
        else:
            test_cfg = cfg.test_cfg
        model_test = build_model(cfg.model_test, train_cfg=cfg.train_cfg, test_cfg=test_cfg)
    else:
        model_test = None
    
    if cfg.train_cfg.get('syncbn', False) and distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.text,
        )

    # meta information
    meta = dict()
    if cfg.get('exp_name', None) is None:
        cfg['exp_name'] = osp.splitext(osp.basename(cfg.work_dir))[0]
    meta['exp_name'] = cfg.exp_name
    meta['seed'] = args.seed
    meta['env_info'] = env_info

    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
        model_test=model_test)


if __name__ == '__main__':
    main()
