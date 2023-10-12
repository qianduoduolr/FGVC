# Copyright (c) OpenMMLab. All rights reserved.
import _init_paths
import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpt.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmpt.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpt.datasets import build_dataloader, build_dataset
from mmpt.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('--config', help='test config file path', default='/home/lr/project/mmpt_ops/configs/test/res18_d8.py')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file', default='')
    parser.add_argument('--out-indices', nargs='+', type=int, default=[0])
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--out', type=str, help='output result file', default='')
    parser.add_argument(
        '--eval',
        type=str,
        default='davis',
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')

    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--save-path',
        default=None,
        type=str,
        help='path to store images and if not given, will not save image')
    parser.add_argument(
        '--task',
        default='davis',
        help='which task')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1

def main():
    args = parse_args()
    rank, _ = get_dist_info()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})

    if eval_config.get('dry_run', False):
        return 0

    # Overwrite eval_config from args.eval
    # eval_config = merge_configs(eval_config, dict(metrics=args.eval))
    
    if 'output_dir' in eval_config and not args.out:
        args.tmpdir = eval_config['output_dir']
        eval_config['output_dir'] = eval_config['output_dir'] + args.task
    else:
        args.tmpdir = args.out
        eval_config['output_dir'] = args.out  + args.task

    if 'checkpoint_path' in eval_config and not args.checkpoint:
        args.checkpoint = eval_config['checkpoint_path']
        eval_config.pop('checkpoint_path')
    else:
        eval_config.pop('checkpoint_path')
        

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)


    # set random seeds
    if args.seed is not None:
        if rank == 0:
            print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    data = cfg.data.get(f'test_{args.task}')
    dataset = build_dataset(data)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }

    data_loader = build_dataloader(dataset, **loader_cfg)
    cfg.test_cfg = cfg.get(f'test_cfg_{args.task}')


    # build the model and load checkpoint
    eval_arc = cfg.get('eval_arc', 'VanillaTracker')
    model = mmcv.ConfigDict(type=eval_arc, backbone=cfg.model.backbone)
    model.backbone.out_indices = cfg.test_cfg.out_indices
    model.backbone.strides = cfg.test_cfg.strides
    
    if cfg.test_cfg.get('dilations', False):
        model.backbone.dilations = cfg.test_cfg.dilations
    
    if 'torchvision_pretrained' in eval_config:
        model.backbone.pretrained = eval_config['torchvision_pretrained']
        eval_config.pop('torchvision_pretrained')


    model = build_model(model, train_cfg=None, test_cfg=cfg.test_cfg)
    model.init_weights()

    args.save_image = args.save_path is not None
    empty_cache = cfg.get('empty_cache', False)
    if not distributed:
        if args.checkpoint:
            _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(
            model,
            data_loader,
            save_path=args.save_path,
            save_image=args.save_image)
    else:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = DistributedDataParallelWrapper(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

        device_id = torch.cuda.current_device()

        if args.checkpoint:

            _ = load_checkpoint(
                model,
                args.checkpoint,
                map_location='cpu')

        outputs = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            save_path=args.save_path,
            save_image=args.save_image,
            empty_cache=empty_cache)

    rank, _ = get_dist_info()
    if rank == 0:
        if eval_config:
            # predict
            eval_res = dataset.evaluate(outputs, **eval_config)
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}')

if __name__ == '__main__':
    main()
