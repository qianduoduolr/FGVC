# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from pickle import FALSE

import _init_paths
import mmcv
import torch
import glob
import numpy as np
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from tensorboardX import SummaryWriter


from mmpt.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmpt.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpt.datasets import build_dataloader, build_dataset
from mmpt.models import build_model
from mmpt.datasets.pipelines import Compose  
from mmpt.utils import *
from mmpt.datasets.flyingthingsplus.utils import improc as pips_vis
from mmpt.datasets.flyingthingsplus.utils import visualize




def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('--config', help='test config file path', default='/home/lr/project/mmpt/configs/train/local/eval/demo_pips.py')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file', default='')
    parser.add_argument('--size', nargs='+', type=int, default=[512, 384])
    parser.add_argument('--query-points', nargs='+', type=float, default=[0, 290, 192])
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--video-dir', type=str, help='output result file', default='/home/lr/dataset/YouTube-VOS/2018/train_all_frames/JPEGImages/c0a1e06710')
    parser.add_argument('--output-dir', type=str, help='output result file', default='./demo/')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
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



def main(args, imgs, query_points):
    
    rank, _ = get_dist_info()
    
    num_point = query_points.shape[0]
    
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # Load eval_config from cfg
    output_dir = args.output_dir
   
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

    
    # data processing
    original_shape = imgs[0].shape[:2]
    clip_len = len(imgs)
    data_pipeline = Compose(cfg.val_pipeline)
    data = {
        'rgbs':copy.deepcopy(imgs), 
        'query_points': query_points, 
        'trajectories': np.zeros((1, clip_len, num_point, 2)), 
        'visibilities': np.ones((1, clip_len, num_point)),
        'video_path': None,
        'original_shape': original_shape,
        'modality': 'RGB',
        'num_clips': 1,
        'num_proposals':1,
        'clip_len': len(imgs)
        } 
    data = data_pipeline(data)

    # build model 
    model = cfg.model

    model = build_model(model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.init_weights()

    
    if args.checkpoint:
        _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
        
    model = model.cuda()
    model.eval()
    
    # inference
    with torch.no_grad():
        _, vis, trajectories_pred, _, _ = model.forward_test(**data)
    
    
    # visualization
    imgs = np.array(imgs)
    writer = SummaryWriter(output_dir, max_queue=10, flush_secs=60)
    log_freq = 99999
    sw = pips_vis.Summ_writer(  
                            writer=writer,
                            log_freq=log_freq,
                            fps=24,
                            scalar_freq=int(log_freq/2),
                            just_gif=True
                            )
    
    
    # B x T x C x H x W
    frames = torch.from_numpy(np.stack(imgs, axis=0))[None].permute(0,1,4,2,3).cuda()
    frames  = frames.float() * 1./255 - 0.5
    
    
    video = visualize.paint_point_track(imgs, trajectories_pred[0].transpose(0,1).cpu().numpy(), vis[0].transpose(0,1).cpu().numpy())
            
    generate_video(video, os.path.join(output_dir, f'demo_prediction.mp4'))
    
    rgbs_input = torch.from_numpy(imgs)[None].permute(0,1,4,2,3).cuda()
    rgbs_input  = rgbs_input.float() * 1./255 - 0.5
    

    for n in range(num_point):
        
        start_idx = 0
        
        # color point/line
        vis_imgs = sw.summ_traj2ds_on_rgbs(f'{output_dir}/kp_demo_{n}_trajs_e_on_rgbs', trajectories_pred[0:1,start_idx:,n:n+1], rgbs_input[0:1,start_idx:], cmap='spring', linewidth=2, only_return=True)  
        
        vis_imgs = vis_imgs[0].permute(0,2,3,1).detach().cpu().numpy()
        
        generate_video(vis_imgs, os.path.join(output_dir, f'demo_prediction_{n}.mp4'))
        

if __name__ == '__main__':
    
    args = parse_args()
    
    # convert to video frames
    imgs = get_video_frames_cv(args.video_dir, size=args.size, is_decode=True)
    if args.query_points:
        query_points = np.array(args.query_points).astype(float)[None, None, ...]
    else:
        query_points = None

            
    pred_coords = main(args, imgs, query_points)
    
    print('finish')