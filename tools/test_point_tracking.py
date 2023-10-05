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




def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('--config', help='test config file path', default='configs/train/local/eval/res18_d1_randomvideo_eval.py')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file', default='')
    parser.add_argument('--out-indices', nargs='+', type=int, default=[2])
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--frame-size',  nargs='+', type=int, default=[256,256])
    
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--video-dir', type=str, help='output result file', default='vis/pixel_tracking_test_lirui/rgb')
    parser.add_argument('--ref-dir', type=str, help='output result file', default='vis/pixel_tracking_test_lirui/000000.txt')
    
    parser.add_argument('--out', type=str, help='output result file', default='')
    parser.add_argument('--vis-out', type=str, help='output result file', default='vis/')
    
    parser.add_argument(
        '--eval',
        type=str,
        default='J&F-Mean',
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

def img2coord(imgs, num_poses, topk=5):
    clip_len = len(imgs)
    height, width = imgs.shape[2:]
    assert imgs.shape[:2] == (clip_len, num_poses)
    coords = np.zeros((2, num_poses, clip_len), dtype=np.float)
    imgs = imgs.reshape(clip_len, num_poses, -1)
    assert imgs.shape[-1] == height * width
    # [clip_len, NUM_KEYPOINTS, topk]
    topk_indices = np.argsort(imgs, axis=-1)[..., -topk:]
    topk_values = np.take_along_axis(imgs, topk_indices, axis=-1)
    topk_values = topk_values / (np.sum(topk_values, keepdims=True, axis=-1)+1e-9)
    topk_x = topk_indices % width
    topk_y = topk_indices // width
    # [clip_len, NUM_KEYPOINTS]
    coords[0] = np.sum(topk_x * topk_values, axis=-1).T
    coords[1] = np.sum(topk_y * topk_values, axis=-1).T
    coords[:, np.sum(imgs.transpose(1, 0, 2), axis=-1) == 0] = -1 

    return coords

def draw_label_map(img, pt, sigma):
    # Draw a 2D gaussian

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[1] - 3 * sigma), int(pt[0] - 3 * sigma)]
    br = [int(pt[1] + 3 * sigma + 1), int(pt[0] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0
            or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def main(args, imgs, ref_seg_map, num_point, ref):
    
    rank, _ = get_dist_info()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True


    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})

    if eval_config.get('dry_run', False):
        return 0
    
    if args.out_indices[0] == 3:
        return 0

    # Overwrite eval_config from args.eval
    # eval_config = merge_configs(eval_config, dict(metrics=args.eval))

    if 'out_indices' in eval_config:
        args.out_indices = eval_config['out_indices']
        eval_config.pop('out_indices')
    
    
    if 'output_dir' in eval_config and not args.out:
        args.tmpdir = eval_config['output_dir']
        eval_config['output_dir'] = eval_config['output_dir'] + f'indices{args.out_indices[0]}'
    else:
        args.tmpdir = args.out
        eval_config['output_dir'] = args.out + f'indices{args.out_indices[0]}'

    if 'checkpoint_path' in eval_config and not args.checkpoint:
        args.checkpoint = eval_config['checkpoint_path']
        eval_config.pop('checkpoint_path')
    else:
        eval_config.pop('checkpoint_path')
    
    if 'torchvision_pretrained' in eval_config: # bug
        eval_config.pop('torchvision_pretrained')
        

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    args.save_path = osp.join(cfg.work_dir, 'test_results')

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    imgs_raw = copy.deepcopy(imgs)
    original_shape = imgs[0].shape[:2]
    clip_len = len(imgs)
    data_pipeline = Compose(cfg.val_pipeline)
    data = {
        'imgs': imgs,
        'ref': ref.astype(np.float32),
        'ref_seg_map': ref_seg_map,
        'video_path': None,
        'original_shape': original_shape,
        'modality': 'RGB',
        'num_clips': 1,
        'num_proposals':1,
        'clip_len': len(imgs)
        } 

    data = data_pipeline(data)
    imgs = data['imgs'].unsqueeze(0).cuda()
    ref_seg_map = data['ref_seg_map'].unsqueeze(0).cuda().permute(0, 3, 1, 2)
    ref = data['ref'].unsqueeze(0).cuda()

    model = cfg.model

    model = build_model(model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.init_weights()

    args.save_image = args.save_path is not None
    empty_cache = cfg.get('empty_cache', False)
    
    if args.checkpoint:
        _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
        
        
    model = model.cuda()
    model.eval()
    
    with torch.no_grad():
        pred_coords = model.forward_test(imgs, ref_seg_map, [data['img_meta'].data], ref)[0]
    
    # pred_coords = img2coord(results, num_point)
    
    # visualization
    writer = SummaryWriter(f'{args.vis_out}', max_queue=10, flush_secs=60)
    log_freq = 99999
    sw = pips_vis.Summ_writer(  
                            writer=writer,
                            log_freq=log_freq,
                            fps=24,
                            scalar_freq=int(log_freq/2),
                            just_gif=True
                            )
    
    
    # B x T x C x H x W
    frames = torch.from_numpy(np.stack(imgs_raw, axis=0))[None].permute(0,1,4,2,3).cuda()
    frames  = frames.float() * 1./255 - 0.5
    gray_rgbs = torch.mean(frames, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
    
    trajs_e = torch.zeros((1, clip_len, num_point, 2)).cuda()
    
    
    for img_idx in range(clip_len):
        for t in range(num_point):
            
            predx = pred_coords[0, t, img_idx]
            predy = pred_coords[1, t, img_idx]
                            
            trajs_e[0, img_idx, t, 0] = predx
            trajs_e[0, img_idx, t, 1] = predy
            
    for n in range(num_point):
        sw.summ_traj2ds_on_rgbs(f'kp_trajs_e_on_{n}_rgbs', trajs_e[0:1,:,n:n+1], gray_rgbs[0:1,:clip_len], cmap='spring', linewidth=2)   

if __name__ == '__main__':
    
    args = parse_args()
    
    scale = 2
    
    # read frames
    imgs = []
    frames_path = sorted(glob.glob(os.path.join(args.video_dir, '*.png')))
    for idx, fp in enumerate(frames_path):
        f = mmcv.imread(fp, channel_order='rgb')
        if idx == 0:
            ori_H, ori_W = f.shape[:2]
        
        if args.frame_size[0] != -1:
            f = mmcv.imresize(f, args.frame_size)
            
        imgs.append(f)
    
    # imgs = imgs[:20]
    
    H, W = imgs[0].shape[:2]
    
    ref = []
    # read ref
    with open(args.ref_dir, 'r') as f:
        for line in f.readlines():
            x, y = line.strip('\n').split()
            
            new_x = int(x) * W / ori_W
            new_y = int(y) * H / ori_H
            
            ref.append(np.array([new_y, new_x]))
    
    ref = np.stack(ref, 0)
            
    
    num_points = ref.shape[0]
    
    pose_map = np.zeros((H//2, W//2, num_points), dtype=np.float)
    pose_coord = ref.transpose(1,0)
    
    for j in range(num_points):

        pose_map[:, :, j] = draw_label_map(pose_map[:, :, j], pose_coord[:, j] / scale, 3)
    
    # return the coords for K points
    pred_coords = main(args, imgs, pose_map, num_points, pose_coord)
    
    print('finish')