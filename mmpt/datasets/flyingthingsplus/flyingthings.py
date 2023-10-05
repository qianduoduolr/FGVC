from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os.path as osp
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
from .utils.basic import *
from .utils.improc import *
from glob import glob
import json
import imageio
import cv2
import re
import sys
from torchvision.transforms import ColorJitter, GaussianBlur

from ..registry import DATASETS
from ..base_dataset import BaseDataset
from ..video_dataset import *
from mmpt.utils import *

from .utils.read_utils import read_gen




@DATASETS.register_module()
class FlyingThingsDatasetNormal(Video_dataset_base):
    def __init__(self,  scale=8, radius=9, sigma=1, return_heat_map=True, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.scale = scale
        self.sigma = sigma
        self.radius = radius
        self.load_annotations()
        self.return_heat_map = return_heat_map

    def load_annotations(self):
        
        self.samples = []
        
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(self.root, 'frames_cleanpass_webp', 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(self.root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.webp')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        
                        sample = {}
                        # note the direction, we make a rule the correlation is computed from frame2 to frame1, the f2-> f1 is the forward flow
                        if direction == 'into_future':
                            sample['frames_path'] = [ images[i], images[i+1] ]
                            sample['flow_back_path'] = flows[i] 
                            sample['flow_path'] = flows[i+1].replace('IntoFuture', 'IntoPast').replace('into_future', 'into_past')
                        elif direction == 'into_past':
                            sample['frames_path'] = [ images[i+1], images[i] ]
                            sample['flow_back_path'] = flows[i+1] 
                            sample['flow_path'] = flows[i].replace('IntoPast', 'IntoFuture').replace('into_past', 'into_future')
                            
        
                        self.samples.append(sample)
                        
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))
                        
    
    
    def draw_label_map(self, img, pt, sigma, normalize=False):
        # Draw a 2D gaussian

        ty = round(pt[1])
        tx = round(pt[0])
        
        flag = (0 <= tx <= 2*self.radius and 0 <= ty <= 2*self.radius)
        if not flag:
            # print('haha')
            return img


        # Check that any part of the gaussian is in-bounds
        x_l = min(tx, 3 * sigma)
        x_r = min(img.shape[1] - tx - 1, 3 * sigma)
        ul = [tx - x_l, tx + x_r+1]
        
        y_t = min(ty, 3 * sigma)
        y_b = min(img.shape[0] - ty - 1, 3 * sigma)
        vl = [ty - y_t, ty + y_b+1]

        # Generate gaussian
        size = 6 * sigma + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        # Usable gaussian range
        g_t = g[y0-y_t:y0+y_b+1, x0-x_l:x0+x_r+1]

        img[vl[0]:vl[1], ul[0]:ul[1]] = g_t
        return img
    
    
    def prepare_train_data(self, idx):
        
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        flows_path = sample['flow_path']
        flows_back_path = sample['flow_back_path']
        num_frames = len(frames_path)
        
        # load frame
        frames = self._parser_rgb_rawframe([0], frames_path, clip_length=2, step=1)
        flow = read_gen(flows_path)
        flow_back = read_gen(flows_back_path)
        
        
        data = {
            'imgs': frames,
            'flow': flow,
            'flow_back': flow_back,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': num_frames
        } 
        
        if not self.return_heat_map:
            
            return self.pipeline(data)
        
        else:
            data = self.pipeline(data)
            
            flow = data['flow'].numpy()
            
            height, width = flow.shape[:2]
            
            height_ = height // self.scale
            width_ = width // self.scale
            
            pose_map = np.zeros((height_, width_, self.radius*2+1, self.radius*2+1), dtype=np.float)
            pose_coord = flow[::self.scale, ::self.scale, :] / self.scale
            
            for i in range(height_):
                for j in range(width_):
                    if self.sigma > 0:
                        pose_map[i,j] = self.draw_label_map(pose_map[i, j], pose_coord[i, j]+self.radius, self.sigma)
                    else:
                        ty = round(pose_coord[i, j, 1] + self.radius)
                        tx = round(pose_coord[i, j, 0] + self.radius)
                        if 0 <= tx <= 2*self.radius and 0 <= ty <= 2*self.radius:
                            pose_map[i, j, ty, tx] = 1.0
                        
            data['valid'] = torch.from_numpy((pose_map.reshape(height_, width_, -1).sum(-1) > 0))          
            data['heat_map'] = torch.from_numpy(pose_map)

            
            return data