import os
from re import I
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size
import mmcv
from PIL import Image
import copy
from mmcv.runner import load_state_dict


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1), norm_mode='0-1'):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    n_dim = tensor.dim()
    
    if norm_mode == '0-1':
        tensor = tensor.float().cpu() # clamp
        
        tensor_max = tensor.flatten(-2).max(-1, keepdim=True)[0].reshape(*tensor.shape[:2], 1, 1)
        tensor_min = tensor.flatten(-2).min(-1, keepdim=True)[0].reshape(*tensor.shape[:2], 1, 1)
        
        tensor = (tensor - tensor_min) * 255 / (tensor_max - tensor_min)  # to range [0,1]
        
        tensor = tensor.squeeze()
        
    elif norm_mode == 'mean-std':
        if n_dim != 2:
            tensor = tensor.squeeze(0).float().cpu() # clamp
            mean=torch.tensor([123.675, 116.28, 103.53]).reshape(3,1,1)
            std=torch.tensor([58.395, 57.12, 57.375]).reshape(3,1,1)
            tensor = (tensor * std) + mean
            tensor = tensor.clamp(0,255)
    elif norm_mode == 'mean-std-lab':
        if n_dim != 2:
            tensor = tensor.squeeze(0).float().cpu() # clamp
            mean=torch.tensor([50,0,0]).reshape(3,1,1)
            std=torch.tensor([50, 127, 127]).reshape(3,1,1)
            tensor = (tensor * std) + mean
            # tensor = tensor.clamp(0,255)
    else:
        tensor = tensor.squeeze().float().cpu().clamp(0, 1)
        tensor = tensor * 255

    if n_dim == 4:
        n_img = len(tensor)
        
        if tensor.dim() == 3:
            tensor = tensor[:, None]
            
        img_np = make_grid(tensor, nrow=int(n_img), normalize=False, pad_value=10).numpy()

        if norm_mode == 'mean-std-lab':
            img_np = np.transpose(img_np, (1, 2, 0))  # HWC
            img_np = cv2.cvtColor(img_np, cv2.COLOR_Lab2BGR) * 255
        else:
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR

    elif n_dim == 3:
        img_np = tensor.numpy()
        if norm_mode == 'mean-std-lab':
            img_np = np.transpose(img_np, (1, 2, 0))  # HWC
            img_np = cv2.cvtColor(img_np, cv2.COLOR_Lab2BGR) * 255
        else:
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            
    elif n_dim == 2:
        img_np = tensor.squeeze().float().cpu().numpy()
        img_np = ((img_np - img_np.min()) * 255 / ( img_np.max() - img_np.min())).astype(np.uint8)
        img_np = cv2.applyColorMap(img_np, cv2.COLORMAP_JET)
        
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    # if out_type == np.uint8:
    #     img_np = (img_np * 255).round() if norm_mode == '0-1' else (img_np ).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)



def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

def copy_params(model, model_test):
    origin_params = {}
    for name, param in model.state_dict().items():
        if name in model_test.state_dict().keys():
            origin_params[name.replace('module.','')] = param.data.detach().cpu()
    load_state_dict(model_test,origin_params, strict=False)

def make_pbs(exp_name, docker_name):
    pbs_data = ""
    with open('/home/lr/project/mmpt/configs/pbs/template.pbs', 'r') as f:
        for line in f:
            line = line.replace('exp_name',f'{exp_name}')
            line = line.replace('docker_name', f'{docker_name}')
            pbs_data += line

    with open(f'/home/lr/project/mmpt/configs/pbs/{exp_name}.pbs',"w") as f:
        f.write(pbs_data)

def make_local_config(exp_name, file='motion_prediction'):
    config_data = ""
    with open(f'/home/lr/project/mmpt/configs/train/local/{file}/{exp_name}.py', 'r') as f:
        for line in f:
            line = line.replace('/home/lr','/gdata/lirui')
            # line = line.replace('/gdata/lirui/dataset/YouTube-VOS','/dev/shm')
            # line = line.replace('/dev/shm/', '/gdata/lirui/dataset/YouTube-VOS/')
            # line = line.replace('/home/lr/dataset','/home/lr/dataset')
            config_data += line

    with open(f'/home/lr/project/mmpt/configs/train/ypb/{file}/{exp_name}.py',"w") as f:
        f.write(config_data)


def make_local_config_hanhai(exp_name, file='motion_prediction'):
    config_data = ""
    with open(f'/home/lr/project/mmpt/configs/train/local/{file}/{exp_name}.py', 'r') as f:
        for line in f:
            line = line.replace('/home/lr','/home/sist/lirui')
            # line = line.replace('/gdata/lirui/dataset/YouTube-VOS','/dev/shm')
            # line = line.replace('/dev/shm/', '/gdata/lirui/dataset/YouTube-VOS/')
            # line = line.replace('/home/lr/dataset','/home/lr/dataset')
            config_data += line

    with open(f'/home/lr/project/mmpt/configs/train/hanhai/{file}/{exp_name}.py',"w") as f:
        f.write(config_data)
        
def make_local_config_back(exp_name, file='motion_prediction'):
    config_data = ""
    with open(f'/home/lr/project/mmpt/configs/train/ypb/{file}/{exp_name}.py', 'r') as f:
        for line in f:
            line = line.replace('/gdata/lirui', '/home/lr')
            line = line.replace('/dev/shm', '/home/lr/dataset/YouTube-VOS')
            # line = line.replace('/home/lr/dataset','/home/lr/dataset')
            config_data += line

    with open(f'/home/lr/project/mmpt/configs/train/local/{file}/{exp_name}.py',"w") as f:
        f.write(config_data)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    
    