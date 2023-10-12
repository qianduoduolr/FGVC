import copy
from operator import length_hint
import os.path as osp
from collections import defaultdict
from pathlib import Path
from glob import glob
import os
import random
import pickle
from mmcv.fileio.io import load
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

from mmcv import scandir
import mmcv

from ..base_dataset import BaseDataset
from ..video_dataset import *
from ..registry import DATASETS

from ..pipelines import Compose
from mmpt.utils import *
from .utils.read_utils import read_gen



@DATASETS.register_module()
class Flyingthings_ytv_dataset_rgb(Video_dataset_base):
    def __init__(self, root_flow, 
                       data_prefix, 
                       pipeline_sup,
                       steps=dict(v=[1], p=[1]),
                       year='2018',
                       **kwargs
                       ):
        super().__init__(**kwargs)

        self.root_flow = root_flow
        self.data_prefix = data_prefix
        self.year = year
        self.steps = steps

        self.step = max(self.steps['v'])
        self.load_annotations()
        
        self.pipeline_sup = Compose(pipeline_sup)

    def __len__(self):
        return len(self.samples_sup)

    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, self.year, self.data_prefix['RGB'])
        list_path = osp.join(self.list_path, f'youtube{self.year}_{self.split}.json')
        data = mmcv.load(list_path)
        
        for vname, frames in data.items():
            sample = dict()
            sample['frames_path'] = []
            for frame in frames:
                sample['frames_path'].append(osp.join(self.video_dir, vname, frame))
                
            sample['num_frames'] = len(sample['frames_path'])
            if sample['num_frames'] <= self.clip_length * self.step:
                continue
        
            self.samples.append(sample)

        self.samples_sup = []
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(self.root_flow, 'frames_cleanpass_webp', 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(self.root_flow, 'optical_flow/TRAIN/*/*')))
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
                            
                        self.samples_sup.append(sample)
        
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples_sup)))

    
    
    def prepare_train_data(self, idx):
        
        if self.data_backend == 'lmdb' and self.env == None and self.txn == None:
            self._init_db(self.video_dir)

        sample = self.samples[idx % len(self.samples)]
        frames_path = sample['frames_path']
        num_frames = sample['num_frames']
        
        step = np.random.choice(self.steps['v'], p=np.array(self.steps['p']).ravel())

        offsets = self.temporal_sampling(num_frames, self.num_clips, self.clip_length, step, mode=self.temporal_sampling_mode)

        # load frame
        if self.data_backend == 'raw_frames':
            frames = self._parser_rgb_rawframe(offsets, frames_path, self.clip_length, step=step)
        elif self.data_backend == 'lmdb':
            frames = self._parser_rgb_lmdb(self.txn, offsets, frames_path, self.clip_length, step=step)
            
            
        data = {
            'imgs': frames,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': self.clip_length
        } 

        data = self.pipeline(data)

        sample_sup = self.samples_sup[idx]
        frames_path = sample_sup['frames_path']
        flows_path = sample_sup['flow_path']
        flows_back_path = sample_sup['flow_back_path']
        num_frames = len(frames_path)
        
        # load frame
        frames_sup = self._parser_rgb_rawframe([0], frames_path, clip_length=2, step=1)
        flow = read_gen(flows_path)
        flow_back = read_gen(flows_back_path)
        
        data_sup = {
            'imgs': frames_sup,
            'flow': flow,
            'flow_back': flow_back,
            'modality': 'RGB',
            'num_clips': self.num_clips,
            'num_proposals':1,
            'clip_len': num_frames
        } 
        
        data_sup = self.pipeline_sup(data_sup)
        data_sup['imgs_sup'] = data_sup['imgs']
        
        final_data = {
                **data,
                **dict((k, data_sup[k]) for k in [
                   'imgs_sup',
                   'flow',
                   'flow_back'
               ] if k in data_sup)
                }
        
        return final_data