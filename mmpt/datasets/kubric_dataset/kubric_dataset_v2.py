import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import copy
import wandb
import json

import torch
from torch.utils import data

import random
from glob import glob
import pickle

from collections import *
import pdb
from mmcv.utils import print_log
import mmcv
from mmpt.utils import *
from mmpt.datasets.flyingthingsplus.utils.figures import *
from tensorboardX import SummaryWriter
from dataclasses import dataclass
from typing import Any, Optional
from torchvision.transforms import ColorJitter, GaussianBlur

from ..base_dataset import BaseDataset
from ..video_dataset import *
from ..registry import DATASETS
from ..tapvid_evaluation_datasets import *



@dataclass(eq=False)
class CoTrackerData:
    """
    Dataclass for storing video tracks data.
    """

    video: torch.Tensor  # B, S, C, H, W
    trajectory: torch.Tensor  # B, S, N, 2
    visibility: torch.Tensor  # B, S, N
    # optional data
    valid: Optional[torch.Tensor] = None  # B, S, N
    seq_name: Optional[str] = None
    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format
    segmentation: Optional[torch.Tensor] = None  # B, S, 1, H, W
    


@DATASETS.register_module()
class KubricDatasetV2(Video_dataset_base):
    """
    An iterator that loads a TAP-Vid dataset and yields its elements.
    The elements consist of videos of arbitrary length.
    """

    def __init__(
        self,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=64,
        sample_vis_1st_frame=False,
        use_augs=False,
        *args, 
        **kwargs
        ):
        super(KubricDatasetV2, self).__init__(*args, **kwargs)
        np.random.seed(0)
        torch.manual_seed(0)
        # self.data_root = data_root
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample
        self.sample_vis_1st_frame = sample_vis_1st_frame
        self.use_augs = use_augs
        self.crop_size = crop_size

        # photometric augmentation
        self.photo_aug = ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14
        )
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))

        self.blur_aug_prob = 0.25
        self.color_aug_prob = 0.25

        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [2, 100]
        self.eraser_max = 10

        # occlusion augmentation
        self.replace_aug_prob = 0.5
        self.replace_bounds = [2, 100]
        self.replace_max = 10

        # spatial augmentations
        self.pad_bounds = [0, 100]
        self.crop_size = crop_size
        self.resize_lim = [0.25, 2.0]  # sample resizes from here
        self.resize_delta = 0.2
        self.max_crop_offset = 50

        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5
        
        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]  # sample resizes from here
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        
        self.load_annotations()
        
    def __len__(self):
        return len(self.samples)
          
    def load_annotations(self):
        
        logger = get_root_logger()
        
        self.samples = glob(osp.join(self.root, self.filename_tmpl))
                      
        logger.info(f"Loaded Kubric dataset")
        logger.info('found %d unique videos' % (len(self.samples)))
        
        return 0

    def getitem_helper(self, index):
        
        gotit = True
        sample = np.load(self.samples[index], allow_pickle=True).item()

        traj_2d = sample["target_points"][0]
        visibility = sample["occluded"][0]
        rgbs = [ (sample['video'][0][i] + 1) * 255 / 2 for i in range(sample['video'].shape[1])]
        
        # a = rgbs[0].astype(np.uint8)
        # b = rgbs[1].astype(np.uint8)
        # c = rgbs[6].astype(np.uint8)

        # random crop
        assert self.seq_len <= len(rgbs)
        if self.seq_len < len(rgbs):
            start_ind = np.random.choice(len(rgbs) - self.seq_len, 1)[0]

            rgbs = rgbs[start_ind : start_ind + self.seq_len]
            traj_2d = traj_2d[:, start_ind : start_ind + self.seq_len]
            visibility = visibility[:, start_ind : start_ind + self.seq_len]

        traj_2d = np.transpose(traj_2d, (1, 0, 2))
        visibility = np.transpose(np.logical_not(visibility), (1, 0))
        if self.use_augs:
            rgbs, traj_2d, visibility = self.add_photometric_augs(
                rgbs, traj_2d, visibility
            )
            rgbs, traj_2d = self.add_spatial_augs(rgbs, traj_2d, visibility)
        else:
            rgbs, traj_2d = self.crop(rgbs, traj_2d)

        visibility[traj_2d[:, :, 0] > self.crop_size[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > self.crop_size[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False
        
        
        # self.visualize_data(rgbs, traj_2d, visibility)
        
        
        sample = {
            'imgs':rgbs,
            'trajs':traj_2d,
            'visibles':visibility,
            'num_clips': 1,
            'modality': 'RGB',
            'clip_len': self.seq_len
        }

        sample = self.pipeline(sample)
        
        visibility = sample['visibles']
        traj_2d = sample['trajs']

        visibile_pts_first_frame_inds = (visibility[0]).nonzero(as_tuple=False)[:, 0]

        if self.sample_vis_1st_frame:
            visibile_pts_inds = visibile_pts_first_frame_inds
        else:
            visibile_pts_mid_frame_inds = (visibility[self.seq_len // 2]).nonzero(
                as_tuple=False
            )[:, 0]
            visibile_pts_inds = torch.cat(
                (visibile_pts_first_frame_inds, visibile_pts_mid_frame_inds), dim=0
            )
        point_inds = torch.randperm(len(visibile_pts_inds))[: self.traj_per_sample]
        if len(point_inds) < self.traj_per_sample:
            gotit = False

        visible_inds_sampled = visibile_pts_inds[point_inds]

        trajs = traj_2d[:, visible_inds_sampled].float()
        visibles = visibility[:, visible_inds_sampled]
        valids = torch.ones((self.seq_len, self.traj_per_sample))
        
        sample.update({
            'trajs': trajs,
            'visibles': visibles,
            'valids': valids
        })
     
        return sample, gotit
    
    def prepare_train_data(self, idx):
            
        gotit = False

        sample, gotit = self.getitem_helper(idx)
        if not gotit:
            # print("warning: sampling failed")
            # fake sample, so we can still collate
            sample = dict(
                imgs=torch.zeros(
                    (1, 3, self.seq_len, self.crop_size[0], self.crop_size[1])
                ),
                trajs=torch.zeros((self.seq_len, self.traj_per_sample, 2)),
                visibles=torch.zeros((self.seq_len, self.traj_per_sample)),
                valids=torch.zeros((self.seq_len, self.traj_per_sample)),
            )

        sample.update(gotit=gotit)
        
        return sample

    def visualize_data(self, imgs, trajs, vis):
        T, P, _ = trajs.shape
        point_id = random.randint(0, P-1)
        # st = inds[point_id].item()
        v = vis[:, point_id]
        
        vis_rgbs = []

        frames = []
        for t in range(T):
            coord = trajs[t, point_id]
            frame = imgs[t].astype(np.uint8)
            frame = cv2.drawMarker(frame.copy(), position=(int(coord[0]),int(coord[1])),color=(255, 255, 0),markerSize = 10, markerType=cv2.MARKER_TILTED_CROSS, thickness=3)
            frames.append(frame)
        frames = np.concatenate(frames, 1)
        vis_rgbs.append(frames)
        
        vis_rgbs = np.concatenate(vis_rgbs, 0)
        
        return 0

    def add_photometric_augs(self, rgbs, trajs, visibles, eraser=True, replace=True):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        if eraser:
            ############ eraser transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.eraser_max + 1)
                    ):  # number of times to occlude

                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(
                            self.eraser_bounds[0], self.eraser_bounds[1]
                        )
                        dy = np.random.randint(
                            self.eraser_bounds[0], self.eraser_bounds[1]
                        )
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        mean_color = np.mean(
                            rgbs[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0
                        )
                        rgbs[i][y0:y1, x0:x1, :] = mean_color

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        if replace:

            rgbs_alt = [
                np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8)
                for rgb in rgbs
            ]
            rgbs_alt = [
                np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8)
                for rgb in rgbs_alt
            ]

            ############ replace transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.replace_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(
                            self.replace_bounds[0], self.replace_bounds[1]
                        )
                        dy = np.random.randint(
                            self.replace_bounds[0], self.replace_bounds[1]
                        )
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        wid = x1 - x0
                        hei = y1 - y0
                        y00 = np.random.randint(0, H - hei)
                        x00 = np.random.randint(0, W - wid)
                        fr = np.random.randint(0, S)
                        rep = rgbs_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        rgbs[i][y0:y1, x0:x1, :] = rep

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [
                np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8)
                for rgb in rgbs
            ]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [
                np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8)
                for rgb in rgbs
            ]

        return rgbs, trajs, visibles

    def add_spatial_augs(self, rgbs, trajs, visibles):
        T, N, __ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        rgbs = [rgb.astype(np.float32) for rgb in rgbs]

        ############ spatial transform ############

        # padding
        pad_x0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])

        rgbs = [
            np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs
        ]
        trajs[:, :, 0] += pad_x0
        trajs[:, :, 1] += pad_y0
        H, W = rgbs[0].shape[:2]

        # scaling + stretching
        scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
        scale_x = scale
        scale_y = scale
        H_new = H
        W_new = W

        scale_delta_x = 0.0
        scale_delta_y = 0.0

        rgbs_scaled = []
        for s in range(S):
            if s == 1:
                scale_delta_x = np.random.uniform(-self.resize_delta, self.resize_delta)
                scale_delta_y = np.random.uniform(-self.resize_delta, self.resize_delta)
            elif s > 1:
                scale_delta_x = (
                    scale_delta_x * 0.8
                    + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                )
                scale_delta_y = (
                    scale_delta_y * 0.8
                    + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                )
            scale_x = scale_x + scale_delta_x
            scale_y = scale_y + scale_delta_y

            # bring h/w closer
            scale_xy = (scale_x + scale_y) * 0.5
            scale_x = scale_x * 0.5 + scale_xy * 0.5
            scale_y = scale_y * 0.5 + scale_xy * 0.5

            # don't get too crazy
            scale_x = np.clip(scale_x, 0.2, 2.0)
            scale_y = np.clip(scale_y, 0.2, 2.0)

            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # make it at least slightly bigger than the crop area,
            # so that the random cropping can add diversity
            H_new = np.clip(H_new, self.crop_size[0] + 10, None)
            W_new = np.clip(W_new, self.crop_size[1] + 10, None)
            # recompute scale in case we clipped
            scale_x = W_new / float(W)
            scale_y = H_new / float(H)

            rgbs_scaled.append(
                cv2.resize(rgbs[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
            )
            trajs[s, :, 0] *= scale_x
            trajs[s, :, 1] *= scale_y
        rgbs = rgbs_scaled

        ok_inds = visibles[0, :] > 0
        vis_trajs = trajs[:, ok_inds]  # S,?,2

        if vis_trajs.shape[1] > 0:
            mid_x = np.mean(vis_trajs[0, :, 0])
            mid_y = np.mean(vis_trajs[0, :, 1])
        else:
            mid_y = self.crop_size[0]
            mid_x = self.crop_size[1]

        x0 = int(mid_x - self.crop_size[1] // 2)
        y0 = int(mid_y - self.crop_size[0] // 2)

        offset_x = 0
        offset_y = 0

        for s in range(S):
            # on each frame, shift a bit more
            if s == 1:
                offset_x = np.random.randint(
                    -self.max_crop_offset, self.max_crop_offset
                )
                offset_y = np.random.randint(
                    -self.max_crop_offset, self.max_crop_offset
                )
            elif s > 1:
                offset_x = int(
                    offset_x * 0.8
                    + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                    * 0.2
                )
                offset_y = int(
                    offset_y * 0.8
                    + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                    * 0.2
                )
            x0 = x0 + offset_x
            y0 = y0 + offset_y

            H_new, W_new = rgbs[s].shape[:2]
            if H_new == self.crop_size[0]:
                y0 = 0
            else:
                y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)

            if W_new == self.crop_size[1]:
                x0 = 0
            else:
                x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)

            rgbs[s] = rgbs[s][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            trajs[s, :, 0] -= x0
            trajs[s, :, 1] -= y0

        H_new = self.crop_size[0]
        W_new = self.crop_size[1]

        # flip
        h_flipped = False
        v_flipped = False
        if self.do_flip:
            # h flip
            if np.random.rand() < self.h_flip_prob:
                h_flipped = True
                rgbs = [rgb[:, ::-1] for rgb in rgbs]
            # v flip
            if np.random.rand() < self.v_flip_prob:
                v_flipped = True
                rgbs = [rgb[::-1] for rgb in rgbs]
        if h_flipped:
            trajs[:, :, 0] = W_new - trajs[:, :, 0]
        if v_flipped:
            trajs[:, :, 1] = H_new - trajs[:, :, 1]

        return rgbs, trajs

    def crop(self, rgbs, trajs):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        ############ spatial transform ############

        H_new = H
        W_new = W

        # simple random crop
        y0 = (
            0
            if self.crop_size[0] >= H_new
            else np.random.randint(0, H_new - self.crop_size[0])
        )
        x0 = (
            0
            if self.crop_size[1] >= W_new
            else np.random.randint(0, W_new - self.crop_size[1])
        )
        rgbs = [
            rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            for rgb in rgbs
        ]

        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        return rgbs, trajs
