# Copyright (c) OpenMMLab. All rights reserved.
from lib2to3.pytree import Base
from typing import Dict, Optional, Sequence, Tuple

import mmcv
import torch
import tempfile

import torch.nn as nn
from numpy import ndarray
import numpy as np
import os.path as osp

import torch.nn.functional as F
from tqdm import tqdm
from mmpt.models.common import (images2video, masked_attention_efficient,
                               non_local_attention, pil_nearest_interpolate,
                               spatial_neighbor, video2images, bilinear_sample)
from mmpt.models.common.occlusion_estimation import *
from mmpt.datasets.flyingthingsplus.utils import samp

from ..builder import MODELS, build_backbone, build_components, build_loss, build_operators
from .base import BaseModel


@MODELS.register_module()
class RAFT(BaseModel):
    """RAFT model.
    Args:
        num_levels (int): Number of levels in .
        radius (int): Number of radius in  .
        cxt_channels (int): Number of channels of context feature.
        h_channels (int): Number of channels of hidden feature in .
        cxt_encoder (dict): Config dict for building context encoder.
        freeze_bn (bool, optional): Whether to freeze batchnorm layer or not.
            Default: False.
    """

    def __init__(self,
                 backbone, 
                 decoder,
                 cxt_backbone,
                 loss,
                 num_levels: int,
                 radius: int,
                 cxt_channels: int,
                 h_channels: int,
                 freeze_bn: bool = False,
                 warp_op_cfg = None,
                 pretrained = None,
                 flow_clamp = -1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = build_backbone(backbone)
        self.decoder = build_components(decoder)

        self.num_levels = num_levels
        self.radius = radius
        self.context = build_backbone(cxt_backbone)
        self.h_channels = h_channels
        self.cxt_channels = cxt_channels
        self.warp = build_operators(warp_op_cfg) if warp_op_cfg is not None else None
        self.flow_clamp = flow_clamp

        self.loss = build_loss(loss)

        assert self.num_levels == self.decoder.num_levels
        assert self.radius == self.decoder.radius
        assert self.h_channels == self.decoder.h_channels
        assert self.cxt_channels == self.decoder.cxt_channels
        assert self.h_channels + self.cxt_channels == self.context.out_channels

        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def img2coord(self, imgs, num_poses, topk=5):
        
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
    
    
    def extract_feat(
        self, imgs: torch.Tensor
    ):
    
        """Extract features from images.
        Args:
            imgs (Tensor): The concatenated input images.
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The feature from the first
                image, the feature from the second image, the hidden state
                feature for GRU cell and the contextual feature.
        """
        bsz, p, c, t, h, w = imgs.shape
        img1 = imgs[:, 0, :, 0, ...]
        img2 = imgs[:, 0, :, 1, ...]

        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        cxt_feat = self.context(img1)

        h_feat, cxt_feat = torch.split(
            cxt_feat, [self.h_channels, self.cxt_channels], dim=1)
        h_feat = torch.tanh(h_feat)
        cxt_feat = torch.relu(cxt_feat)

        return feat1, feat2, h_feat, cxt_feat

    def forward_train(
            self,
            imgs: torch.Tensor,
            flows: torch.Tensor,
            valid: torch.Tensor = None,
            flow_init: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None
    ):
        """Forward function for RAFT when model training.
        Args:
            imgs (Tensor): The concatenated input images.
            flow_gt (Tensor): The ground truth of optical flow.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.
        Returns:
            Dict[str, Tensor]: The losses of output.
        """

        feat1, feat2, h_feat, cxt_feat = self.extract_feat(imgs)
        B, _, H, W = feat1.shape

        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        pred, _, _ = self.decoder(
            False,
            feat1,
            feat2,
            flow=flow_init,
            h_feat=h_feat,
            cxt_feat=cxt_feat,
            valid=valid,
            )

        losses = {}

        losses['flow_loss'] = self.loss(pred, flows[:,0,:,0])

        return losses        

    def forward_test_pair(
            self,
            imgs: torch.Tensor,
            flow_init: Optional[torch.Tensor] = None,
            return_lr: bool = False,
            img_metas: Optional[Sequence[dict]] = None):
        """Forward function for RAFT when model testing.
        Args:
            imgs (Tensor): The concatenated input images.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.
        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted optical flow
                with the same size of images after augmentation.
        """
        train_iter = self.decoder.iters
        
        if self.test_cfg is not None and self.test_cfg.get(
                'iters') is not None:
            self.decoder.iters = self.test_cfg.get('iters')

        feat1, feat2, h_feat, cxt_feat = self.extract_feat(imgs)
        B, _, H, W = feat1.shape

        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        results = self.decoder(
            test_mode=True,
            feat1=feat1,
            feat2=feat2,
            flow=flow_init,
            h_feat=h_feat,
            cxt_feat=cxt_feat,
            img_metas=img_metas,
            return_lr=return_lr
            )
        # recover iter in train
        self.decoder.iters = train_iter

        return results
    
        
        
    def forward_test(self, rgbs, query_points, trajectories, visibilities,
        save_image=False,
        save_path=None,
        iteration=None):
        
        
        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = query_points.shape[1]


        flows_forward = []
        flows_backward = []
        for t in range(1, n_frames):
            rgb0 = rgbs[:, t - 1]
            rgb1 = rgbs[:, t]
            
            flows_forward.append(self.forward_test_pair(torch.stack([rgb0[:,None], rgb1[:,None]], 3)
            )[0][-1])
            flows_backward.append(self.forward_test_pair(torch.stack([rgb1[:,None], rgb0[:,None]], 3)
            )[0][-1])
            
        flows_forward = torch.stack(flows_forward, dim=1)
        flows_backward = torch.stack(flows_backward, dim=1)
        assert flows_forward.shape == flows_backward.shape == (batch_size, n_frames - 1, 2, height, width)

        coords = []
        for t in range(n_frames):
            if t == 0:
                coord = torch.zeros_like(query_points[:, :, 1:])
            else:
                prev_coord = coords[t - 1]
                delta = samp.bilinear_sample2d(
                    im=flows_forward[:, t - 1],
                    x=prev_coord[:, :, 0],
                    y=prev_coord[:, :, 1],
                ).permute(0, 2, 1)
                assert delta.shape == (batch_size, n_points, 2), "Forward flow at the discrete points"
                coord = prev_coord + delta

            # Set the ground truth query point location if the timestep is correct
            query_point_mask = query_points[:, :, 0] == t
            coord = coord * ~query_point_mask.unsqueeze(-1) + query_points[:, :, 1:] * query_point_mask.unsqueeze(-1)

            coords.append(coord)

        for t in range(n_frames - 2, -1, -1):
            coord = coords[t]
            successor_coord = coords[t + 1]

            delta = samp.bilinear_sample2d(
                im=flows_backward[:, t],
                x=successor_coord[:, :, 0],
                y=successor_coord[:, :, 1],
            ).permute(0, 2, 1)
            assert delta.shape == (batch_size, n_points, 2), "Backward flow at the discrete points"

            # Update only the points that are located prior to the query point
            prior_to_query_point_mask = t < query_points[:, :, 0]
            coord = (coord * ~prior_to_query_point_mask.unsqueeze(-1) +
                     (successor_coord + delta) * prior_to_query_point_mask.unsqueeze(-1))
            coords[t] = coord

        trajectories_pred = torch.stack(coords, dim=1)
        visibilities_pred = (trajectories_pred[:, :, :, 0] >= 0) & \
                       (trajectories_pred[:, :, :, 1] >= 0) & \
                       (trajectories_pred[:, :, :, 0] < width) & \
                       (trajectories_pred[:, :, :, 1] < height)
        return trajectories, visibilities, trajectories_pred, visibilities_pred, query_points