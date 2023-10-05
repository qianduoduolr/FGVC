import os.path as osp
import tempfile

import mmcv
import numpy as np
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from ...utils import *
from ..backbones import ResNet
from ..builder import build_backbone, build_components, build_loss
from ..common import (cat, images2video, masked_attention_efficient, masked_attention_efficient_v2,                     masked_attention_efficient_correlation, masked_attention_efficient_correlation_v2, 
                      non_local_attention, pil_nearest_interpolate,
                      spatial_neighbor, video2images, norm_mask, pad_divide_by, unpad, bilinear_sample, coords_grid)



from ..registry import MODELS
from .base import BaseModel


@MODELS.register_module()
class BaseTracker(BaseModel):
    """Base class for recognizers.

    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict): Classification head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self, backbone, head=None,  *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = build_backbone(backbone)
        if head is not None:
            self.head = build_components(head)
        else:
            self.head = None


        self.fp16_enabled = False
        self.register_buffer('iteration', torch.tensor(0, dtype=torch.float))

    @auto_fp16()
    def extract_feat(self, imgs):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        x = self.backbone(imgs)
        
        if self.head is not None:
            try:
                x = self.head(x)
            except Exception as e:
                return x
        return x

@MODELS.register_module()
class VanillaTracker(BaseTracker):
    """Pixel Tracker framework."""

    def __init__(self, *args, **kwargs):
        super(VanillaTracker, self).__init__(*args, **kwargs)
        self.save_np = self.test_cfg.get('save_np', False)
        self.hard_prop = self.test_cfg.get('hard_prop', False)
        self.norm_mask = self.test_cfg.get('norm_mask', True)
        self.stride_sample = self.test_cfg.get('stride_sample', False)
        

    @auto_fp16()
    def extract_feat(self, imgs):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        x = self.backbone(imgs)
        if self.stride_sample:
            x = x[:,:,::self.stride_sample,::self.stride_sample]
        
        if self.head is not None:
            try:
                x = self.head(x)
            except Exception as e:
                return x
        return x
    
    
    def extract_feat_test(self, imgs):
        outs = []
        if self.test_cfg.get('all_blocks', False):
            assert isinstance(self.backbone, ResNet)
            x = self.backbone.conv1(imgs)
            x = self.backbone.maxpool(x)
            outs = []
            for i, layer_name in enumerate(self.backbone.res_layers):
                res_layer = getattr(self.backbone, layer_name)
                if i in self.test_cfg.out_indices:
                    for block in res_layer:
                        x = block(x)
                        outs.append(x)
                else:
                    x = res_layer(x)
            return tuple(outs)
        return self.extract_feat(imgs)

    def extract_single_feat(self, imgs, idx):
        feats = self.extract_feat_test(imgs)
        if isinstance(feats, (tuple, list)):
            return feats[idx]
        else:
            return feats

    def get_feats(self, imgs, num_feats):
        assert imgs.shape[0] == 1
        batch_step = self.test_cfg.get('batch_step', 5)
        feat_bank = [[] for _ in range(num_feats)]
        clip_len = imgs.size(2)
        imgs = video2images(imgs)
        for batch_ptr in range(0, clip_len, batch_step):
            feats = self.extract_feat_test(imgs[batch_ptr:batch_ptr +
                                                batch_step])
            if isinstance(feats, tuple):
                assert len(feats) == len(feat_bank)
                for i in range(len(feats)):
                    feat_bank[i].append(feats[i].cpu())
            else:
                feat_bank[0].append(feats.cpu())
        for i in range(num_feats):
            feat_bank[i] = images2video(
                torch.cat(feat_bank[i], dim=0), clip_len)
            assert feat_bank[i].size(2) == clip_len

        return feat_bank

    def get_corrspondence(self, q, k, t=0.001, norm=True, mode='dot', mask=None, per_ref=True):

        query = self.backbone(q)
        key =  self.backbone(torch.cat(k, 0))

        bsz, c, h, w = query.shape
        key = key.view(bsz, -1, c, h, w)

        # bi(tj)
        _, corr = non_local_attention(query, key, temprature=t, norm=norm, mode=mode, mask=mask, per_ref=per_ref)

        # btij
        if not per_ref:
            corr = corr.view(bsz, corr.shape[1], -1, corr.shape[1]).permute(0, 2, 1, 3)

        return corr
    
    def img2coord(self, imgs, num_poses, topk=5):
            
        clip_len = len(imgs)
        height, width = imgs.shape[2:]
        assert imgs.shape[:2] == (clip_len, num_poses)
        coords = np.zeros((2, num_poses, clip_len), dtype=float)
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
    
    def get_coords_grid(self, shape, feat_shape):
        xx = torch.arange(0, shape[-1]).cuda()
        yy = torch.arange(0, shape[1]).cuda()
        grid = coords_grid(shape[0], xx, yy)
        stride = shape[1] // feat_shape[0] 
        
        # B x 2 x H x W
        grid_ = grid[:,:,::stride,::stride]
        
        return grid, grid_, stride
    
    def draw_gaussion_map_online(self, coord, grid, sigma=6, stride=8):
            
        B, _, H, W = grid.shape
        # B x P x 2 x H x W
        grid = grid.unsqueeze(1).repeat(coord.shape[0] // B, coord.shape[1], 1, 1, 1)
        coord = coord[:,:,:,None,None]
        
        # B x P x H x W
        g = torch.exp(-((grid[:,:,0,:,:] - coord[:,:,0])**2 + (grid[:,:,1,:,:] - coord[:,:,1])**2) / (2 * sigma**2))
        
        # a = g[0,0].detach().cpu().numpy()
        # a2 = g[0,3].detach().cpu().numpy()
        
        
        resize_g = g[:,:,::stride,::stride]
        # a3 = g[0,0].detach().cpu().numpy()
 
        return g.to(torch.float32), resize_g.to(torch.float32)
    

    def forward_train(self, imgs, labels=None):
        raise NotImplementedError
    
    def forward_test(self, rgbs, query_points, trajectories, visibilities,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        """
        _summary_

        Args:
            rgbs (_type_): B x T x C x H x W
            query_points (_type_): B x P x 3 (t,x,y)
            trajectories (_type_): B x T x P x 2
            visibilities (_type_): B x T x P
            save_image (bool, optional): _description_. Defaults to False.
            save_path (_type_, optional): _description_. Defaults to None.
            iteration (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if self.test_cfg.get("with_first", False):
            B, T, P = trajectories.shape[:3]
            
            ts = torch.unique(query_points[:,:,0])
            
            query_points_remap = torch.zeros_like(query_points).cuda()
            trajectories_remap = torch.zeros_like(trajectories).cuda()
            visibilities_remap = torch.zeros_like(visibilities).cuda()
            trajectories_pred_remap = torch.zeros_like(trajectories).cuda()
            visibilities_pred_remap = torch.zeros_like(visibilities).cuda()
            
            K = 0
            
            
            for i in range(ts.shape[0]):
                t = int(ts[i].item())
                rgbs_ = rgbs[:, t:]
                
                L = T - t
                
                # B x P x 1
                query_mask = t == query_points[:,:,0].unsqueeze(-1)
                
                # B x P' x 3
                query_points_ = torch.masked_select(query_points.clone(), query_mask).reshape(B, -1, 3)
                P_ =  query_points_.shape[1]
                
                query_points_remap[:,K:K+P_] = query_points_
                query_points_[:,:,0] -= t
                
                
                # B x T x P x 2
                query_mask_r = query_mask[:,None].repeat(1, T, 1, 2)
                trajectories_ = torch.masked_select(trajectories.clone(), query_mask_r).reshape(B, T, -1, 2)
                # B x T x P
                query_mask_r = query_mask[:,None,:,0].repeat(1, T, 1)
                visibilities_ = torch.masked_select(visibilities.clone(), query_mask_r.squeeze(-1)).reshape(B, T, -1)
                
                _, _, trajectories_pred, visibilities_pred, _  = self.forward_test_main(rgbs_, query_points_, torch.zeros(B, L, P_, 2).cuda(), torch.zeros(B, L, P_).cuda())
                
                trajectories_pred = torch.cat([torch.zeros(B, t, P_, 2).cuda(), trajectories_pred],  1)
                visibilities_pred = torch.cat([torch.zeros(B, t, P_).cuda(), visibilities_pred],  1)
                
                trajectories_remap[:,:,K:K+P_] = trajectories_
                visibilities_remap[:,:,K:K+P_] = visibilities_
                
                trajectories_pred_remap[:,:,K:K+P_] = trajectories_pred
                visibilities_pred_remap[:,:,K:K+P_] = visibilities_pred
                
                K += P_

            assert K == P
            
            return trajectories_remap, visibilities_remap, trajectories_pred_remap, visibilities_pred_remap, query_points_remap
                
        
        else:
            return self.forward_test_main(rgbs, query_points, trajectories, visibilities)

    def forward_test_main(self, rgbs, query_points, trajectories, visibilities):
        """Defines the computation performed at every call when evaluation and
        testing."""
        imgs = rgbs.transpose(1,2)
        
        B,  C, clip_len, h, w = imgs.shape
        
        clip_len = imgs.size(2)
        
        # get target shape
        dummy_feat = self.extract_feat_test(imgs[0:1, :, 0])
        feat_shape = dummy_feat.shape

        feat_bank = self.get_feats(imgs, len(dummy_feat))[0]
        
        
        grid, grid_, stride = self.get_coords_grid((B,h,w), feat_shape[-2:])
        ref_seg_map, resized_seg_map = self.draw_gaussion_map_online(query_points[:,:,1:], grid, stride=stride)
        
        
        C = resized_seg_map.shape[1]    
            
        seg_bank = []

        seg_preds = [ref_seg_map.detach()]
        neighbor_range = self.test_cfg.get('neighbor_range', None)
        
        if neighbor_range is not None and self.test_cfg.get('test_mode','v1') == 'v1':
            mode = self.test_cfg.get('mask_mode', 'circle')
            spatial_neighbor_mask = spatial_neighbor(
                feat_shape[0],
                *feat_shape[2:],
                neighbor_range=neighbor_range,
                device=imgs.device,
                dtype=imgs.dtype,
                mode=mode)
        else:
            spatial_neighbor_mask = None

        seg_bank.append(resized_seg_map.detach())
        for frame_idx in range(1, clip_len):
            key_start = max(0, frame_idx - self.test_cfg.precede_frames)
            query_feat = feat_bank[:, :,
                                                frame_idx].to(imgs.device)
            key_feat = feat_bank[:, :, key_start:frame_idx].to(
                imgs.device)
            value_logits = torch.stack(
                seg_bank[key_start:frame_idx], dim=2).to(imgs.device)
            if self.test_cfg.get('with_first', True):
                key_feat = torch.cat([
                    feat_bank[:, :, 0:1].to(imgs.device),
                    key_feat
                ],
                                        dim=2)
                value_logits = cat([
                    seg_bank[0].unsqueeze(2).to(imgs.device), value_logits
                ],
                                    dim=2)
            
            
            if self.test_cfg.get('test_mode','v1') == 'v1':
                seg_logit = masked_attention_efficient(
                    query_feat,
                    key_feat,
                    value_logits,
                    spatial_neighbor_mask,
                    temperature=self.test_cfg.temperature,
                    topk=self.test_cfg.topk,
                    step=self.test_cfg.get('step', 32),
                    normalize=self.test_cfg.get('with_norm', True),
                    non_mask_len=0 if self.test_cfg.get(
                        'with_first_neighbor', True) else 1,
                    sim_mode=self.test_cfg.get(
                        'sim_mode', 'dot_product'))
            else:
                seg_logit = masked_attention_efficient_v2(
                    query_feat,
                    key_feat,
                    value_logits,
                    neighbor_range//2,
                    temperature=self.test_cfg.temperature,
                    topk=self.test_cfg.topk,
                    step=self.test_cfg.get('step', 32),
                    normalize=self.test_cfg.get('with_norm', True),
                    non_mask_len=0 if self.test_cfg.get(
                        'with_first_neighbor', True) else 1,
                    sim_mode=self.test_cfg.get(
                        'sim_mode', 'dot_product'))
            
            seg_bank.append(seg_logit)

            seg_pred = F.interpolate(
                seg_logit,
                size=(h,w),
                mode='bilinear',
                align_corners=False)
            
            seg_preds.append(seg_pred.detach())

        seg_preds = torch.stack(seg_preds, 1).cpu().numpy()

        trajectories_pred = self.img2coord(seg_preds[0], num_poses=C)
        trajectories_pred = torch.from_numpy(trajectories_pred).permute(2,1,0).unsqueeze(0).cuda()
        
        visibilities_pred = torch.zeros_like(visibilities).cuda()
            
            
        return trajectories, visibilities, trajectories_pred, visibilities_pred, query_points

      


@MODELS.register_module()
class HRVanillaTracker(VanillaTracker):
    def __init__(self, stride=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from mmcv.ops import Correlation
        
        self.stride = stride
        
        self.infer_radius = self.test_cfg.get('neighbor_range', 24) // 2
        self.infer_dilations = self.test_cfg.get('dilations', 1)
        self.grid_size_hr = 2 * self.infer_radius + 1
        self.corr_infer = Correlation(max_displacement=self.infer_radius, dilation=self.infer_dilations)
        
        self.infer_mode = self.test_cfg.get('infer_mode', 'backward')
        self.is_dense = self.test_cfg.get('is_dense', True)
        self.save_mem = self.test_cfg.get('save_mem', False)
    
    
    def get_correspondence(self, query_feat, key_feats):
        
        if self.test_cfg.get('withnorm', True):
            query_feat = F.normalize(query_feat, dim=1)
            key_feats = F.normalize(key_feats, dim=1)
        
        aff = self.corr_infer(query_feat, key_feats).flatten(1,2)
        
        return aff
    
    def get_coord(self, query_feat, key_feats, shape, scale):
        
        ori_H, ori_W = shape
        
        # B x R^2 x H x W
        aff = self.get_correspondence(query_feat, key_feats)
        
        # print(aff.max(), aff.min())
        
        B = aff.shape[0]
        H, W = aff.shape[-2:]
        
        xx = torch.arange(0, ori_W, device=aff.device)
        yy = torch.arange(0, ori_H, device=aff.device)
        
        # B x 2 x h x w
        grid = coords_grid(B, xx, yy) 
        
        
        grid_sample = grid[:, :, ::scale, ::scale]
        
        # B x 2 x R^2 x H x W
        grid_unfold = F.unfold(grid_sample, kernel_size=(self.grid_size_hr, self.grid_size_hr), padding=self.infer_radius).reshape(B, 2, -1, H, W)
        
        # 1 x (K x R^2) x H x W
        aff = aff.unsqueeze(0).flatten(1,2)
        
        # 1 x 2 x (K x R^2) x H x W
        grid_unfold = grid_unfold.unsqueeze(0).transpose(1,2).flatten(2,3)
        
        # 1 x topk x H x W
        topk_affinity, topk_indices = aff.topk(k=self.test_cfg.get('topk', 10), dim=1)
        
        # 1 x 2 x topk x H x W
        topk_grid = torch.gather(grid_unfold, dim=2, index=topk_indices.unsqueeze(1).repeat(1, 2, 1, 1, 1))
        
        topk_affinity /= self.test_cfg.get('temperature', 1)
        topk_affinity = topk_affinity.softmax(dim=1)

        # 1 x 2 x H x W
        coord = torch.einsum('bckhw,bkhw->bchw', topk_grid,
                                topk_affinity)
                
        return coord
    
            
    
    def forward_test_main(self, rgbs, query_points, trajectories, visibilities):
        """backward warpping"""
        imgs = rgbs.transpose(1,2)
        
        B,  C, clip_len, h, w = imgs.shape
        
        clip_len = imgs.size(2)
        
        # get target shape
        dummy_feat = self.extract_feat_test(imgs[0:1, :, 0])
        feat_shape = dummy_feat.shape

        feat_bank = self.get_feats(imgs, len(dummy_feat))[0]
        
        
        grid, grid_, stride = self.get_coords_grid((B,h,w), feat_shape[-2:])
        ref_seg_map, resized_seg_map = self.draw_gaussion_map_online(query_points[:,:,1:], grid, stride=stride)
        
        C = resized_seg_map.shape[1]
        
        seg_bank = []
        seg_preds = [ref_seg_map.detach()]
        seg_bank.append(resized_seg_map.cpu())
        
        for frame_idx in range(1, clip_len):
            
            key_start = max(0, frame_idx - self.test_cfg.precede_frames)
            
            # B x C x T x Hf x Wf
            value_logits = torch.stack(
                seg_bank[key_start:frame_idx], dim=2).to(imgs.device)

            if not self.save_mem:
                # B x C x Hf x Wf
                query_feat = feat_bank[:, :, frame_idx].to(imgs.device)
                
                # B x C x T x Hf x Wf
                key_feat = feat_bank[:, :, key_start:frame_idx].to(
                    imgs.device)
                
                if self.test_cfg.get('with_first', True):
                    key_feat = torch.cat([feat_bank[:, :, 0:1].to(imgs.device),key_feat], dim=2)
                    value_logits = cat([seg_bank[0].unsqueeze(2).to(imgs.device), value_logits],dim=2)
                
            else:
                # B x C x Hf x Wf
                query_feat = self.extract_feat(imgs[0:1, :, frame_idx])
                
                # B x C x T x Hf x Wf
                key_feat = self.extract_feat(imgs[0:1, :, key_start]).unsqueeze(2)

        
            mem_len = key_feat.shape[2]
            
            # K x Rf^2 x Hf x Wf
            corr_up = self.get_correspondence(query_feat.repeat(mem_len, 1, 1, 1), key_feat[0,:].transpose(0,1))
            
            # K x C x R^2 x Hf x Wf
            unfold_v = F.unfold(value_logits[0,:].transpose(0,1), kernel_size=(self.grid_size_hr, self.grid_size_hr), padding=self.infer_radius).reshape(mem_len, C, -1, *feat_shape[-2:])
            
            # 1 x (K x R^2) x Hf x Wf
            corr_up = corr_up.unsqueeze(0).flatten(1,2)
            # 1 x C x (K x R^2) x Hf x Wf
            unfold_v = unfold_v.unsqueeze(0).transpose(1,2).flatten(2,3)
            
            # 1 x topk x Hf x Wf
            topk_affinity, topk_indices = corr_up.topk(k=self.test_cfg.get('topk', 10), dim=1)
            
            # 1 x C x topk x Hf x Wf
            topk_value = torch.gather(unfold_v, dim=2, index=topk_indices.unsqueeze(1).repeat(1, C, 1, 1, 1))
            
            topk_affinity /= self.test_cfg.get('temperature', 1)
            topk_affinity = topk_affinity.softmax(dim=1)

            seg_logit = torch.einsum('bckhw,bkhw->bchw', topk_value, topk_affinity)
            seg_bank.append(seg_logit.cpu())
            
            seg_pred = F.interpolate(
                seg_logit,
                size=(h,w),
                mode='bilinear',
                align_corners=False)
            
            seg_preds.append(seg_pred.detach())

        seg_preds = torch.stack(seg_preds, 1).cpu().numpy()

        trajectories_pred = self.img2coord(seg_preds[0], num_poses=C)
        trajectories_pred = torch.from_numpy(trajectories_pred).permute(2,1,0).unsqueeze(0).cuda()
        
        visibilities_pred = torch.zeros_like(visibilities).cuda()
            
            
        return trajectories, visibilities, trajectories_pred, visibilities_pred, query_points
               
                


    
    def forward_test_forward(self, imgs, ref_seg_map, img_meta, ref,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        """forward warpping."""
        h, w = imgs.shape[-2:]

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        clip_len = imgs.size(2)
        # get target shape
        dummy_feat = self.extract_feat(imgs[0:1, :, 0])
        
        feat_shape = dummy_feat.shape
            
        all_seg_preds = []
        
        if not self.save_mem:
            feat_bank = self.get_feats(imgs, len(dummy_feat))
        
        
        scale = w // feat_shape[-1]
        
        ref = torch.flip(ref, (1,))
        coords = [ref]
        coord = ref.clone()
    
        for frame_idx in range(1, clip_len):
            
            start = max(0, frame_idx - self.test_cfg.precede_frames)
            
            if not self.save_mem:
                # B x C x Hf x Wf
                key_feat = feat_bank[0][:, :,frame_idx].to(imgs.device)
                
                # B x C x Hf x Wf
                query_feat = feat_bank[0][:, :, start].to(
                imgs.device)
            else:
                # B x C x Hf x Wf
                query_feat = self.extract_feat(imgs[0:1, :, start])
                
                # B x C x Hf x Wf
                key_feat = self.extract_feat(imgs[0:1, :, frame_idx])
                            

        
            # 1 x 2 x h x w
            coord_candi = self.get_coord(query_feat, key_feat, shape=(h,w), scale=scale)
            
            coord = bilinear_sample(coord_candi, coord.clone().unsqueeze(-1) / scale, align_corners=True).squeeze(-1)
            coords.append(coord)
                

        coords = torch.stack(coords, -1).cpu().numpy()

        all_seg_preds.append(coords.astype(float))


        if self.test_cfg.get('save_np', False):
            if len(all_seg_preds) > 1:
                return [all_seg_preds]
            else:
                return [all_seg_preds[0]]
        else:
            if len(all_seg_preds) > 1:
                all_seg_preds = np.stack(all_seg_preds, axis=1)
            else:
                all_seg_preds = all_seg_preds[0]
            # unravel batch dim
            return list(all_seg_preds)
        
        
    def forward_test_backward_save_mem(self, imgs, ref_seg_map, img_meta,
                    ref=None,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        """backward warpping"""
        h, w = imgs.shape[-2:]
        
        imgs, _ = pad_divide_by(imgs, d=self.stride)
        ref_seg_map, pa = pad_divide_by(ref_seg_map, d=self.stride)

        pad_shape = ref_seg_map.shape[-2:]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        clip_len = imgs.size(2)
        # get target shape
        dummy_feat = self.extract_feat(imgs[0:1, :, 0])
        
        H, W = dummy_feat.shape[-2:]    
        
        
        if isinstance(dummy_feat, (list, tuple)):
            feat_shapes = [_.shape for _ in dummy_feat]
        else:
            feat_shapes = [dummy_feat.shape]
            
        all_seg_preds = []
    
        # feat_bank = self.get_feats(imgs, len(dummy_feat))
        
        for feat_idx, feat_shape in enumerate(feat_shapes):
            
            input_onehot = ref_seg_map.ndim == 4
            if not input_onehot:
                shape_ = feat_shape[2:]
                resized_seg_map = pil_nearest_interpolate(
                    ref_seg_map.unsqueeze(1),
                    size=shape_).squeeze(1).long()
                resized_seg_map = F.one_hot(resized_seg_map).permute(
                    0, 3, 1, 2).float()
                ref_seg_map = F.interpolate(
                    unpad(ref_seg_map, pa).unsqueeze(1).float(),
                    size=img_meta[0]['original_shape'][:2],
                    mode='nearest').squeeze(1)
            else:
                shape_ = feat_shape[2:]
                resized_seg_map = F.interpolate(
                    ref_seg_map,
                    size=shape_,
                    mode='bilinear',
                    align_corners=False).float()
                ref_seg_map = F.interpolate(
                    ref_seg_map,
                    size=img_meta[0]['original_shape'][:2],
                    mode='bilinear',
                    align_corners=False)
            seg_bank = []

            seg_preds = [ref_seg_map.detach().cpu().numpy()]
            
            C = resized_seg_map.shape[1]
            

            seg_bank.append(resized_seg_map.cpu())
            for frame_idx in tqdm.tqdm(range(1, clip_len), total=clip_len-1):
                
                key_start = max(0, frame_idx - self.test_cfg.precede_frames)
                
                # B x C x T x Hf x Wf
                value_logits = torch.stack(
                    seg_bank[key_start:frame_idx], dim=2).to(imgs.device)


                # B x C x Hf x Wf
                query_frame = imgs[:, :,frame_idx].to(imgs.device)
                
                # B x C x T x Hf x Wf
                key_frames = imgs[:, :, key_start:frame_idx].to(
                    imgs.device)
                
                if self.test_cfg.get('with_first', True):
                    key_frames = torch.cat([imgs[:, :, 0:1].to(imgs.device),key_frames], dim=2)
                    value_logits = cat([seg_bank[0].unsqueeze(2).to(imgs.device), value_logits],dim=2)
            
                
                seg_logit = masked_attention_efficient_correlation(
                    query_frame,
                    key_frames,
                    value_logits,
                    radius=self.infer_radius,
                    corr_infer=self.corr_infer,
                    feat_extractor=self.backbone,
                    temperature=self.test_cfg.temperature,
                    topk=self.test_cfg.topk,
                    sstep=self.test_cfg.get('sstep', 32),
                    tstep=self.test_cfg.get('tstep', 5),
                    normalize=self.test_cfg.get('with_norm', True),
                    )
                   

                if not self.hard_prop:
                    seg_bank.append(seg_logit.cpu())
                else:
                    seg_logit_hard = seg_logit.argmax(1,keepdim=True)
                    seg_logit_hard = F.one_hot(seg_logit_hard)[:,0].permute(0,3,1,2).float()
                    seg_bank.append(seg_logit_hard.cpu())

                # resize to pad_size
                seg_pred = F.interpolate(
                    seg_logit,
                    size=pad_shape,
                    mode='bilinear',
                    align_corners=False)
                # unpad, now it has same size with input mask
                seg_pred = unpad(seg_pred, pa)
                
                # resize to final ori size for
                seg_pred = F.interpolate(
                    seg_pred,
                    size=img_meta[0]['original_shape'][:2],
                    mode='bilinear',
                    align_corners=False)
                
                if not input_onehot:
                    seg_pred_min = seg_pred.view(*seg_pred.shape[:2], -1).min(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    seg_pred_max = seg_pred.view(*seg_pred.shape[:2], -1).max(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    normalized_seg_pred = (seg_pred - seg_pred_min) / (
                        seg_pred_max - seg_pred_min + 1e-12)
                    seg_pred = torch.where(seg_pred_max > 0,
                                           normalized_seg_pred, seg_pred)
                    seg_pred = seg_pred.argmax(dim=1)
                    seg_pred = F.interpolate(
                        seg_pred.float().unsqueeze(1),
                        size=img_meta[0]['original_shape'][:2],
                        mode='nearest').squeeze(1)
                
                seg_preds.append(seg_pred.detach().cpu().numpy())
                

            seg_preds = np.stack(seg_preds, axis=1)
            if self.save_np:
                assert seg_preds.shape[0] == 1
                eval_dir = save_path
                mmcv.mkdir_or_exist(eval_dir)
                temp_file = tempfile.NamedTemporaryFile(
                    dir=eval_dir, suffix='.npy', delete=False)
                file_path = osp.join(eval_dir, temp_file.name)
                np.save(file_path, seg_preds[0])
                all_seg_preds.append(file_path)
            else:
                if not self.test_cfg.get('coords', False):
                    all_seg_preds.append(seg_preds)
                else:
                    coord_preds = self.img2coord(seg_preds[0], num_poses=C)
                    all_seg_preds.append(coord_preds[None])

        if self.save_np:
            if len(all_seg_preds) > 1:
                return [all_seg_preds]
            else:
                return [all_seg_preds[0]]
        else:
            if len(all_seg_preds) > 1:
                all_seg_preds = np.stack(all_seg_preds, axis=1)
            else:
                all_seg_preds = all_seg_preds[0]
            # unravel batch dim
            return list(all_seg_preds)