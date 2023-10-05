# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmpt.models.common.embedding import get_3d_embedding, get_1d_sincos_pos_embed_from_grid, get_2d_embedding
from mmpt.models.common.sampling import sample_pos_embed
from mmpt.utils import tensor2img
from einops import rearrange

from .flow_decorder import *

from ...builder import build_operators, build_components

from ...registry import COMPONENTS
from .base_decoder import BaseDecoder
    

class CorrelationPyramid(BaseModule):
    """Pyramid Correlation Module.
    The neck of RAFT-Net, which calculates correlation tensor of input features
    with the method of 4D Correlation Pyramid mentioned in RAFT-Net.
    Args:
        num_levels (int): Number of levels in the module.
            Default: 4.
    """

    def __init__(self, num_levels: int = 4, norm=False, scaling=True, temp=1):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.num_levels = num_levels
        self.norm = norm
        self.scaling = scaling
        self.temp = temp

    def forward(self, query, feats):
        """Forward function for Correlation pyramid.
        Args:
            feat1 (Tensor): The feature from first input image.
            feat2 (Tensor): The feature from second input image.
        Returns:
            Sequence[Tensor]: The list of correlation which is pooled using
                average pooling with kernel sizes {1, 2, 4, 8}.
        """
        B, T, C, H, W = feats.shape
        P = query.shape[2]

        if self.norm:
            query = F.normalize(query, dim=-1)
            feats = F.normalize(feats, dim=2)
        
        corr = torch.matmul(
            query,
            feats.view(B, T, C, -1)).view(B, T, P, H, W) / self.temp
        
        corr = corr.reshape(B*T*P, 1, H, W)
        
        if self.scaling:
            corr = corr / torch.sqrt(
            torch.tensor(C).float())
        
        corr_pyramid = [corr]
        for _ in range(self.num_levels - 1):
            
            _corr = self.pool(corr_pyramid[-1])
            corr_pyramid.append(_corr)

        return corr_pyramid



@COMPONENTS.register_module()
class TRAJ_PyramidDecoder(BaseModule):
    """The decoder of RAFT Net.
    The decoder of RAFT Net, which outputs list of upsampled flow estimation.
    Args:
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
        num_levels (int): Number of levels used when calculating
            correlation tensor.
        radius (int): Radius used when calculating correlation tensor.
        iters (int): Total iteration number of iterative update of RAFTDecoder.
        corr_op_cfg (dict): Config dict of correlation operator.
            Default: dict(type='CorrLookup').
        gru_type (str): Type of the GRU module. Choices: ['Conv', 'SeqConv'].
            Default: 'SeqConv'.
        feat_channels (Sequence(int)): features channels of prediction module.
        mask_channels (int): Output channels of mask prediction layer.
            Default: 64.
        conv_cfg (dict, optional): Config dict of convolution layers in motion
            encoder. Default: None.
        norm_cfg (dict, optional): Config dict of norm layer in motion encoder.
            Default: None.
        act_cfg (dict, optional): Config dict of activation layer in motion
            encoder. Default: None.
    """

    def __init__(
        self,
        traj_pred, 
        radius = 3,
        iters = 6,
        input_dim = 128,
        time_dim = 64, 
        stride = 8, 
        use_update_feat = True,
        corr_block_config = dict(num_levels=4, norm=False, scaling=True, temp=1),
        norm = False,
        scaling = True,
        temp = 1,
        corr_op_cfg: dict = dict(type='CorrLookup', align_corners=True),
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.corr_block_config = corr_block_config
        self.corr_block = CorrelationPyramid(**corr_block_config)

        # self.num_levels = num_levels
        self.radius = radius

        self.iters = iters
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.stride = stride
        self.use_update_feat = use_update_feat

        corr_op_cfg['radius'] = radius
        self.corr_lookup = build_operators(corr_op_cfg)

        traj_pred['update_feat'] = use_update_feat
        self.traj_pred = build_components(traj_pred)
        
        if self.use_update_feat:
            self.ffeat_updater = nn.Sequential(
                nn.GroupNorm(1, input_dim),
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
            )



    def forward(
                self, 
                feats,
                coords,
                query_feat,
                vis=False
                ):
        """Forward function for RAFTDecoder.
        Args:
            feat1 (Tensor): The feature from the first input image.
            feat2 (Tensor): The feature from the second input image.
            flow (Tensor): The initialized flow when warm start.
            h (Tensor): The hidden state for GRU cell.
            cxt_feat (Tensor): The contextual feature from the first image.
        Returns:
            Sequence[Tensor]: The list of predicted optical flow.
        """

            
        _, T, C, H, W = feats.shape
        
        B, T, P, _ = coords.shape
        
        # B x T x P x C        
        query_feat_init = query_feat.clone()
        
        preds = []
        
        if vis:
            vis_results = {}
            vis_results['vis_crop_corrs'] = []
            vis_results['vis_coords'] = []
            point_id = random.randint(0, P-1)

        for _ in range(self.iters):
            coords = coords.detach()
            
            corr_pyramid = self.corr_block(query_feat, feats)
            cur_corrs = self.corr_lookup(corr_pyramid, coords)
            
              # B x T x P x 2
            coords_ = coords - coords[:,0:1]
            
            # B x T x P x 1
            times_ = torch.linspace(0, T, T).cuda().reshape(1, T, 1, 1).repeat(B, 1, P, 1)
            
            t_ = torch.cat([coords_, times_], dim=-1).transpose(1,2).flatten(0,1)
            time_emb = get_3d_embedding(t_, self.time_dim, cat_coords=True).reshape(B,P,T,-1).transpose(1,2)
            x = torch.cat([cur_corrs, time_emb, query_feat], dim=-1) 
            
            # B T P x C
            delta_ = self.traj_pred(x)
            
            if self.use_update_feat:
                delta_feats_ = delta_[:,:,:,:-2].flatten(0,2)
                delta_feats_ = self.ffeat_updater(delta_feats_).reshape(B, T, P, C)
                query_feat = delta_feats_ + query_feat
            
            coords = coords + delta_[:,:,:,-2:]
            
            preds.append(coords * self.stride)
            
            if vis:
                crop_corrs = cur_corrs[0,:,point_id,:49].reshape(T,1,7,7).transpose(2,3).detach().abs().cpu()
                crop_corrs = tensor2img(crop_corrs, norm_mode='0-1')
                vis_results['vis_crop_corrs'].append(crop_corrs)
                vis_results['vis_coords'].append(coords[0, :, point_id] * self.stride)
        
                if _ == self.iters - 1:
                    vis_results['vis_corrs'] = tensor2img(corr_pyramid[0].reshape(B, T, P, 1, H, W)[0, :, point_id], norm_mode='0-1')

        if not vis:
            return preds, query_feat_init, query_feat, None
        else:
            return preds, query_feat_init, query_feat, vis_results




@COMPONENTS.register_module()
class TRAJ_PyramidDecoderV2(TRAJ_PyramidDecoder):
    """The decoder of RAFT Net.
    The decoder of RAFT Net, which outputs list of upsampled flow estimation.
    Args:
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
        num_levels (int): Number of levels used when calculating
            correlation tensor.
        radius (int): Radius used when calculating correlation tensor.
        iters (int): Total iteration number of iterative update of RAFTDecoder.
        corr_op_cfg (dict): Config dict of correlation operator.
            Default: dict(type='CorrLookup').
        gru_type (str): Type of the GRU module. Choices: ['Conv', 'SeqConv'].
            Default: 'SeqConv'.
        feat_channels (Sequence(int)): features channels of prediction module.
        mask_channels (int): Output channels of mask prediction layer.
            Default: 64.
        conv_cfg (dict, optional): Config dict of convolution layers in motion
            encoder. Default: None.
        norm_cfg (dict, optional): Config dict of norm layer in motion encoder.
            Default: None.
        act_cfg (dict, optional): Config dict of activation layer in motion
            encoder. Default: None.
    """
    
    
    def forward(
                self, 
                feats,
                coords,
                query_feat,
                vis_init,
                track_mask,
                vis=False
                ):
        """Forward function for RAFTDecoder.
        Args:
            feat1 (Tensor): The feature from the first input image.
            feat2 (Tensor): The feature from the second input image.
            flow (Tensor): The initialized flow when warm start.
            h (Tensor): The hidden state for GRU cell.
            cxt_feat (Tensor): The contextual feature from the first image.
        Returns:
            Sequence[Tensor]: The list of predicted optical flow.
        """
            
        _, T, C, H, W = feats.shape
        
        B, T, P, _ = coords.shape
        
        # B x T x P x C        
        query_feat_init = query_feat.clone()
        coords = coords.clone()
        
        preds = []
        
        if vis:
            vis_results = {}
            vis_results['vis_crop_corrs'] = []
            vis_results['vis_coords'] = []
            point_id = random.randint(0, P-1)
            
        times_ = torch.linspace(0, T - 1, T).reshape(1, T, 1)

        pos_embed = sample_pos_embed(
            grid_size=(H, W),
            embed_dim=456,
            coords=coords,
        )
        
        pos_embed = rearrange(pos_embed, "b e n -> (b n) e").unsqueeze(1)
        times_embed = (
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(456, times_[0]))[None]
            .repeat(B, 1, 1)
            .float()
            .cuda()
        )

        for _ in range(self.iters):
            coords = coords.detach()
            
            corr_pyramid = self.corr_block(query_feat, feats)
            cur_corrs = self.corr_lookup(corr_pyramid, coords)
            
            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * P, T, 2)
            flows_cat = get_2d_embedding(flows_, 64, cat_coords=True)
            
            
            if track_mask.shape[1] < vis_init.shape[1]:
                track_mask = torch.cat(
                    [
                        track_mask,
                        torch.zeros_like(track_mask[:, 0]).repeat(
                            1, vis_init.shape[1] - track_mask.shape[1], 1, 1
                        ),
                    ],
                    dim=1,
                )
                
            concat = (
                torch.cat([track_mask, vis_init], dim=2)
                .permute(0, 2, 1, 3)
                .reshape(B * P, T, 2)
            )
            
            # concat = torch.ones((B * P, T, 2)).cuda()
            
            x = torch.cat([flows_cat, cur_corrs.permute(0, 2, 1, 3).flatten(0,1), query_feat.permute(0, 2, 1, 3).flatten(0,1), concat], dim=2)
            x = x + pos_embed + times_embed
            x = rearrange(x, "(b n) t d -> b n t d", b=B).permute(0, 2, 1, 3)
            
            # B x T x P x C
            delta_ = self.traj_pred(x)
            
            if self.use_update_feat:
                delta_feats_ = delta_[:,:,:,:-2].flatten(0,2)
                delta_feats_ = self.ffeat_updater(delta_feats_).reshape(B, T, P, C)
                query_feat = delta_feats_ + query_feat
            
            coords = coords + delta_[:,:,:,-2:]
            
            preds.append(coords * self.stride)
            
            if vis:
                crop_corrs = cur_corrs[0,:,point_id,:49].reshape(T,1,7,7).transpose(2,3).detach().abs().cpu()
                crop_corrs = tensor2img(crop_corrs, norm_mode='0-1')
                vis_results['vis_crop_corrs'].append(crop_corrs)
                vis_results['vis_coords'].append(coords[0, :, point_id] * self.stride)
        
                if _ == self.iters - 1:
                    vis_results['vis_corrs'] = tensor2img(corr_pyramid[0].reshape(B, T, P, 1, H, W)[0, :, point_id], norm_mode='0-1')

        if not vis:
            return preds, query_feat_init, query_feat, None
        else:
            return preds, query_feat_init, query_feat, vis_results
        
        


@COMPONENTS.register_module()
class TRAJ_PyramidDecoderV3(TRAJ_PyramidDecoder):
    """The decoder of RAFT Net.
    The decoder of RAFT Net, which outputs list of upsampled flow estimation.
    Args:
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
        num_levels (int): Number of levels used when calculating
            correlation tensor.
        radius (int): Radius used when calculating correlation tensor.
        iters (int): Total iteration number of iterative update of RAFTDecoder.
        corr_op_cfg (dict): Config dict of correlation operator.
            Default: dict(type='CorrLookup').
        gru_type (str): Type of the GRU module. Choices: ['Conv', 'SeqConv'].
            Default: 'SeqConv'.
        feat_channels (Sequence(int)): features channels of prediction module.
        mask_channels (int): Output channels of mask prediction layer.
            Default: 64.
        conv_cfg (dict, optional): Config dict of convolution layers in motion
            encoder. Default: None.
        norm_cfg (dict, optional): Config dict of norm layer in motion encoder.
            Default: None.
        act_cfg (dict, optional): Config dict of activation layer in motion
            encoder. Default: None.
    """

    def __init__(self, 
                 use_corr_pre=True,
                 corr_layer=None,
                 context_layer=None,
                *args, 
                **kwargs
                ):
        super().__init__(*args, **kwargs)
        
        self.use_corr_pre = use_corr_pre
        
        self.corr_block_pre = CorrelationPyramid(**self.corr_block_config)
        
        if corr_layer is not None:
            dim = (self.radius * 2 + 1) ** 2
            self.corr_layer = nn.Sequential(nn.Linear(dim, 2 * dim),
                                            nn.GELU(),
                                            nn.Linear(2 * dim, 2 * dim))
            
            self.corr_layer_pre = nn.Sequential(nn.Linear(dim, 2 * dim),
                                            nn.GELU(),
                                            nn.Linear(2 * dim, 2 * dim))
        else:
            self.corr_layer = None
            
        
        if context_layer is not None:
            self.context_layer = nn.Sequential(nn.Linear(context_layer['in_dim'], context_layer['hid_dim']),
                                            nn.GELU(),
                                            nn.Linear(context_layer['hid_dim'], context_layer['hid_dim']),
                                            nn.GELU(),
                                            nn.Linear(context_layer['hid_dim'], context_layer['out_dim']))
        else:
            self.context_layer = None
            

    def forward(
                self, 
                feats,
                coords,
                query_feat,
                feats_pre,
                query_feat_pre,
                vis=False
                ):
        """Forward function for RAFTDecoder.
        Args:
            feat1 (Tensor): The feature from the first input image.
            feat2 (Tensor): The feature from the second input image.
            flow (Tensor): The initialized flow when warm start.
            h (Tensor): The hidden state for GRU cell.
            cxt_feat (Tensor): The contextual feature from the first image.
        Returns:
            Sequence[Tensor]: The list of predicted optical flow.
        """

            
        _, T, C, H, W = feats.shape
        
        B, T, P, _ = coords.shape
        
        # B x T x P x C        
        query_feat_init = query_feat.clone()
        
        preds = []
        
        if vis:
            vis_results = {}
            vis_results['vis_crop_corrs'] = []
            vis_results['vis_coords'] = []
            point_id = random.randint(0, P-1)
        
        if self.use_corr_pre:
            corr_pyramid_pre = self.corr_block_pre(query_feat_pre, feats_pre)

        for _ in range(self.iters):
            coords = coords.detach()
            
            corr_pyramid = self.corr_block(query_feat, feats)
            
            cur_corrs = self.corr_lookup(corr_pyramid, coords)
            if self.corr_layer is not None:
                cur_corrs = self.corr_layer(cur_corrs)
            
            if self.use_corr_pre:
                cur_corrs_pre = self.corr_lookup(corr_pyramid_pre, coords/2)
            
                if self.corr_layer is not None:
                    cur_corrs_pre = self.corr_layer_pre(cur_corrs_pre)
                
                corr_input = torch.cat([cur_corrs, cur_corrs_pre], -1) 
            else:
                corr_input = cur_corrs
            
            # B x T x P x 2
            coords_ = coords - coords[:,0:1]
            
            # B x T x P x 1
            times_ = torch.linspace(0, T, T).cuda().reshape(1, T, 1, 1).repeat(B, 1, P, 1)
            
            t_ = torch.cat([coords_, times_], dim=-1).transpose(1,2).flatten(0,1)
            time_emb = get_3d_embedding(t_, self.time_dim, cat_coords=True).reshape(B,P,T,-1).transpose(1,2)
            
            if self.context_layer is not None:
                feat_input = self.context_layer(torch.cat([query_feat, query_feat_pre], -1))
            else:
                feat_input = query_feat
            
            x = torch.cat([corr_input, feat_input, time_emb], dim=-1) 
            
            # B T P x C
            delta_ = self.traj_pred(x)
            
            if self.use_update_feat:
                delta_feats_ = delta_[:,:,:,:-2].flatten(0,2)
                delta_feats_ = self.ffeat_updater(delta_feats_).reshape(B, T, P, C)
                query_feat = delta_feats_ + query_feat
            
            coords = coords + delta_[:,:,:,-2:]
            
            preds.append(coords * self.stride)
            
            if vis:
                crop_corrs = cur_corrs_pre[0,:,point_id,:49].reshape(T,1,7,7).transpose(2,3).detach().abs().cpu()
                crop_corrs = tensor2img(crop_corrs, norm_mode='0-1')
                vis_results['vis_crop_corrs'].append(crop_corrs)
                vis_results['vis_coords'].append(coords[0, :, point_id] * self.stride)
        
                if _ == self.iters - 1:
                    vis_results['vis_corrs'] = tensor2img(corr_pyramid_pre[0].reshape(B, T, P, 1, H//2, W//2)[0, :, point_id], norm_mode='0-1')

        if not vis:
            return preds, query_feat_init, query_feat, None
        else:
            return preds, query_feat_init, query_feat, vis_results