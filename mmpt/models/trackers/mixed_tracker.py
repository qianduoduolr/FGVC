import enum
import os.path as osp
import tempfile
from builtins import isinstance, list
from collections import *
from pickle import NONE
from re import A
from turtle import forward

import mmcv
import torch.nn as nn
import torch.nn.functional as F
from dall_e import load_model, map_pixels, unmap_pixels
from mmcv.ops import Correlation
from mmcv.runner import CheckpointLoader, auto_fp16, load_checkpoint
from torch import bilinear, unsqueeze
from tqdm import tqdm

from mmpt.models.common import (coords_grid, bilinear_sample)
from mmpt.models.components import XHead
from mmpt.models.common.correlation import *
from mmpt.models.common.gradient_reversal import *

from mmpt.utils import *

from ..builder import (build_backbone, build_components, build_loss,
                       build_model, build_operators)

from mmpt.models.common.occlusion_estimation import *

from ..registry import MODELS
from .base import BaseModel
from .vanilla_tracker import VanillaTracker
from .modules import *

class GradReverseDiscriminator(BaseModule):
    def __init__(self, feat_dim=256, alpha=1):
        super().__init__()
        
        self.cls_layer = nn.Sequential(
                        GradientReversal(alpha=alpha),
                        nn.Linear(feat_dim, feat_dim//2),
                        nn.ReLU(),
                        nn.Linear(feat_dim//2, feat_dim//4),
                        nn.ReLU(),
                        nn.Linear(feat_dim//4, 1)
                    )
        
    def forward(self, x):
        
        x = x.permute(0,2,3,1).flatten(0,2)
        x = self.cls_layer(x)
        
        return x
    

class VanillaDiscriminator(BaseModule):
    def __init__(self, feat_dim=256, alpha=1):
        super().__init__()
        
        self.cls_layer = nn.Sequential(
                        nn.Linear(feat_dim, feat_dim//2),
                        nn.ReLU(),
                        nn.Linear(feat_dim//2, feat_dim//4),
                        nn.ReLU(),
                        nn.Linear(feat_dim//4, 1)
                    )
        
    def forward(self, x):
        
        x = x.permute(0,2,3,1).flatten(0,2)
        
        return self.cls_layer(x)


@MODELS.register_module()
class Memory_Tracker_Custom_V2(VanillaTracker):
    def __init__(self,
                 backbone,
                 loss_weight=dict(l1_loss=1),
                 head=None,
                 downsample_rate=[4,],
                 radius=[12,],
                 temperature=1,
                 feat_size=[64,],
                 conc_loss=None,
                 forward_backward_t=-1,
                 scaling=True,
                 upsample=True,
                 drop_ch=True,
                 weight=20,
                 rec_sampling='stride',
                 pretrained=None,
                 *args, **kwargs
                 ):
        """ MAST  (CVPR2020) using MMCV Correlation Module

        Args:
            backbone ([type]): [description]
            test_cfg ([type], optional): [description]. Defaults to None.
            train_cfg ([type], optional): [description]. Defaults to None.
        """
        super().__init__(backbone=backbone, *args,
                 **kwargs)
        
        self.downsample_rate = downsample_rate
        if head is not None:
            self.head =  build_components(head)
        else:
            self.head = None
            
        self.logger = get_root_logger()
        
        self.pretrained = pretrained
        self.scaling = scaling
        self.upsample = upsample
        self.drop_ch = drop_ch
        self.rec_sampling = rec_sampling
        self.weight = weight
        self.temperature = temperature
        self.forward_backward_t = forward_backward_t
        self.conc_loss = build_loss(conc_loss) if conc_loss is not None else None
        
        self.radius = radius 
        self.feat_size = feat_size

        self.loss_weight = loss_weight

        assert len(self.feat_size) == len(self.radius) == len(self.downsample_rate) 
    
        self.corr = [ Correlation(max_displacement=R) for R in self.radius ]
        
    def get_correspondence(self, query_feat, key_feats):
        
        if self.head is not None:
            query_feat, key_feats = self.head(query_feat, key_feats)
        
        if self.test_cfg.get('withnorm', True):
            query_feat = F.normalize(query_feat, dim=1)
            key_feats = F.normalize(key_feats, dim=1)
        
        aff = self.corr_infer(query_feat, key_feats).flatten(1,2)
        
        return aff
        
    def dropout2d_lab(self, arr): # drop same layers for all images
        if not self.training:
            return arr

        drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
        drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)

        for idx, a in enumerate(arr):
            if self.drop_ch:
                for dropout_ch in drop_ch_ind:
                    a[:, dropout_ch] = 0
                a *= (3 / (3 - drop_ch_num))

        return arr, drop_ch_ind # return channels not masked
    
    def compute_lphoto(self, images_lab_gt, ch, outputs, upsample=True, mask=None, gt_idx=-1):
        b, c, h, w = images_lab_gt[0].size()

        tar_y = images_lab_gt[gt_idx][:,ch]  # y4

        if upsample:
            outputs = F.interpolate(outputs, (h, w), mode='bilinear')
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='none')
            if mask == None:
                loss = loss.mean()
            else:
                loss = (loss * mask).sum() / (mask.sum() + 1e-9)
                
        else:
            tar_y = self.prep(images_lab_gt[-1], self.downsample_rate[-1])[:,ch]
            loss = F.smooth_l1_loss(outputs * self.weight, tar_y * self.weight, reduction='none')
            if mask == None:
                loss = loss.mean()
            else:
                loss = (loss * mask).sum() / (mask.sum() + 1e-9)

        err_maps = torch.abs(outputs - tar_y).sum(1).detach()

        return loss, err_maps
    
    def prep(self, image, downsample_rate=8):
        _,c,_,_ = image.size()

        if self.rec_sampling == 'stride':
            x = image.float()[:,:,::downsample_rate,::downsample_rate]
        elif self.rec_sampling == 'centered':
            x = image.float()[:,:,downsample_rate//2::downsample_rate, downsample_rate//2::downsample_rate]
        else:
            raise NotImplementedError

        return x
        
    def forward_train(self, images_lab, imgs=None):
            
        bsz, _, n, c, h, w = images_lab.shape
        
        images_lab_gt = [images_lab[:,0,i].clone() for i in range(n)]
        images_lab = [images_lab[:,0,i] for i in range(n)]
        _, ch = self.dropout2d_lab(images_lab)
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
        fs = fs.reshape(bsz, n, *fs.shape[-3:])
        
        if self.head is not None:
            # self && cross transformer
            f2, f1 = self.head(fs[:,-1], fs[:,0])
            fs = torch.stack([f1,f2], 1)
        
        tar, refs = fs[:, -1], fs[:, :-1]
        
        # get correlation attention map      
        corr = self.corr[0](tar, refs[:,0]) 
        if self.scaling:
            corr = corr / torch.sqrt(torch.tensor(tar.shape[1]).float()) 
        corr = corr.flatten(1,2).softmax(1)
        
        losses = {}
        
        # for mast l1_loss
        ref_gt = self.prep(images_lab_gt[0][:,ch], self.downsample_rate[0])
        ref_gt = F.unfold(ref_gt, self.radius[0] * 2 + 1, padding=self.radius[0])
        
        corr = corr.reshape(bsz, -1, tar.shape[-1]**2)
        outputs = (corr * ref_gt).sum(1).reshape(bsz, -1, *fs.shape[-2:])        
        losses['l1_loss'], err_map = self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample)
        
        vis_results = dict(err=err_map[0], imgs=imgs[0,0])

        return losses, vis_results

        
@MODELS.register_module()
class Mixed_Tracker(Memory_Tracker_Custom_V2):
    def __init__(self, 
                 teacher,
                 neck=None,
                loss=dict(type='Soft_Ce_Loss'),
                scale=8,
                temperature_t=0.07, 
                rec_mask=True,
                norm=True,
                bilateral=False,
                *args, 
                **kwargs
                ):
        """
        Main framework of our mixed training using synthetic and unlabeled videos, which
        consists of threee parts: (i) Self-supervised frame reconstruction with unlabeled 
        videos ; (ii) Supervised training with labeled synthetic videos ; (iii) adversarial
        modules.

        Args:
            teacher: the pre-trained 2D encoder.
            loss: loss function. Default to cross-entropy loss. 
            scale: feature down-sample scale..
            temperature_t: temperature for the probilistic mapping. Default to 0.07.
            rec_mask: whether use mask in frame rec. Defaults to True.
            norm: wheher normalize features. Defaults to True.
            bilateral: whether use bilateral filtering. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        
        
        self.loss = build_loss(loss)
        self.loss_type = loss.get('type', 'Soft_Ce_Loss')
        
        if teacher is not None:
            self.teacher = build_backbone(teacher)
            self.teacher.eval()
        else:
            self.teacher = None
            
        self.scale = scale
        self.grid_size = self.radius[0]*2+1
        self.rec_mask = rec_mask
        self.norm = norm
        self.temperature_t = temperature_t
        self.bilateral = bilateral
        
        self.corr_discriminator = GradReverseDiscriminator(self.grid_size**2)
        self.discriminator = GradReverseDiscriminator(256)
        
        self.neck = build_components(neck) if neck is not None else None
        
    def draw_gaussion_map_online(self, flow, flow_back, sigma=3, imgs=None):
            
        flow = flow.permute(0,3,1,2)
        flow_back = flow_back.permute(0,3,1,2)
        
        mask = occlusion_estimation(flow, flow_back)['occ_fw']    
        # m = mask[0,0].cpu().numpy()*255   
        mask = mask[:,:,::2, ::2].reshape(-1)
        
        # B x 2 x H x W
        flow_d = flow[:, :, ::2, ::2] / 2
        flow_d = torch.round(flow_d + self.radius[0])
        valid_w =   (flow_d[:,0] >= 0) & (flow_d[:,0] <= self.radius[0]*2)
        valid_h =   (flow_d[:,1] >= 0) & (flow_d[:,1] <= self.radius[0]*2)
        valid = (valid_w & valid_h).view(-1) * mask.bool()
        
        B, _, H, W = flow_d.shape
        xx = torch.arange(0, self.grid_size, device=flow.device)
        yy = torch.arange(0, self.grid_size, device=flow.device)
        grid = coords_grid(B*H*W, xx, yy)  # shape BHW, 2, R, R
        
        flow_d = flow_d.permute(0,2,3,1).flatten(0,2)[:,:,None,None]   # BHW, 2, 1, 1 
        
        # BHW, R, R
        g = torch.exp(-((grid[:,0,:,:] - flow_d[:,0,:,:])**2 + (grid[:,1,:,:] - flow_d[:,1,:,:])**2) / (2 * sigma**2))
        # g[~valid, :, :] = 1
        g = g.reshape(B, -1, self.grid_size ** 2)
        
        return g, valid
    
    def draw_selfatt_map_online(self, flow, flow_back, feats, imgs=None):
            
        flow = flow.permute(0,3,1,2)
        flow_back = flow_back.permute(0,3,1,2)
        
        mask = occlusion_estimation(flow, flow_back)['occ_fw']    
        mask = mask[:,:,::self.scale, ::self.scale].reshape(-1)
        
        # B x 2 x H x W
        flow_ = flow[:, :, ::self.scale, ::self.scale] / 2
        B, _, H, W = flow_.shape
        
        flow_d = flow_ + self.radius[0]
        valid_w =   (flow_d[:,0] >= 0) & (flow_d[:,0] <= self.radius[0]*2)
        valid_h =   (flow_d[:,1] >= 0) & (flow_d[:,1] <= self.radius[0]*2)
        valid = (valid_w & valid_h).view(-1) * mask.bool()
        
        xx = torch.arange(0, W, device=flow.device)
        yy = torch.arange(0, H, device=flow.device)
        grid = coords_grid(B, xx, yy) + flow_  # shape N, 2, H, W
        grid = grid.permute(0, 2, 3, 1)  # shape N, H, W, 2
        
        warp_feats = bilinear_sample(feats, grid, align_corners=True)
        
        att = self.corr[0](warp_feats, feats).flatten(1,2)
        if self.temperature_t != -1:
            att = att / self.temperature_t

        g = att.flatten(2).permute(0,2,1)
        
        
        return g, valid
    
    def vis_corr(self, imgs_sup, flow, flow_back):
        bsz, _, n, c, h, w = imgs_sup.shape
        
        
        images_lab_sup_gt = imgs_sup[:,0,0].clone()
        images_lab_sup = [imgs_sup[:,0,i] for i in range(n)]
        _, ch = self.dropout2d_lab(images_lab_sup)
        
        self_f = self.teacher(images_lab_sup_gt).detach()
        self_f = F.normalize(self_f, dim=2)
        
        self_att, valid = self.draw_selfatt_map_online(flow, flow_back, self_f.detach())
        gaussion, _ = self.draw_gaussion_map_online(flow, flow_back)
        
        return self_att, gaussion, valid
    
    def forward_train(self, imgs, imgs_sup, flow, flow_back):
        
        bsz, _, n, c, h, w = imgs.shape
        
        losses = {}

        if self.loss_weight['l1_loss'] > 0:
            
            images_lab_gt = [imgs[:,0,i].clone() for i in range(n)]
            images_lab = [imgs[:,0,i] for i in range(n)]
            _, ch = self.dropout2d_lab(images_lab)
                        
            # forward to get feature
            fs = self.backbone(torch.stack(images_lab,1).flatten(0,1))
            fs = fs.reshape(bsz, n, *fs.shape[-3:])
            
            if self.head is not None:
                # self && cross transformer
                f2, f1 = self.head(fs[:,-1], fs[:,0])
                fs = torch.stack([f1,f2], 1)
            
            tar, refs = fs[:, -1], fs[:, :-1]
            
            # get correlation attention map      
            corr_target = self.corr[0](tar, refs[:,0]) 
            if self.scaling:
                corr_rec = corr_target / torch.sqrt(torch.tensor(tar.shape[1]).float()) 
            corr_rec = corr_rec.flatten(1,2).softmax(1)
            
            # for mast l1_loss
            ref_gt = self.prep(images_lab_gt[0][:,ch], self.downsample_rate[0])
            ref_gt = F.unfold(ref_gt, self.radius[0] * 2 + 1, padding=self.radius[0])
            
            corr_rec = corr_rec.reshape(bsz, -1, tar.shape[-1]**2)
            outputs = (corr_rec * ref_gt).sum(1).reshape(bsz, -1, *fs.shape[-2:])        
            losses['l1_loss'] = self.loss_weight['l1_loss'] * self.compute_lphoto(images_lab_gt, ch, outputs, upsample=self.upsample, mask=None, gt_idx=-1)[0]
        
        
        ############## Flow Sup ##############
        images_lab_sup_gt = imgs_sup[:,0,0].clone()
        images_lab_sup = [imgs_sup[:,0,i] for i in range(n)]
        _, ch = self.dropout2d_lab(images_lab_sup)
        
        
        # forward to get pseduo labels
        with torch.no_grad():
            if self.teacher is not None:
                self.teacher.eval()
                self_f = self.teacher(images_lab_sup_gt).detach()
            else:
                self_f = self.backbone(images_lab_sup_gt).detach()
                
            if self.norm:
                self_f = F.normalize(self_f, dim=2)


        heat_map, valid = self.draw_selfatt_map_online(flow, flow_back, self_f.detach())
        
        if self.bilateral:
            heat_map_gaussion, _ = self.draw_gaussion_map_online(flow, flow_back, sigma=6)
            heat_map *= heat_map_gaussion
            
        heat_map = heat_map.reshape(-1, self.grid_size**2)
    
        
        # forward to get feature
        fs = self.backbone(torch.stack(images_lab_sup,1).flatten(0,1))
        fs = fs.reshape(bsz, n, *fs.shape[-3:])
        
        if self.head is not None:
            # self && cross transformer
            f2, f1 = self.head(fs[:,-1], fs[:,0])
            fs = torch.stack([f1,f2], 1)

        if self.norm:
            fsn = F.normalize(fs, dim=2)
        
        tar, refs = fsn[:, -1], fsn[:, :-1]
        
        # get correlation attention map      
        corr = self.corr[0](tar, refs[:,0]).flatten(1,2)
        if self.temperature_t != -1:
            corr = corr / self.temperature_t
            
        pred = corr.permute(0,2,3,1).reshape(-1, self.grid_size**2)[valid]
        gt = heat_map[valid].to(torch.float32)
        

        losses['sup_loss'] = self.loss_weight['sup_loss'] * self.loss(pred, gt)
            
        
        
        if self.loss_weight['corr_da_loss'] > 0:
            corr_source = self.corr[0](fs[:,-1], fs[:,0]).flatten(1,2)
            
            pred_corr_target = self.corr_discriminator(corr_target.flatten(1,2))
            pred_corr_source = self.corr_discriminator(corr_source)
            
            preds_corr = torch.cat([pred_corr_source, pred_corr_target], 0)
            gt_corr = torch.cat([torch.zeros(pred_corr_source.shape[0]), torch.ones(pred_corr_target.shape[0])],0).cuda()
            
            losses['corr_da_loss'] = self.loss_weight['corr_da_loss'] * F.binary_cross_entropy_with_logits(preds_corr, gt_corr[:,None])
    
    
        return losses, None
