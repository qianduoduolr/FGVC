from ...registry import COMPONENTS

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as cp
from mmcv.runner import BaseModule
from mmpt.models.components.transformer_modules import *

class UpsampleBlock(BaseModule):
    def __init__(self, d_model, n_head, attention, layer_names, size, pos_emb, scale_factor=-1, align_corners=True):
        super(UpsampleBlock, self).__init__()
        self.trans = FeatureTransformer(d_model, n_head, attention, layer_names)
        self.align_corners = align_corners
        self.scale_factor = scale_factor
        self.size = size


    def forward(self, x):
        B, R, H, W = x.shape
        
        # B x HW x C
        x, _  = self.trans(x)
        
        x = x.permute(0,2,1).reshape(B, R, H, W)
        
        # B x C x 2H x 2W
        x = F.interpolate(x, size=self.size, align_corners=self.align_corners, mode='bilinear')
       
        return x
    
    
    
@COMPONENTS.register_module()
class CorrTransDecoder(BaseModule):
    def __init__(self, 
                 d_model=[1024, 2401], 
                 n_head=[8, 7], 
                 size=[(49,49), (128,128)], 
                 attention='linear', 
                 layer_names=[['self'],['self']], 
                 pos_emb=True, 
                 block_num=2
                 ):
        super(CorrTransDecoder, self).__init__()
        
        assert len(d_model) == len(n_head) == len(layer_names) == block_num
        
        decoder_modules = []
        for i in range(block_num):
            decoder_modules.append(UpsampleBlock(d_model[i], n_head[i], attention, layer_names[i], size[i], pos_emb))
            
        self.decoder_modules = nn.ModuleList(decoder_modules)
    
    def forward(self, x, shape):
        for idx, module in enumerate(self.decoder_modules):
            
            if idx == 1:
                B, C_, R, R = x.shape

                x = x.flatten(-2).permute(0,2,1).reshape(B, R*R, *shape)
                
            x = module(x)
            
        return x