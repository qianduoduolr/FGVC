import copy

import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ....registry import COMPONENTS
from .linear_attention import FullAttention, LinearAttention
from ..position_encoding import PositionEncodingSine


class TransEncoderLayer(BaseModule):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(TransEncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

@COMPONENTS.register_module()
class FeatureTransformer(BaseModule):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, d_model, nhead, attention, layer_names, temp_bug_fix=True, pos_emb=True):
        super(FeatureTransformer, self).__init__()

        self.d_model = d_model
        
        self.nhead = nhead
        self.layer_names = layer_names
        encoder_layer = TransEncoderLayer(d_model, nhead, attention)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self.pos_emb = pos_emb
        
        if self.pos_emb:
            self.pos_encoding = PositionEncodingSine(
            d_model,
            temp_bug_fix=temp_bug_fix)
            

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1=None, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        if feat0.dim() != 3:
            bsz, C, H, W = feat0.shape
            assert self.d_model == C, "the feature number of src and transformer must be equal"
        
        
        if self.pos_emb:
            feat0 = self.pos_encoding(feat0)
        
            if feat1 is not None:
                feat1 = self.pos_encoding(feat1)
        
        feat0 = feat0.permute(0,2,3,1).flatten(1,2)
        if feat1 is not None:
            feat1 = feat1.permute(0,2,3,1).flatten(1,2)
        

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                if feat1 is not None:
                    feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
        
        
        feat0 = feat0.permute(0,2,1).reshape(bsz, C, H, W)
        if feat1 is not None:
            feat1 = feat1.permute(0,2,1).reshape(bsz, C, H, W)
                    
        if feat1 is None:
            return feat0
        else:
            return feat0, feat1
        
        
        


@COMPONENTS.register_module()
class FeatureTransformerTemp(BaseModule):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, d_model, nhead, attention, layer_names, temp_bug_fix=True, pos_emb=True):
        super(FeatureTransformerTemp, self).__init__()

        self.d_model = d_model
        
        self.nhead = nhead
        self.layer_names = layer_names
        encoder_layer = TransEncoderLayer(d_model, nhead, attention)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self.pos_emb = pos_emb
        
        if self.pos_emb:
            self.pos_encoding = PositionEncodingSine(
            d_model,
            temp_bug_fix=temp_bug_fix)
            

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1=None, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        if feat0.dim() != 3:
            bsz, C, H, W = feat0.shape
            assert self.d_model == C, "the feature number of src and transformer must be equal"
        
        
        if self.pos_emb:
            feat0 = self.pos_encoding(feat0)
        
            T = feat1.shape[2]
            
            feat1 = feat1.transpose(1,2).flatten(0,1)
            feat1 = self.pos_encoding(feat1).reshape(bsz, T, C, H, W).transpose(1,2)
    
        feat0 = feat0.permute(0,2,3,1).flatten(1,2)
        feat1 = feat1.permute(0,2,3,4,1).flatten(1,3)
        

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                if feat1 is not None:
                    feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
        
        
        feat0 = feat0.permute(0,2,1).reshape(bsz, C, H, W)
        feat1 = feat1.permute(0,2,1).reshape(bsz, C, T, H, W)
            
        return feat0, feat1