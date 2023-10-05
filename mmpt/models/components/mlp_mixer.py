from ..registry import COMPONENTS
from ..builder import build_components, build_loss, build_drop_layer
from mmcv.cnn import ConvModule, build_norm_layer, build_plugin_layer, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
import torch
import torch.nn as nn

from functools import partial
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(BaseModule):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

@COMPONENTS.register_module()
class MLP_Mixer(BaseModule):
    
    def __init__(self, 
                 input_dim,
                 dim, 
                 depth, 
                 expansion_factor=4, 
                 T=5,
                 dropout=0., 
                 init_cfg=None
                 ):
        super().__init__(init_cfg)
        
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.input_dim = input_dim
        self.window_size = T
        
        self.pre_layer = nn.Linear(input_dim, dim)
        
        self.mlp_layers = nn.Sequential(
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(T, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean')
        )
        
        self.post_layer = nn.Linear(dim, T*2)
        
        
    def forward(self, x):
        
        B, T, P, C = x.shape

        x = x.transpose(1,2).flatten(0,1)
        x = self.pre_layer(x)
        x = self.mlp_layers(x)
        
        x = self.post_layer(x).reshape(B,P,T,2).transpose(1,2)

        return x
    


@COMPONENTS.register_module()
class MLP_Mixer_PIPS(BaseModule):
    
    def __init__(self, 
                 input_dim,
                 dim, 
                 depth, 
                 expansion_factor=4, 
                 T=5,
                 corr_levels=4, corr_radius=3,
                 dropout=0., 
                 update_feat=True,
                 init_cfg=None
                 ):
        super().__init__(init_cfg)
        
        self.window_size = T
        
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        
        self.input_dim = input_dim
        
        kitchen_dim = (corr_levels * (2*corr_radius + 1)**2) + input_dim + 64*3 + 3

        
        self.pre_layer = nn.Linear(kitchen_dim, dim)
        
        self.mlp_layers = nn.Sequential(
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(T, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean')
        )
        
        self.update_feat = update_feat
        
        if self.update_feat:
            self.post_layer = nn.Linear(dim, T*(input_dim+2))
        else:
            self.post_layer = nn.Linear(dim, T*2)
        
        
    def forward(self, x):
        
        B, T, P, C = x.shape

        x = x.transpose(1,2).flatten(0,1)
        x = self.pre_layer(x)
        x = self.mlp_layers(x)
        
        if self.update_feat:
            x = self.post_layer(x).reshape(B,P,T,(self.input_dim+2)).transpose(1,2)
        else:
            x = self.post_layer(x).reshape(B,P,T,2).transpose(1,2)

        return x



class DepthwiseSeparableConv1D(BaseModule):
    def __init__(self, nin, nout, kernel_size=3):
        super(DepthwiseSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(nin, nin, kernel_size=kernel_size, padding=1, groups=nin)
        self.pointwise = nn.Conv1d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DepthwiseConv1DBlock(BaseModule):
    def __init__(self, dim, expansion_factor=4, dropout=0., init_cfg=None):
        super().__init__(init_cfg)
        
        self.dim = dim
        self.expansion_factor = expansion_factor
        self.depth_conv = nn.Conv1d(dim, dim * expansion_factor, kernel_size=3, stride=1, padding=1, groups=dim)
        self.act =  nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj_conv = nn.Conv1d(dim, dim, 1, groups=dim)

    def forward(self, x):

        B, T, C = x.shape        
        
        x = x.transpose(1,2)
        
        x = self.depth_conv(x)
        x = self.act(x)
        x = self.drop(x)
        
        x = x.reshape(B, C, self.expansion_factor, T).transpose(1,2).flatten(0,1)
        
        x = self.proj_conv(x).reshape(B, self.expansion_factor, -1, T)
        
        x = torch.sum(x, 1)
        
        x = self.drop(x)
        
        return x.transpose(1,2)




@COMPONENTS.register_module()
class Depthwise_Conv_Mixer_PIPS(BaseModule):
    
    def __init__(self, 
                 input_dim,
                 dim, 
                 depth, 
                 T=-1,
                 expansion_factor=4, 
                 corr_levels=4, corr_radius=3,
                 dropout=0., 
                 init_cfg=None
                 ):
        super().__init__(init_cfg)
        
        self.window_size = T
        
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        
        self.input_dim = input_dim
        
        kitchen_dim = (corr_levels * (2*corr_radius + 1)**2) + input_dim + 64*3 + 3
        
        self.pre_layer = nn.Linear(kitchen_dim, dim)
        
        self.mlp_layers = nn.Sequential(
        *[nn.Sequential(
            PreNormResidual(dim, DepthwiseConv1DBlock(dim, expansion_factor, dropout)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        )
        
        self.post_layer = nn.Linear(dim, input_dim+2)
        
        
    def forward(self, x):
        
        B, T, P, C = x.shape

        x = x.transpose(1,2).flatten(0,1)
        x = self.pre_layer(x)
        x = self.mlp_layers(x)
        
        x = self.post_layer(x).reshape(B,P,T,(self.input_dim+2)).transpose(1,2)

        return x