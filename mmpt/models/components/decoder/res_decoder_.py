from ...registry import COMPONENTS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as cp
from mmcv.runner import BaseModule


class ResBlock(BaseModule):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r

class UpsampleBlock(BaseModule):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        
        if skip_f is not None:
            x = self.skip_conv(skip_f)
            x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        x = self.out_conv(x)
        return x

@COMPONENTS.register_module()
class Decoder(BaseModule):
    def __init__(self, in_c=1024, mid_c=512, out_c=256, scale=4):
        super().__init__()
        self.scale = scale
        self.compress = ResBlock(in_c, mid_c)
        self.up_16_8 = UpsampleBlock(mid_c, mid_c, out_c) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(out_c, out_c, out_c) # 1/8 -> 1/4

        self.pred = nn.Conv2d(out_c, 3, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8=None, f4=None):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x
