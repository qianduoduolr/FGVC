from functools import partial

import numpy as np
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint, BaseModule
from mmcv.utils import _BatchNorm
from torch import nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as cp

from ...utils import get_root_logger
from ..common import change_stride
from ..registry import BACKBONES
from ..registry import COMPONENTS


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, kernel_size=3, activation=F.relu):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.activation(out)
        return out

@BACKBONES.register_module()
class ResNet18MAST(nn.Module):
    def __init__(self, in_ch=3):
        super(ResNet18MAST, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=1)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer4 = self.make_layer(ResidualBlock, 256, 2, stride=1)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


@BACKBONES.register_module()
class ResNet18VQ(nn.Module):
    def __init__(self, in_ch=3):
        super(ResNet18VQ, self).__init__()
        
        self.inchannel = 256
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )

        self.layer1 = self.make_layer(ResidualBlock, 256,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 512, 2, stride=1)
        # self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        # self.layer4 = self.make_layer(ResidualBlock, 256, 2, stride=1)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = self.conv1(x)
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        return out