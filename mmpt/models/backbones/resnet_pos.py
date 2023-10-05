import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import get_root_logger
from ..common import change_stride
from ..registry import BACKBONES
from ..registry import COMPONENTS


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=-1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.SyncBatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.SyncBatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
    
@BACKBONES.register_module()
class ResNet18_Pos(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2,2,2,2], in_channel=3, width=1,
                 groups=1, width_per_group=64,
                 mid_dim=1024, low_dim=128):
        super(ResNet18_Pos, self).__init__()
        self.inplanes = 64 * width
        self.base = int(64 * width)
        self.groups = groups
        self.base_width = width_per_group

        mid_dim = self.base * 8 * block.expansion

        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.SyncBatchNorm(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 64, 128, 128))
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=1)
        self.layer4 = self._make_layer(block, self.base * 4, layers[3], stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # zero gamma for batch norm: reference bag of tricks
        if block is BasicBlock:
            gamma_name = "bn2.weight"
        else:
            raise RuntimeError(f"block {block} not supported")
        for name, value in self.named_parameters():
            if name.endswith(gamma_name):
                value.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.SyncBatchNorm(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x, ape=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        pos_embed1 = F.interpolate(self.pos_embed1, size=(x.size(2), x.size(3)), mode='bicubic')
        if ape:
            x = x + pos_embed1
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        # c5 = self.layer4(c4)
       
        return c4