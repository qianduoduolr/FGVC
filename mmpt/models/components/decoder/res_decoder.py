from ...registry import COMPONENTS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as cp
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule, constant_init, kaiming_init


class BasicBlock(BaseModule):
    """Basic block for ResNet.

    Args:
        inplanes (int): Number of channels for the input in first conv2d layer.
        planes (int): Number of channels produced by some norm/conv2d layers.
        stride (int): Stride in the conv layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (BaseModule): Downsample layer. Default: None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)

        assert style in ['pytorch', 'caffe']
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.style = style
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class Bottleneck(BaseModule):
    """Bottleneck block for ResNet.

    Args:
        inplanes (int):
            Number of channels for the input feature in first conv layer.
        planes (int):
            Number of channels produced by some norm layes and conv layers
        stride (int): Spatial stride in the conv layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (BaseModule): Downsample layer. Default: None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """


    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False,
                init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv3 = ConvModule(
            planes,
            inplanes,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class UpsampleBlock(BaseModule):
    def __init__(self, in_c, out_c, size, align_corners=True, bottleneck=True):
        super().__init__()
        # self.pre_conv = ResBlock(in_c, in_c)
        if bottleneck:
            self.out_conv = Bottleneck(in_c, in_c//4)
        else:
            self.out_conv = BasicBlock(in_c, out_c)
            
        self.align_corners = align_corners
        self.size = size

    def forward(self, x):
        
        B, R, H, W = x.shape
        
        # B x HW x C
        x  = self.out_conv(x)
        
        # B x C x 2H x 2W
        x = F.interpolate(x, size=self.size, align_corners=self.align_corners, mode='bilinear')
       
        return x
    
    
@COMPONENTS.register_module()
class CorrResDecoder(BaseModule):
    def __init__(self, in_c=[1024, 2401],  out_c=[1024, 2401], size=[(49,49), (128,128)], block_num=2, align_corners=True):
        super().__init__()
        
        self.block_num = block_num
        
        decoder_modules = []
        for i in range(block_num):
            decoder_modules.append(UpsampleBlock(in_c[i], out_c[i], size[i], align_corners))
            
        self.decoder_modules = nn.ModuleList(decoder_modules)

    def forward(self, x, shape):
        for idx, module in enumerate(self.decoder_modules):
            
            if idx == self.block_num // 2:
                B, C_, R, R = x.shape

                x = x.flatten(-2).permute(0,2,1).reshape(B, R*R, *shape)
                
            x = module(x)
        
        return x
            
