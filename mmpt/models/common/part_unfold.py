import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .corr_lookup import *
from ..registry import OPERATORS


def part_unfold(x, radius, coord, t, mode='nearest', align_corner=False):
    """_summary_

    Args:
        x (_type_): values to be unfolded, B x C x  H x W
        radius (_type_): radius
        coord (_type_): coords to be unfold, B x S x 2
    """
    # B x C x (H+radius) x (W+radius)
    # a1 = (x[0].sum(0) > 0).cpu().numpy()
    x = F.pad(x, (radius,radius, radius, radius), mode='constant', value=0)
    
    # a2 = (x[0].sum(0) > 0).cpu().numpy()
    
    
    B, S = coord.shape[:2]
    L = 2 * radius + 1
    C = x.shape[1]
    
    centroid_lvl = coord.reshape(B * S, 1, 1, 2)
    
    dx = torch.linspace(-radius, radius, 2 * radius + 1, device=x.device)
    dy = torch.linspace(-radius, radius, 2 * radius + 1, device=x.device)
    delta = torch.stack(torch.meshgrid(dy, dx)[::-1], axis=-1)
    delta_lvl = delta.view(1, 2 * radius + 1, 2 * radius + 1, 2)
    
    coords_lvl = centroid_lvl + delta_lvl
    # coords_lvl = coords_lvl.transpose(1,2)
    
    # (t*S) x  R x R x 2
    coords_lvl = coords_lvl.repeat(t, 1, 1, 1)
    
    
    # (t*S) x C x (H+radius) x (W+radius)
    x = x[:,None].repeat(1,coord.shape[1], 1, 1, 1).flatten(0,1)
    
    # t x S x C x R^2
    unfold_x = bilinear_sample(x, coords_lvl, mode=mode, align_corners=align_corner).reshape(-1, S, C, L**2)

    
    return unfold_x