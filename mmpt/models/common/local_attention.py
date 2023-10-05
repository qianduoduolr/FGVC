import math
import time
from typing import List

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmpt.models.common import part_unfold


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """Efficient version of torch.cat that avoids a copy if there is only a
    single element in a list."""
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def coords_grid(batch: int, xx, yy):
    """Coordinate grid.
    Args:
        batch (int): The batch size of feature.
        xx (Tensor): 1-D tensor of size W with values from the interval
            [0, W-1].
        yy (Tensor): 1-D tensor of size H with values from the interval
            [0, H-1].
    Returns:
        Tensor: Tensor of shape (batch, 2, H, W) with values of items'
            coordinate.
    """
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()

    return coords[None].repeat(batch, 1, 1, 1)  # shape(batch, 2, H, W)


def local_square_attention(query,
                           key,
                           value,
                           kernel_size,
                           temperature=1,
                           topk=None,
                           batch_as_context=False):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, H, W)
        kernel_size (int | tuple[int]):
        temperature (float)
        topk (int)
        batch_as_context (bool): Take batches as context for key

    Returns:

    """
    assert query.ndim == key.ndim == 4
    assert query.shape[1:] == key.shape[1:]
    assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    assert value.shape[0] == key.shape[0]
    channels, height, width = query.shape[1:]
    kernel_size = _pair(kernel_size)
    padding = tuple(k // 2 for k in kernel_size)
    # [N, Cxhxw, HxW]
    unfolded_key = F.unfold(key, kernel_size=kernel_size, padding=padding)
    unfolded_value = F.unfold(value, kernel_size=kernel_size, padding=padding)
    # [N, C, hxw, HxW]
    unfolded_key = unfolded_key.view(unfolded_key.shape[0], channels,
                                     kernel_size[0] * kernel_size[1],
                                     height * width)
    unfolded_value = unfolded_value.view(unfolded_value.shape[0],
                                         value.shape[1],
                                         kernel_size[0] * kernel_size[1],
                                         height * width)
    # [N, C, 1, HxW]
    unfolded_query = query.reshape(query.shape[0], channels,
                                   height * width).unsqueeze(2)
    if batch_as_context:
        # [1, C, Nxhxw, HxW]
        unfolded_key.transpose_(0, 1)
        unfolded_key = unfolded_key.reshape(
            1, channels, key.shape[0] * kernel_size[0] * kernel_size[1],
            height * width)
        unfolded_value.transpose_(0, 1)
        unfolded_value = unfolded_value.reshape(
            1, value.shape[1],
            value.shape[0] * kernel_size[0] * kernel_size[1], height * width)
    # [N, 1, hxw, HxW] or [N, 1, Nxhxw, HxW]
    attention = torch.zeros(query.shape[0], 1,
                            *unfolded_key.shape[2:]).to(unfolded_query)
    spatial_step = 512
    for ptr in range(0, height * width, spatial_step):
        attention[..., ptr:ptr + spatial_step] = torch.sum(
            unfolded_query[..., ptr:ptr + spatial_step] *
            unfolded_key[..., ptr:ptr + spatial_step],
            dim=1,
            keepdim=True)
    attention /= temperature
    # attention = torch.sum(unfolded_query * unfolded_key, dim=1,
    #                       keepdim=True) / temperature
    if topk is not None:
        topk_attention, topk_indices = attention.topk(k=topk, dim=2)
        # [N, 1, topk, HxW]
        attention = topk_attention
        # [N, C, topk, HxW]
        unfolded_value = unfolded_value.gather(
            dim=2, index=topk_indices.expand(-1, value.shape[1], -1, -1))
    # [N, C, HxW]
    output = torch.sum(attention * unfolded_value, dim=2)
    output = output.reshape(output.shape[0], output.shape[1], height, width)

    return output


def local_corr_attention(query,
                         key,
                         value,
                         kernel_size,
                         temperature=1,
                         topk=None,
                         batch_as_context=False):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, H, W)
        kernel_size (int | tuple[int]):
        temperature (float)
        topk (int)
        batch_as_context (bool): Take batches as context for key

    Returns:

    """
    # not tested
    from spatial_correlation_sampler import spatial_correlation_sample
    assert query.ndim == key.ndim == 4
    assert query.shape[1:] == key.shape[1:]
    assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    assert value.shape[0] == key.shape[0]
    channels, height, width = query.shape[1:]
    kernel_size = _pair(kernel_size)
    padding = tuple(k // 2 for k in kernel_size)
    assert batch_as_context
    assert query.shape[0] == 1
    # [N, Cxhxw, HxW]
    unfolded_value = F.unfold(value, kernel_size=kernel_size, padding=padding)
    # [N, C, hxw, HxW]
    unfolded_value = unfolded_value.view(unfolded_value.shape[0],
                                         value.shape[1],
                                         kernel_size[0] * kernel_size[1],
                                         height * width)
    key_batch_size = key.shape[0]
    attentions = []
    for i in range(key_batch_size):
        # [1, h, w, H, W]
        attention = spatial_correlation_sample(
            query, key[i:i + 1], kernel_size=1, patch_size=kernel_size)
        attentions.append(attention)
    # [N, h, w, H, W]
    attentions = cat(attentions, dim=0)
    attentions /= temperature
    # [C, Nxhxw, HxW]
    unfolded_value.transpose_(0, 1)
    unfolded_value = unfolded_value.reshape(
        value.shape[1], value.shape[0] * kernel_size[0] * kernel_size[1],
        height * width)
    # [1, Nxhxw, HxW]
    attentions = attentions.view(
        1, key.shape[0] * kernel_size[0] * kernel_size[1], height * width)

    if topk is not None:
        topk_attentions, topk_indices = attentions.topk(k=topk, dim=1)
        # [1, topk, HxW]
        attentions = topk_attentions
        # [C, topk, HxW]
        unfolded_value = unfolded_value.gather(
            dim=1, index=topk_indices.expand(value.shape[1], -1, -1))
    # [C, 1, HxW]
    output = torch.einsum('cij,bij->cbj', attentions.softmax(dim=1),
                          unfolded_value)
    output.transpose_(0, 1)
    output = output.reshape(1, value.shape[1], height, width)

    return output


def masked_attention(query,
                     key,
                     value,
                     mask,
                     temperature=1,
                     topk=None,
                     normalize=True,
                     step=100):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, T, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, T, H, W)
        temperature (float)
        topk (int)
        normalize (bool)
        step (int)

    Returns:

    """
    batches = query.size(0)
    assert query.size(0) == key.size(0) == value.size(0)
    assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    if key.ndim == 4:
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
    assert value.ndim == key.ndim == 5
    clip_len = key.size(2)
    assert query.shape[2:] == key.shape[3:]
    att_channels, height, width = query.shape[1:]
    C = value.size(1)
    if normalize:
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)
    query_vec = query.view(batches, att_channels, query.shape[2:].numel())
    key_vec = key.view(batches, att_channels, key.shape[2:].numel())
    value_vec = value.view(batches, C, value.shape[2:].numel())
    # [N, TxHxW, HxW]
    affinity = torch.einsum('bci,bcj->bij', key_vec, query_vec) / temperature
    mask = mask.view(1, height * width,
                     height * width).expand(clip_len, -1,
                                            -1).reshape_as(affinity)
    affinity.masked_fill_(~mask.bool(), float('-inf'))
    output = torch.zeros(batches, C, height * width).to(query)
    for ptr in range(0, height * width, step):
        # [N, TxHxW, step]
        cur_affinity = affinity[:, :, ptr:ptr + step]
        if topk is not None:
            # [N, topk, step]
            topk_affinity, topk_indices = cur_affinity.topk(k=topk, dim=1)
            # cur_affinity, idx = cur_affinity.sort(descending=True, dim=1)
            # topk_affinity, topk_indices = cur_affinity[:, :topk], idx[:,
            # :topk]
            # assert torch.allclose(topk_affinity, topk_affinity_)
            # assert torch.allclose(topk_indices, topk_indices_)
            topk_value = value_vec.transpose(0, 1).reshape(
                C, -1).index_select(
                    dim=1, index=topk_indices.reshape(-1))
            # [N, C, topk, step]
            topk_value = topk_value.reshape(C,
                                            *topk_indices.shape).transpose(
                                                0, 1)
            cur_output = torch.einsum('bcks,bks->bcs', topk_value,
                                      topk_affinity.softmax(dim=1))
        else:
            cur_output = torch.einsum('bck,bks->bcs', value_vec,
                                      cur_affinity.softmax(dim=1))
        output[..., ptr:ptr + step] = cur_output

    output = output.reshape(batches, C, height, width)

    return output


def masked_attention_efficient(query,
                               key,
                               value,
                               mask,
                               temperature=1,
                               topk=None,
                               normalize=True,
                               step=32,
                               non_mask_len=0,
                               mode='softmax',
                               sim_mode='dot_product'):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, T, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, T, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    assert mode in ['softmax', 'cosine']
    batches = query.size(0)
    assert query.size(0) == key.size(0) == value.size(0)
    assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    if key.ndim == 4:
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
    assert value.ndim == key.ndim == 5
    clip_len = key.size(2)
    assert 0 <= non_mask_len < clip_len
    # assert query.shape[2:] == key.shape[3:]
    att_channels, query_height, query_width = query.shape[1:]
    key_height, key_width = key.shape[3:]
    C = value.size(1)
    if normalize:
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)
    query_vec = query.view(batches, att_channels, query.shape[2:].numel())
    key_vec = key.view(batches, att_channels, key.shape[2:].numel())
    value_vec = value.view(batches, C, value.shape[2:].numel())
    output = torch.zeros(batches, C,
                         query_height * query_width).to(query)
    if step is None:
        step = query_height * query_width
    for ptr in range(0, query_height * query_width, step):
        # [N, TxHxW, step]
        if sim_mode == 'dot_product':
            cur_affinity = torch.einsum('bci,bcj->bij', key_vec,
                                        query_vec[...,
                                                ptr:ptr + step]) / temperature
        elif sim_mode == 'l2-distance':
            a_sq = key_vec.pow(2).sum(1).unsqueeze(2)
            ab = key_vec.transpose(1, 2) @ query_vec[...,ptr:ptr + step]
            cur_affinity = (2*ab-a_sq) / math.sqrt(att_channels)

        if mask is not None:
            if mask.ndim == 2:
                assert mask.shape == (key_height * key_width,
                                      query_height * query_width)
                cur_mask = mask.view(1, 1, key_height * key_width,
                                     query_height *
                                     query_width)[..., ptr:ptr + step].expand(
                                         batches, clip_len - non_mask_len, -1,
                                         -1).reshape(batches, -1,
                                                     cur_affinity.size(2))
            else:
                cur_mask = mask.view(1, 1, key_height * key_width,
                                     query_height *
                                     query_width)[..., ptr:ptr + step].expand(
                                         batches, clip_len - non_mask_len, -1,
                                         -1).reshape(batches, -1,
                                                     cur_affinity.size(2))

            if non_mask_len > 0:
                cur_mask = cat([
                    torch.ones(batches, non_mask_len * key_height * key_width,
                               cur_affinity.size(2)).to(cur_mask), cur_mask
                ],
                               dim=1)
            cur_affinity.masked_fill_(~cur_mask.bool(), float('-inf'))
        if topk is not None:
            # [N, topk, step]
            topk_affinity, topk_indices = cur_affinity.topk(k=topk, dim=1)
            # cur_affinity, idx = cur_affinity.sort(descending=True, dim=1)
            # topk_affinity, topk_indices = cur_affinity[:, :topk], idx[:,
            # :topk]
            topk_value = value_vec.transpose(0, 1).reshape(
                C, -1).index_select(
                    dim=1, index=topk_indices.reshape(-1))
            # [N, C, topk, step]
            topk_value = topk_value.reshape(C,
                                            *topk_indices.shape).transpose(
                                                           0, 1)

            if mode == 'softmax':
                topk_affinity = topk_affinity.softmax(dim=1)
            elif mode == 'cosine':
                topk_affinity = topk_affinity.clamp(min=0)**2
            else:
                raise ValueError
            cur_output = torch.einsum('bcks,bks->bcs', topk_value,
                                      topk_affinity)
        else:
            if mode == 'softmax':
                cur_affinity = cur_affinity.softmax(dim=1)
            elif mode == 'cosine':
                cur_affinity = cur_affinity.clamp(min=0)**2
            else:
                raise ValueError
            cur_output = torch.einsum('bck,bks->bcs', value_vec, cur_affinity)
        output[..., ptr:ptr + step] = cur_output

    output = output.reshape(batches, C, query_height,
                            query_width)

    return output


def masked_attention_efficient_v2(query,
                               key,
                               value,
                               radius,
                               temperature=1,
                               topk=None,
                               normalize=True,
                               step=32,
                               non_mask_len=0,
                               mode='softmax',
                               sim_mode='dot_product'):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, T, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, T, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    assert mode in ['softmax', 'cosine']
    batches = query.size(0)
    assert query.size(0) == key.size(0) == value.size(0)
    assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    if key.ndim == 4:
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
    assert value.ndim == key.ndim == 5
    clip_len = key.size(2)
    assert 0 <= non_mask_len < clip_len
    # assert query.shape[2:] == key.shape[3:]
    att_channels, query_height, query_width = query.shape[1:]
    key_height, key_width = key.shape[3:]
    C = value.size(1)
    if normalize:
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)
    query_vec = query.view(batches, att_channels, query.shape[2:].numel())
    key_vec = key.view(batches, att_channels, key.shape[2:].numel())
    value_vec = value.view(batches, C, value.shape[2:].numel())
    output = torch.zeros(batches, C,
                         query_height * query_width).to(query)
    if step is None:
        step = query_height * query_width
        
    grid_x, grid_y = torch.meshgrid(
            torch.arange(query_height, device=query.device),
            torch.arange(query_width, device=query.device))
    
    grid_x_flat = grid_x.reshape(-1)
    grid_y_flat = grid_y.reshape(-1)
    
        
    for ptr in range(0, query_height * query_width, step):
        # [N, TxHxW, step]
        cur_affinity = torch.einsum('bci,bcj->bij', key_vec,
                                        query_vec[...,
                                                ptr:ptr + step]) / temperature
        
        cur_grid_x = grid_x_flat[ptr:ptr+step]
        cur_grid_y = grid_y_flat[ptr:ptr+step]
        
        s = cur_grid_x.shape[0]
        
        dist_mat = ((cur_grid_x[None,None].repeat(query_height, query_width, 1) -
                     grid_x.unsqueeze(-1).repeat(1, 1, s))**2 +
                    (cur_grid_y[None,None].repeat(query_height, query_width, 1) -
                     grid_y.unsqueeze(-1).repeat(1, 1, s))**2)**0.5
        mask = dist_mat < radius
        mask = mask.view(1, query_height * query_width, s).repeat(1, clip_len, 1)
        
        cur_affinity.masked_fill_(~mask.bool(), float('-inf'))

    
        if topk is not None:
            # [N, topk, step]
            topk_affinity, topk_indices = cur_affinity.topk(k=topk, dim=1)
            # cur_affinity, idx = cur_affinity.sort(descending=True, dim=1)
            # topk_affinity, topk_indices = cur_affinity[:, :topk], idx[:,
            # :topk]
            topk_value = value_vec.transpose(0, 1).reshape(
                C, -1).index_select(
                    dim=1, index=topk_indices.reshape(-1))
            # [N, C, topk, step]
            topk_value = topk_value.reshape(C,
                                            *topk_indices.shape).transpose(
                                                           0, 1)

            if mode == 'softmax':
                topk_affinity = topk_affinity.softmax(dim=1)
            elif mode == 'cosine':
                topk_affinity = topk_affinity.clamp(min=0)**2
            else:
                raise ValueError
            cur_output = torch.einsum('bcks,bks->bcs', topk_value,
                                      topk_affinity)
        else:
            if mode == 'softmax':
                cur_affinity = cur_affinity.softmax(dim=1)
            elif mode == 'cosine':
                cur_affinity = cur_affinity.clamp(min=0)**2
            else:
                raise ValueError
            cur_output = torch.einsum('bck,bks->bcs', value_vec, cur_affinity)
        output[..., ptr:ptr + step] = cur_output

    output = output.reshape(batches, C, query_height,
                            query_width)

    return output



def flow_guided_attention_efficient(
                               preds,
                               corrs,
                               value,
                               sample_fn,
                               topk=None,
                               step=32,
                               radius=6,
                               temperature=0.07,
                               mode='softmax',
                               ):
    """

    Args:
        query(torch.Tensor): Value tensor, shape (T, C, H, W)
        key (torch.Tensor): Value tensor, shape (T, C, H, W)
        value (torch.Tensor): Value tensor, shape (T, C, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    L, C, H, W = value.shape

    if step is None:
        step = H * W

    output = torch.zeros(1, C,
                         H * W).to(value)

    for ptr in range(0, H*W, step):
        s = min(H*W - ptr, step)
        affinity = torch.zeros(L, (radius*2 +1)**2,
                            s).to(value)
        value_ = []

        t_start = 0
        for pred, corr in zip(preds, corrs):

            t = pred.shape[0]
            value_feat = value[t_start:t_start+t]

            xx = torch.arange(0, W, device=pred.device)
            yy = torch.arange(0, H, device=pred.device)
            # L' x 2 x H x W
            grid = coords_grid(t, xx, yy) + pred
            grid = grid.permute(0, 2, 3, 1)
            grid = grid.flatten(1,2)

            corr = corr.reshape(t, H*W, 1, H, W)
            # L' x S x 2
            g = grid[:, ptr:ptr + step]
            c = corr[:, ptr:ptr + step]
            # L'S x 1 x H x W
            c = c.flatten(0,1)

            # L' x r^2 x S
            cur_affinity = sample_fn([c], flow=None, grid=g)
            affinity[t_start:t_start+t, ...] = cur_affinity

            # for valune
            # L'S x C x H x W 
            v = value_feat.repeat(1, s, 1, 1, 1).flatten(0,1)
            # 1 x L' x C x r^2 x S
            ref_v = sample_fn([v], flow=None, mode='nearest', grid=g).reshape(t, C, -1, s).unsqueeze(0)
            value_.append(ref_v)

            t_start += t

        if topk is not None:
            # 1 x (L' x r^2) x S
            affinity = affinity.reshape(1, -1, s)

            # [1, topk, S]
            topk_affinity, topk_indices = affinity.topk(k=topk, dim=1)
            
            # 1 x C x (L x r^2) x S
            value_ = torch.cat(value_, 1).transpose(1,2).flatten(2,3)

            topk_value = value_.transpose(0, 1).reshape(
                C, -1).index_select(
                    dim=1, index=topk_indices.reshape(-1))
            # [N, C, topk, step]
            topk_value = topk_value.reshape(C,
                                            *topk_indices.shape).transpose(
                                                0, 1)
            if mode == 'softmax':
                topk_affinity = topk_affinity.softmax(dim=1)
            elif mode == 'cosine':
                topk_affinity = topk_affinity.clamp(min=0)**2
            else:
                raise ValueError

            cur_output = torch.einsum('bcks,bks->bcs', topk_value,
                                    topk_affinity)

        output[...,ptr:ptr+step] = cur_output
        
    return output

def flow_guided_attention_efficient_v2(corr,
                                value,
                                pred,
                                sample_fn,
                                topk=10,
                                step=32,
                                mode='softmax',
                                zero_flow=False,
                                boundary_clip=False,
                                ):
    """

    Args:
        query(torch.Tensor): Value tensor, shape (T, C, H, W)
        key (torch.Tensor): Value tensor, shape (T, C, H, W)
        value (torch.Tensor): Value tensor, shape (T, C, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    L, C, H, W = value.shape

    if step is None:
        step = H * W

    output = torch.zeros(1, C,
                         H * W).to(value)

    xx = torch.arange(0, W, device=pred.device)
    yy = torch.arange(0, H, device=pred.device)

    # check err
    if zero_flow:
        pred = 0

    # L x 2 x H x W
    grid = coords_grid(L, xx, yy) + pred
    grid = grid.permute(0, 2, 3, 1)

    if boundary_clip:
        grid[:,:,:,0] = torch.clamp(grid[:,:,:,0], 0, H-1)
        grid[:,:,:,1] = torch.clamp(grid[:,:,:,1], 0, W-1)

    grid = grid.flatten(1,2)
    corr = corr.reshape(L, H*W, 1, H, W)
    
    for ptr in range(0, H*W, step):
        s = min(H*W - ptr, step)

        # L x S x 2
        g = grid[:, ptr:ptr + step]

        c = corr[:, ptr:ptr + step]
        # LS x 1 x H x W
        c = c.flatten(0,1)

        # L x r^2 x S
        cur_affinity = sample_fn([c], grid=g, mask_oob=True, shape=(W,H))
        
        # 1 x (L x r^2) x S
        cur_affinity = cur_affinity.reshape(1, -1, s)

        v = value.unsqueeze(1).repeat(1, s, 1, 1, 1) # fix the bug
        v = v.flatten(0,1)

        # 1 x L x C x r^2 x S
        value_ = sample_fn([v], mode='bilinear', grid=g).reshape(L, C, -1, s).unsqueeze(0)
        # 1 x C x (L x r^2) x S
        value_ = value_.transpose(1,2).flatten(2,3)

        if topk is not None:
            
            # [1, topk, S]
            topk_affinity, topk_indices = cur_affinity.topk(k=topk, dim=1)
            # 1 x 3 x topk x S
            topk_value = torch.gather(value_, dim=2, index=topk_indices.repeat(1,C,1,1))

            if mode == 'softmax':
                topk_affinity = topk_affinity.softmax(dim=1)
            elif mode == 'cosine':
                topk_affinity = topk_affinity.clamp(min=0)**2
            else:
                raise ValueError

            cur_output = torch.einsum('bcks,bks->bcs', topk_value,
                                    topk_affinity)

        else:
            cur_affinity = cur_affinity.softmax(dim=1)
            cur_output = torch.einsum('bcks,bks->bcs', value_,
                                    cur_affinity)


        output[...,ptr:ptr+step] = cur_output

    return output


def masked_attention_efficient_c2f(query,
                               key,
                               query_fine,
                               key_fine,
                               value,
                               mask,
                               temperature=1,
                               topk=None,
                               normalize=True,
                               step=32,
                               non_mask_len=0,
                               mode='softmax',
                               sim_mode='dot_product',
                               radius_fine=12):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, T, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, T, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    assert mode in ['softmax', 'cosine']
    batches = query.size(0)
    assert query.size(0) == key.size(0) == value.size(0)
    # assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    if key.ndim == 4:
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
    assert value.ndim == key.ndim == 5
    clip_len = key.size(2)
    assert 0 <= non_mask_len < clip_len
    # assert query.shape[2:] == key.shape[3:]
    att_channels, query_height, query_width = query.shape[1:]
    att_channels_fine, query_height_fine, query_width_fine = query_fine.shape[1:]
    
    key_height, key_width = key.shape[3:]
    key_height_fine, key_width_fine = key_fine.shape[3:]
    
    C = value.size(1)
    scale = key_height_fine // key_height
    
    if normalize:
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)
        query_fine = F.normalize(query_fine, p=2, dim=1)
        key_fine = F.normalize(key_fine, p=2, dim=1)
        
    query_vec = query.view(batches, att_channels, query.shape[2:].numel())
    key_vec = key.view(batches, att_channels, key.shape[2:].numel())
    value_vec = value.view(batches, C, value.shape[2:].numel())

    output = torch.zeros(batches, C,
                         query_height * query_width).to(query)
    
    # [N, C, H*W]
    query_fine = query_fine[:,:,::scale,::scale].reshape(batches, att_channels_fine, -1)

    # query_fine = F.interpolate(query_fine, (query_height, query_width)).flatten(-2)
    
    # [N*T, C, R^2, H*W]
    key_fine_unfold = F.unfold(key_fine.transpose(1,2).flatten(0,1), kernel_size=(2*radius_fine+1, 2*radius_fine+1), padding=radius_fine, stride=scale).reshape(batches*clip_len, att_channels_fine, -1, key_height*key_width)
    
    # [NT, 3, R^2, H*W]
    value_unfold = F.unfold(value.transpose(1,2).flatten(0,1), kernel_size=(2*radius_fine+1, 2*radius_fine+1), padding=radius_fine, stride=scale).reshape(batches*clip_len, C, -1, key_height*key_width)

    
    
    if step is None:
        step = query_height * query_width
    
    for ptr in range(0, query_height * query_width, step):
        # [N, TxHxW, step]
        s = min(query_height * query_width- ptr, step)
        
        cur_affinity = torch.einsum('bci,bcj->bij', key_vec,
                                    query_vec[...,
                                            ptr:ptr + s]) / temperature
        
        if mask is not None:
            if mask.ndim == 2:
                assert mask.shape == (key_height * key_width,
                                      query_height * query_width)
                cur_mask = mask.view(1, 1, key_height * key_width,
                                     query_height *
                                     query_width)[..., ptr:ptr + s].expand(
                                         batches, clip_len - non_mask_len, -1,
                                         -1).reshape(batches, -1,
                                                     cur_affinity.size(2))
            else:
                cur_mask = mask.view(1, 1, key_height * key_width,
                                     query_height *
                                     query_width)[..., ptr:ptr + s].expand(
                                         batches, clip_len - non_mask_len, -1,
                                         -1).reshape(batches, -1,
                                                     cur_affinity.size(2))

            if non_mask_len > 0:
                cur_mask = cat([
                    torch.ones(batches, non_mask_len * key_height * key_width,
                               cur_affinity.size(2)).to(cur_mask), cur_mask
                ],
                               dim=1)
            cur_affinity.masked_fill_(~cur_mask.bool(), float('-inf'))
        
        # [N, T, HxW, s]
        cur_affinity = cur_affinity.reshape(batches, clip_len, -1, s).softmax(-2)
        # [NT, 1, 1, s]
        idxs = torch.argmax(cur_affinity, dim=-2).flatten(0,1)[:,None,None,:]
        
        # [NT, C, R^2, s]
        i = idxs.repeat(1, *key_fine_unfold.shape[1:3], 1)
        cur_key_fine_unfold = torch.gather(key_fine_unfold, dim=-1, index=i)
        
        # [NT, C, R^2, s]
        cur_query_fine = query_fine[...,ptr:ptr + s].unsqueeze(2).repeat(clip_len, 1, key_fine_unfold.shape[2], 1)

        # [NT, R^2, s]
        affinity = (cur_key_fine_unfold * cur_query_fine).sum(1) / temperature
        
        # [N, TR^2, s]
        affinity = affinity.reshape(batches, -1, s)
        
        # [N, T, 3, R^2, s]
        cur_value_unfold = torch.gather(value_unfold, dim=-1, index=idxs.repeat(1, C, key_fine_unfold.shape[2], 1)).reshape(batches, clip_len, C, -1, s)
        # [N, 3, TR^2, s]
        cur_value_unfold = cur_value_unfold.transpose(1,2).flatten(2,3)
        
        if topk is not None:
            # [N, topk, s]
            topk_affinity, topk_indices = affinity.topk(k=topk, dim=1)

            # [N, 3, topk, s]
            topk_value = torch.gather(cur_value_unfold, dim=-2, index=topk_indices.unsqueeze(1).repeat(1,C,1,1))

            if mode == 'softmax':
                topk_affinity = topk_affinity.softmax(dim=1)
            elif mode == 'cosine':
                topk_affinity = topk_affinity.clamp(min=0)**2
            else:
                raise ValueError
            cur_output = torch.einsum('bcks,bks->bcs', topk_value,
                                      topk_affinity)
        else:
            pass
        
        output[..., ptr:ptr + s] = cur_output

    output = output.reshape(batches, C, query_height,
                            query_width)

    return output


def masked_attention_efficient_correlation(query_frame,
                               key_frames,
                               value,
                               radius, 
                               corr_infer,
                               feat_extractor,
                               temperature=1,
                               topk=None,
                               normalize=True,
                               sstep=32,
                               tstep=5,
                                ):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, T, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, T, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    B = query_frame.size(0)


    clip_len = key_frames.size(2)


    # print(sstep, tstep)
    query = feat_extractor(query_frame)
    if normalize:
        query = F.normalize(query, p=2, dim=1)
    
    
    
    C_F, H, W = query.shape[1:]
    C = value.size(1)
    L = 2*radius+1

    # query_vec = query.view(B, C_F, query.shape[2:].numel())
    # B x HW x 2
    xx = torch.arange(0, W, device=query.device)
    yy = torch.arange(0, H, device=query.device)
    coords = coords_grid(B, xx, yy) + radius # shape N, 2, H, W
    coords = coords.permute(0, 2, 3, 1).flatten(1,2)  # shape N, H, W, 2
    
    
    output = torch.zeros(B, C, H * W).to(query)
    for ptr in range(0, H * W, sstep):
        # B x C x sstep
        values = []
        affs = []
        
        # query_vec_n = query_vec[..., ptr:ptr + sstep]
        for ptrt in range(0, clip_len, tstep):
            act = min(clip_len, tstep + ptrt) - ptrt
    
            keys = key_frames[:,:,ptrt:ptrt+tstep].transpose(1,2).flatten(0,1)
            key_vec = feat_extractor(keys)
            
            if normalize:
                key_vec = F.normalize(key_vec, p=2, dim=1)
                
            value_vec = value[:,:,ptrt:ptrt+tstep].transpose(1,2).flatten(0,1)
            
            # K x R^2 x HW
            corr_up = corr_infer(query.repeat(act, 1, 1, 1), key_vec).flatten(1,2).flatten(-2)
            # B x (K x R^2) x sstep
            cur_affinity = corr_up[..., ptr:ptr+sstep].unsqueeze(0).flatten(1,2)
            affs.append(cur_affinity)
            del corr_up
            
            # B x sstep x 2
            coord = coords[:,ptr:ptr+sstep]
            unfold_v = part_unfold.part_unfold(value_vec, radius, coord, act)
            
             # K x C x R^2 x HW
            unfold_v = unfold_v.permute(0,2,3,1)
        
            # B x C x (K x R^2) x sstep
            cur_v = unfold_v.unsqueeze(0).transpose(1,2).flatten(2,3)
            values.append(cur_v)
            
            del unfold_v
            del key_vec
            del value_vec

        
        # B x M  x sstep
        affs = torch.cat(affs, 1)
        
        # B x C x M x sstep 
        values = torch.cat(values, 2)
        
        
         # 1 x topk x sstep
        topk_affinity, topk_indices = affs.topk(k=topk, dim=1)
        
        # 1 x C x topk x sstep
        topk_value = torch.gather(values, dim=2, index=topk_indices.unsqueeze(1).repeat(1, C, 1, 1))
        
        
        topk_affinity /= temperature
        topk_affinity = topk_affinity.softmax(dim=1)


        cur_output = torch.einsum('bckm,bkm->bcm', topk_value,
                            topk_affinity)
        
    
        output[..., ptr:ptr + sstep] = cur_output
        
        del affs
        del values

    output = output.reshape(B, C, H, W)

    return output


def masked_attention_efficient_corrup(query,
                               key,
                               value,
                               radius, 
                               corr_infer,
                               temperature=1,
                               topk=None,
                               normalize=True,
                               sstep=32,
                               tstep=5,
                                ):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, T, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, T, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    B = query.size(0)
    
    C, clip_len, H, W = value.shape[1:]

    # query_vec = query.view(B, C_F, query.shape[2:].numel())
    # B x HW x 2
    xx = torch.arange(0, W, device=query.device)
    yy = torch.arange(0, H, device=query.device)
    coords = coords_grid(B, xx, yy) + radius # shape N, 2, H, W
    coords = coords.permute(0, 2, 3, 1).flatten(1,2)  # shape N, H, W, 2
    
    output = torch.zeros(B, C, H * W).to(query)
    for ptr in range(0, H * W, sstep):
        # B x C x sstep
        values = []
        affs = []
        
        # query_vec_n = query_vec[..., ptr:ptr + sstep]
        for ptrt in range(0, clip_len, tstep):
            act = min(clip_len, tstep + ptrt) - ptrt

            query_vec = query.repeat(act, 1, 1, 1)
            key_vec = key[:,:,ptrt:ptrt+tstep].transpose(1,2).flatten(0,1)
            value_vec = value[:,:,ptrt:ptrt+tstep].transpose(1,2).flatten(0,1)
            
            # K x R^2 x HW
            corr_up = corr_infer(query_vec, key_vec, h=H*2, w=W*2, use_feat=True).flatten(-2)
            # B x (K x R^2) x sstep
            cur_affinity = corr_up[..., ptr:ptr+sstep].unsqueeze(0).flatten(1,2)
            affs.append(cur_affinity)
            del corr_up
            
            # B x sstep x 2
            coord = coords[:,ptr:ptr+sstep]
            unfold_v = part_unfold.part_unfold(value_vec, radius, coord, act)
            
             # K x C x R^2 x HW
            unfold_v = unfold_v.permute(0,2,3,1)
        
            # B x C x (K x R^2) x sstep
            cur_v = unfold_v.unsqueeze(0).transpose(1,2).flatten(2,3)
            values.append(cur_v)
            del unfold_v

            del key_vec
            del value_vec

        
        # B x M  x sstep
        affs = torch.cat(affs, 1)
        
        # B x C x M x sstep 
        values = torch.cat(values, 2)
        
        
         # 1 x topk x sstep
        topk_affinity, topk_indices = affs.topk(k=topk, dim=1)
        
        # 1 x C x topk x sstep
        topk_value = torch.gather(values, dim=2, index=topk_indices.unsqueeze(1).repeat(1, C, 1, 1))
        
        
        topk_affinity /= temperature
        topk_affinity = topk_affinity.softmax(dim=1)


        cur_output = torch.einsum('bckm,bkm->bcm', topk_value,
                            topk_affinity)
        
    
        output[..., ptr:ptr + sstep] = cur_output
        
        del affs
        del values

    output = output.reshape(B, C, H, W)

    return output



def masked_attention_efficient_correlation_v2(query_frame,
                               key_frames,
                               value,
                               radius, 
                               corr_infer,
                               feat_extractor,
                               temperature=1,
                               topk=None,
                               normalize=True,
                               sstep=32,
                               tstep=5,
                                ):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, T, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, T, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    B = query_frame.size(0)


    clip_len = key_frames.size(2)


    # print(sstep, tstep)
    query = feat_extractor(query_frame)
    if normalize:
        query = F.normalize(query, p=2, dim=1)
    
    
    
    C_F, H, W = query.shape[1:]
    C = value.size(1)
    L = 2*radius+1

    # query_vec = query.view(B, C_F, query.shape[2:].numel())
    # B x HW x 2
    xx = torch.arange(0, W, device=query.device)
    yy = torch.arange(0, H, device=query.device)
    coords = coords_grid(B, xx, yy) + radius # shape N, 2, H, W
    coords = coords.permute(0, 2, 3, 1).flatten(1,2)  # shape N, H, W, 2
    
    
    output = torch.zeros(B, C, H * W).to(query)
    for ptr in range(0, H * W, sstep):
        # B x C x sstep
        values = []
        affs = []
        
        # query_vec_n = query_vec[..., ptr:ptr + sstep]
        for ptrt in range(0, clip_len, tstep):
            act = min(clip_len, tstep + ptrt) - ptrt
    
            keys = key_frames[:,:,ptrt:ptrt+tstep].transpose(1,2).flatten(0,1)
            key_vec = feat_extractor(keys)
            
            if normalize:
                key_vec = F.normalize(key_vec, p=2, dim=1)
                
            # K x C_f x H x W
            value_vec = value[:,:,ptrt:ptrt+tstep].transpose(1,2).flatten(0,1)
            
            # K x C_f x sstep
            query_vec = query.repeat(act, 1, 1, 1).flatten(-2)
            cur_q = query_vec[..., ptr:ptr+sstep]
            
            # K x C_f x R^2 x sstep
            coord = coords[:,ptr:ptr+sstep]
            cur_k = part_unfold.part_unfold(key_vec, radius, coord.clone(), act, mode='bilinear', align_corner=True).permute(0,2,3,1)
            
            # B x (K x R^2) x sstep
            cur_affinity = torch.einsum("kcs,kcls->kls", [cur_q, cur_k]).unsqueeze(0).flatten(1,2)
            
            affs.append(cur_affinity)

            
            # B x sstep x 2
            unfold_v = part_unfold.part_unfold(value_vec, radius, coord.clone(), act)
            
             # K x C x R^2 x sstep
            unfold_v = unfold_v.permute(0,2,3,1)
        
            # B x C x (K x R^2) x sstep
            cur_v = unfold_v.unsqueeze(0).transpose(1,2).flatten(2,3)
            values.append(cur_v)
            
            del unfold_v
            del key_vec
            del value_vec
            del cur_k
            del cur_q
            del cur_affinity

        
        # B x M  x sstep
        affs = torch.cat(affs, 1)
        
        # B x C x M x sstep 
        values = torch.cat(values, 2)
        
        
         # 1 x topk x sstep
        topk_affinity, topk_indices = affs.topk(k=topk, dim=1)
        
        # 1 x C x topk x sstep
        topk_value = torch.gather(values, dim=2, index=topk_indices.unsqueeze(1).repeat(1, C, 1, 1))
        
        
        topk_affinity /= temperature
        topk_affinity = topk_affinity.softmax(dim=1)


        cur_output = torch.einsum('bckm,bkm->bcm', topk_value,
                            topk_affinity)
        
    
        output[..., ptr:ptr + sstep] = cur_output
        
        del affs
        del values

    output = output.reshape(B, C, H, W)

    return output



if __name__ == '__main__':
    import time
    s = 128
    b = 1

    q = torch.rand((b, 256, s, s)).cuda()
    k = torch.rand((b, 256, 20, s, s)).cuda()
    v = torch.rand((b, 3, 20, s, s)).cuda()
    mask = torch.ones((s*s,s*s)).cuda()

    start = time.time()
    # _ = masked_attention_efficient(q, k, v, mask, topk=10, step=32)
    _ = masked_attention_efficient_v2(q, k, v, topk=10, radius=24)


    print(time.time()-start)