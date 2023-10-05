from typing import List

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single, _triple

def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.
    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def change_stride(conv, stride):
    """Inplace change conv stride.

    Args:
        conv (nn.Module):
        stride (int):
    """
    if isinstance(conv, nn.Conv1d):
        conv.stride = _single(stride)
    if isinstance(conv, nn.Conv2d):
        conv.stride = _pair(stride)
    if isinstance(conv, nn.Conv3d):
        conv.stride = _triple(stride)


def pil_nearest_interpolate(input, size):
    # workaround for https://github.com/pytorch/pytorch/issues/34808
    resized_imgs = []
    input = input.permute(0, 2, 3, 1)
    for img in input:
        img = img.squeeze(-1)
        img = img.detach().cpu().numpy()
        resized_img = mmcv.imresize(
            img,
            size=(size[1], size[0]),
            interpolation='nearest',
            backend='pillow')
        resized_img = torch.from_numpy(resized_img).to(
            input, non_blocking=True)
        resized_img = resized_img.unsqueeze(2).permute(2, 0, 1)
        resized_imgs.append(resized_img)

    return torch.stack(resized_imgs, dim=0)


def video2images(imgs):
    batches, channels, clip_len = imgs.shape[:3]
    if clip_len == 1:
        new_imgs = imgs.squeeze(2).reshape(batches, channels, *imgs.shape[3:])
    else:
        new_imgs = imgs.transpose(1, 2).contiguous().reshape(
            batches * clip_len, channels, *imgs.shape[3:])

    return new_imgs


def images2video(imgs, clip_len):
    batches, channels = imgs.shape[:2]
    if clip_len == 1:
        new_imgs = imgs.unsqueeze(2)
    else:
        new_imgs = imgs.reshape(batches // clip_len, clip_len, channels,
                                *imgs.shape[2:]).transpose(1, 2).contiguous()

    return new_imgs


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class StrideContext(object):

    def __init__(self, backbone, strides, out_indices=None):
        self.backbone = backbone
        self.strides = strides
        self.out_indices = out_indices

    def __enter__(self):
        if self.strides is not None:
            self.backbone.switch_strides(self.strides)
        if self.out_indices is not None:
            self.backbone.switch_out_indices(self.out_indices)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.strides is not None:
            self.backbone.switch_strides()
        if self.out_indices is not None:
            self.backbone.switch_out_indices()


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


@torch.no_grad()
def _batch_shuffle_ddp(x):
    """Batch shuffle, for making use of BatchNorm.

    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def _batch_unshuffle_ddp(x, idx_unshuffle):
    """Undo batch shuffle.

    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]


class Clamp(nn.Module):

    def __init__(self, min=None, max=None):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max
        assert self.min is not None or self.max is not None

    def forward(self, x):
        kwargs = {}
        if self.min is not None:
            kwargs['min'] = self.min
        if self.max is not None:
            kwargs['max'] = self.max
        return x.clamp(**kwargs)

    def extra_repr(self):
        """Extra repr."""
        s = f'min={self.min}, max={self.max}'
        return s


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """Efficient version of torch.cat that avoids a copy if there is only a
    single element in a list."""
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def normalize_logit(seg_logit):
    seg_logit_min = seg_logit.view(*seg_logit.shape[:2], -1).min(
        dim=-1)[0].view(*seg_logit.shape[:2], 1, 1)
    seg_logit_max = seg_logit.view(*seg_logit.shape[:2], -1).max(
        dim=-1)[0].view(*seg_logit.shape[:2], 1, 1)
    normalized_seg_logit = (seg_logit - seg_logit_min) / (
        seg_logit_max - seg_logit_min + 1e-12)
    seg_logit = torch.where(seg_logit_max > 0, normalized_seg_logit, seg_logit)

    return seg_logit


def mean_list(input_list):
    ret = input_list[0].clone()
    for i in range(1, len(input_list)):
        ret += input_list[i]
    ret /= len(input_list)
    return ret


def interpolate3d(input,
                  size=None,
                  scale_factor=None,
                  mode='nearest',
                  align_corners=False):
    results = []
    clip_len = input.size(2)
    for i in range(clip_len):
        results.append(
            F.interpolate(
                input[:, :, i],
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners))

    return torch.stack(results, dim=2)


def image_meshgrid_from(x):
    # input: b,c,h,w
    # output: b,c,h,2
    shape = x.shape  # assume b,c,h,w
    _y, _x = torch.meshgrid(torch.arange(shape[2]), torch.arange(shape[3]))
    grid = torch.stack([_x, _y], dim=-1)
    return torch.stack([grid] * shape[0], dim=0).type(x.type()).to(x.device)


def normalize_meshgrid(grid):
    # normalize wrt to image size
    # input: b,h,w,2
    # output: b,h,w,2 (range = [-1,1])
    grid_new = torch.zeros_like(grid)
    b, h, w, _ = grid.shape
    grid_new[..., 0] = grid[..., 0] / (w - 1) * 2 - 1
    grid_new[..., 1] = grid[..., 1] / (h - 1) * 2 - 1
    return grid_new



def deform_im2col(im, offset, kernel_size=3):
    # Faster on gpu, slower on CPU
    # input: b,c,h,w
    # output: b,N*c,h*w
    with torch.no_grad():
        grid = image_meshgrid_from(im)
        b, c, h, w = im.shape

    N = kernel_size * kernel_size

    grid_ = torch.zeros(b * N, h, w, 2,  device=im.device).contiguous()
    im_ = im.repeat(N, 1, 1, 1)

    for dy in range(kernel_size):
        for dx in range(kernel_size):
            grid_[(dy * kernel_size + dx) * b:(dy * kernel_size + dx + 1) * b] =\
                grid + offset + torch.tensor([dx - kernel_size // 2, dy - kernel_size // 2])[None, None, None, :].float().to(im.device)

    out = F.grid_sample(im_.contiguous(), normalize_meshgrid(grid_).contiguous())
    out = out.reshape(N, b, c, h * w).permute(1,2,0,3)

    return out.reshape(b, kernel_size * kernel_size * c, h * w)


def one_hot(labels, C):
    one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3))
    if labels.is_cuda: one_hot = one_hot.cuda()

    target = one_hot.scatter_(1, labels, 1)
    if labels.is_cuda: target = target.cuda()

    return target


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)

    
def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]

def make_mask(size, t_size, eq=True):
    
    if isinstance(size, tuple):
        size_ = size[1]
        size = size[0]
    else:
        size_ = size
    
    size = int(size)
    t_size = int(t_size)
        
    masks = []
    for i in range(size):
        for j in range(size):
            mask = torch.zeros((size_, size_)).cuda()
            if eq:
                mask[max(0, i-t_size):min(size_, i+t_size+1), max(0, j-t_size):min(size_, j+t_size+1)] = 1
            else:
                mask[max(0, i-t_size):min(size_, i+t_size+1), max(0, j-t_size):min(size_, j+t_size+1)] = 0.7
                mask[i,j] = 1
                
            masks.append(mask.reshape(-1))
    return torch.stack(masks)


def pad_divide_by(in_img, d):
    h, w = in_img.shape[-2:]

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array

def unpad(img, pad):
    if img.dim() == 3:
        if pad[2]+pad[3] > 0:
            img = img[:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            img = img[:,:,pad[0]:-pad[1]]
    elif img.dim() == 4:
        if pad[2]+pad[3] > 0:
                img = img[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            img = img[:,:,:,pad[0]:-pad[1]]
        
    return img

def norm_mask(mask):
    b, c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[:,cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[:,cnt,:,:] = mask_cnt
    return mask