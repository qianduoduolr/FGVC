import random
import copy
from collections.abc import Sequence
import torch
import numbers
from typing import Tuple, List, Optional
from torch import Tensor
import math

import cv2
import mmcv
import numpy as np
from numpy import random as npr
from PIL import Image, ImageFilter
from skimage.util import view_as_windows
from scipy.ndimage.filters import maximum_filter
from scipy import signal
import scipy.ndimage as ndimage

from torch.nn.modules.utils import _pair
from torchvision.transforms import ColorJitter as _ColorJitter
from torchvision.transforms import RandomAffine as _RandomAffine
from torchvision.transforms import RandomResizedCrop as _RandomResizedCrop
from torchvision.transforms import functional as F

# from mmpt.utils.visualize import affanity

from ..registry import PIPELINES


def _init_lazy_if_proper(results, lazy, keys='imgs'):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results[keys][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


@PIPELINES.register_module()
class Fuse(object):
    """Fuse lazy operations.

    Fusion order:
        crop -> resize -> flip

    Required keys are "imgs", "img_shape" and "lazy", added or modified keys
    are "imgs", "lazy".
    Required keys in "lazy" are "crop_bbox", "interpolation", "flip_direction".
    """
    def __init__(self, keys='imgs'):
        self.keys = keys

    def __call__(self, results):
        if 'lazy' not in results:
            raise ValueError('No lazy operation detected')
        lazyop = results['lazy']
        imgs = results[self.keys]

        # crop
        left, top, right, bottom = lazyop['crop_bbox'].round().astype(int)
        imgs = [img[top:bottom, left:right] for img in imgs]

        # resize
        img_h, img_w = results['img_shape']
        if lazyop['interpolation'] is None:
            interpolation = 'bilinear'
        else:
            interpolation = lazyop['interpolation']
        imgs = [
            mmcv.imresize(img, (img_w, img_h), interpolation=interpolation)
            for img in imgs
        ]

        # flip
        if lazyop['flip']:
            for img in imgs:
                mmcv.imflip_(img, lazyop['flip_direction'])

        results[self.keys] = imgs
        del results['lazy']

        return results


@PIPELINES.register_module()
class RandomCrop(object):
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "imgs" and "img_shape", added or
    modified keys are "imgs", "lazy"; Required keys in "lazy" are "flip",
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False, keys='imgs'):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy
        self.keys = keys

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy, self.keys)

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        new_h, new_w = self.size, self.size

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results[self.keys] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results[self.keys]
            ]
            try:
                flow = results['flow']
                results['flow'] = flow[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
            except Exception as e:
                pass
            
            try:
                flow = results['flow_back']
                results['flow_back'] = flow[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
            except Exception as e:
                pass
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class RandomResizedCrop(object):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "imgs", "img_shape", "crop_bbox" and "lazy",
    added or modified keys are "imgs", "crop_bbox" and "lazy"; Required keys
    in "lazy" are "flip", "crop_bbox", added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 same_on_clip=True,
                 same_across_clip=False,
                 same_clip_indices=None,
                 same_frame_indices=None,
                 crop_ratio=0.9,
                 lazy=False,
                 keys='imgs',
                 with_bbox=False):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        if not mmcv.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmcv.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip
        if same_clip_indices is not None:
            assert isinstance(same_clip_indices, Sequence)
        self.same_clip_indices = same_clip_indices
        if same_frame_indices is not None:
            assert isinstance(same_frame_indices, Sequence)
        self.same_frame_indices = same_frame_indices
        self.crop_ratio = crop_ratio
        self.keys = keys
        self.with_bbox = with_bbox
    

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      bbox=None,
                      crop_ratio=None,
                      max_attempts=20):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        def calc_over_lab(gt, gen):
            out = 1
            for i in range(2):
                z_min = max(gt[i], gen[i])
                z_max = min(gt[i+2], gen[i+2])
                if z_min >= z_max:
                    return 0
                else:
                    out *= (z_max - z_min) / (gt[i+2] - gt[i])
            return out

        max_attempts = 40 if bbox != None else 20

        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        crop_list = []
        ratio_list = []

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                if bbox != None:
                    overlap = calc_over_lab(bbox, [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h])
                    if overlap >= crop_ratio:
                        return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h
                    else:
                        crop_list.append([x_offset, y_offset, x_offset + crop_w, y_offset + crop_h])
                        ratio_list.append(overlap)
                else:
                    return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        if bbox != None:
            if len(ratio_list) >= 1:
                idx = np.argsort(np.array(ratio_list))[-1]
                return crop_list[idx]
            else:
                left, top, right, bottom = random.randint(0, bbox[0] - 1), random.randint(0, bbox[1] - 1), random.randint(bbox[2], img_w - 1),\
                    random.randint(bbox[3], img_h - 1)
                return  left, top, right, bottom
        
        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy, self.keys)

        img_h, img_w = results['img_shape']
        
        if results.get('bboxs', None) and self.with_bbox:
            bbox = results['bboxs'][0]
            results['crop_bbox_ratio'] = []
        else:
            bbox = None

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range, bbox=bbox, crop_ratio=self.crop_ratio)

        new_h, new_w = bottom - top, right - left

        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            for i, img in enumerate(results[self.keys]):
                is_new_clip = not self.same_across_clip and i % results[
                    'clip_len'] == 0 and i > 0
                generate_new = not self.same_on_clip or is_new_clip
                if self.same_clip_indices is not None:
                    assert min(self.same_clip_indices) >= 0
                    assert max(self.same_clip_indices) < results['num_clips']
                    keep_same = i // results[
                        'clip_len'] in self.same_clip_indices
                    generate_new = generate_new and not keep_same
                if self.same_frame_indices is not None:
                    assert min(self.same_frame_indices) >= 0
                    assert max(self.same_frame_indices) < results['clip_len']
                    keep_same = i % results[
                        'clip_len'] in self.same_frame_indices
                    generate_new = generate_new and not keep_same
                if generate_new:
                    if results.get('bboxs', None) and self.with_bbox:
                        bbox = results['bboxs'][i]
                    else:
                        bbox = None
                    left, top, right, bottom = self.get_crop_bbox(
                        (img_h, img_w), self.area_range,
                        self.aspect_ratio_range, bbox=bbox, crop_ratio=self.crop_ratio)
                    new_h, new_w = bottom - top, right - left

                results['crop_bbox'] = np.array([left, top, right, bottom])
                results['img_shape'] = (new_h, new_w)
                results[self.keys][i] = img[top:bottom, left:right]
                if results.get('flows', False):
                    flow = results['flows'][i]
                    results['flows'][i] = flow[top:bottom, left:right]

                if results.get('bboxs', None) and self.with_bbox:
                    results['crop_bbox_ratio'].append(self.get_ratio(results, left, top, new_w, new_h, i))
                
                if 'grids' in results:
                    grid = results['grids'][i]
                    results['grids'][i] = grid[top:bottom, left:right]

                if results.get('masks', None):
                    mask = results['masks'][i]
                    results['masks'][i] = mask[top:bottom, left:right]
            
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)
        return results

    def get_ratio(self, results, left, top, new_w, new_h, idx):
        return [max((results['bboxs'][idx][0] - left)/new_w, 0), max((results['bboxs'][idx][1] - top)/new_h, 0), \
            min((results['bboxs'][idx][2] - left)/new_w, 1), min((results['bboxs'][idx][3] - top)/new_h, 1)]

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class MultiScaleCrop(object):
    """Crop images with a list of randomly selected scales.

    Randomly select the w and h scales from a list of scales. Scale of 1 means
    the base size, which is the minimal of image weight and height. The scale
    level of w and h is controlled to be smaller than a certain value to
    prevent too large or small aspect ratio.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox", "img_shape", "lazy" and "scales". Required keys in "lazy" are
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        input_size (int | tuple[int]): (w, h) of network input.
        scales (tuple[float]): Weight and height scales to be selected.
        max_wh_scale_gap (int): Maximum gap of w and h scale levels.
            Default: 1.
        random_crop (bool): If set to True, the cropping bbox will be randomly
            sampled, otherwise it will be sampler from fixed regions.
            Default: False.
        num_fixed_crops (int): If set to 5, the cropping bbox will keep 5
            basic fixed regions: "upper left", "upper right", "lower left",
            "lower right", "center".If set to 13, the cropping bbox will append
            another 8 fix regions: "center left", "center right",
            "lower center", "upper center", "upper left quarter",
            "upper right quarter", "lower left quarter", "lower right quarter".
            Default: 5.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 input_size,
                 scales=(1, ),
                 max_wh_scale_gap=1,
                 random_crop=False,
                 num_fixed_crops=5,
                 lazy=False,
                 keys='imgs'):
        self.input_size = _pair(input_size)
        if not mmcv.is_tuple_of(self.input_size, int):
            raise TypeError(f'Input_size must be int or tuple of int, '
                            f'but got {type(input_size)}')

        if not isinstance(scales, tuple):
            raise TypeError(f'Scales must be tuple, but got {type(scales)}')

        if num_fixed_crops not in [5, 13]:
            raise ValueError(f'Num_fix_crops must be in {[5, 13]}, '
                             f'but got {num_fixed_crops}')

        self.scales = scales
        self.max_wh_scale_gap = max_wh_scale_gap
        self.random_crop = random_crop
        self.num_fixed_crops = num_fixed_crops
        self.lazy = lazy
        self.keys = keys

    def __call__(self, results):
        """Performs the MultiScaleCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy, self.keys)

        img_h, img_w = results['img_shape']
        base_size = min(img_h, img_w)
        crop_sizes = [int(base_size * s) for s in self.scales]

        candidate_sizes = []
        for i, h in enumerate(crop_sizes):
            for j, w in enumerate(crop_sizes):
                if abs(i - j) <= self.max_wh_scale_gap:
                    candidate_sizes.append([w, h])

        crop_size = random.choice(candidate_sizes)
        for i in range(2):
            if abs(crop_size[i] - self.input_size[i]) < 3:
                crop_size[i] = self.input_size[i]

        crop_w, crop_h = crop_size

        if self.random_crop:
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)
        else:
            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4
            candidate_offsets = [
                (0, 0),  # upper left
                (4 * w_step, 0),  # upper right
                (0, 4 * h_step),  # lower left
                (4 * w_step, 4 * h_step),  # lower right
                (2 * w_step, 2 * h_step),  # center
            ]
            if self.num_fixed_crops == 13:
                extra_candidate_offsets = [
                    (0, 2 * h_step),  # center left
                    (4 * w_step, 2 * h_step),  # center right
                    (2 * w_step, 4 * h_step),  # lower center
                    (2 * w_step, 0 * h_step),  # upper center
                    (1 * w_step, 1 * h_step),  # upper left quarter
                    (3 * w_step, 1 * h_step),  # upper right quarter
                    (1 * w_step, 3 * h_step),  # lower left quarter
                    (3 * w_step, 3 * h_step)  # lower right quarter
                ]
                candidate_offsets.extend(extra_candidate_offsets)
            x_offset, y_offset = random.choice(candidate_offsets)

        new_h, new_w = crop_h, crop_w

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['img_shape'] = (new_h, new_w)
        results['scales'] = self.scales

        if not self.lazy:
            results[self.keys] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results[self.keys]
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'input_size={self.input_size}, scales={self.scales}, '
                    f'max_wh_scale_gap={self.max_wh_scale_gap}, '
                    f'random_crop={self.random_crop}, '
                    f'num_fixed_crops={self.num_fixed_crops}, '
                    f'lazy={self.lazy})')
        return repr_str

@PIPELINES.register_module()
class RandomScaleCrop(object):

    def __init__(self,
                 scale_range=(0.5, 1.0),
                 identity=False,
                 same_on_clip=True,
                 same_across_clip=False,
                 same_clip_indices=None,
                 same_frame_indices=None,
                 center_crop=False,
                 crop_size=64,
                 keys='imgs',
                 ):
        self.scale_range = scale_range

        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip
        if same_clip_indices is not None:
            assert isinstance(same_clip_indices, Sequence)
        self.same_clip_indices = same_clip_indices
        if same_frame_indices is not None:
            assert isinstance(same_frame_indices, Sequence)
        self.same_frame_indices = same_frame_indices
        self.center_crop = center_crop
        self.crop_size = crop_size
        self.keys = keys
        self.identity = identity
    
    @staticmethod
    def get_params(h, w, new_scale):
        # generating random crop
        # preserves aspect ratio
        new_h = int(new_scale * h)
        new_w = int(new_scale * w)

        # generating 
        if new_scale <= 1.:
            assert w >= new_w and h >= new_h, "{} vs. {} | {} / {}".format(w, new_w, h, new_h)
            i = random.randint(0, h - new_h)
            j = random.randint(0, w - new_w)
        else:
            assert w <= new_w and h <= new_h, "{} vs. {} | {} / {}".format(w, new_w, h, new_h)
            i = random.randint(h - new_h, 0)
            j = random.randint(w - new_w, 0)

        return i, j, new_h, new_w
    
    def get_scale(self, size=None):
        if not self.center_crop:
            return random.uniform(self.scale_range[0], self.scale_range[1])
        else:
            return (size - self.crop_size * 2) / size

    
    def __call__(self, results):
    
        if results.get('affine', True):
            results[f'affine_params_{self.keys}'] = [[0.,0.,0.,1.,1.] for _ in  results[self.keys]]

        if self.identity:
            return results
        
        H, W, _ =  results[self.keys][0].shape

        i2 = H / 2
        j2 = W / 2

        masks_new = []
        
        # one crop for all
        if not self.center_crop:
            s = self.get_scale()
            ii, jj, h, w = self.get_params(H, W, s)
            # displacement of the centre
            dy = ii + h / 2 - i2
            dx = jj + w / 2 - j2
        else:
            s = self.get_scale(H)
            dy = 0
            dx = 0

            ii = jj = self.crop_size
            h = w = H - self.crop_size * 2
        
        for k, image in enumerate(results[self.keys]):

            results[f'affine_params_{self.keys}'][k][0] = dy
            results[f'affine_params_{self.keys}'][k][1] = dx
            results[f'affine_params_{self.keys}'][k][3] = 1 / s # scale

            if s <= 1.:
                assert ii >= 0 and jj >= 0
                # zooming in
                image_crop = results[self.keys][k][ii:ii+h, jj:jj+w]
                results[self.keys][k] = mmcv.imresize(image_crop, (W,H))
            else:
                assert ii <= 0 and jj <= 0
                # zooming out
                pad_l = abs(jj)
                pad_r = w - W - pad_l
                pad_t = abs(ii)
                pad_b = h - H - pad_t

                image_pad = mmcv.impad(image, (pad_l, pad_t, pad_r, pad_b))
                # image_pad = F.pad(image, (pad_l, pad_t, pad_r, pad_b))
                results[self.keys][k] = mmcv.imresize(image_pad, (W,H))

        return results
    


@PIPELINES.register_module()
class Resize(object):
    """Resize images to a specific size.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "lazy",
    "resize_size". Required keys in "lazy" is None, added or modified key is
    "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False,
                 keys='imgs'):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy
        self.keys = keys

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy, self.keys)

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            results[self.keys] = [
                mmcv.imresize(
                    img, (new_w, new_h), interpolation=self.interpolation)
                for img in results[self.keys]
            ]
            if 'grids' in results:
                results['grids'] = [
                    mmcv.imresize(
                        grid, (new_w, new_h), interpolation=self.interpolation)
                    for grid in results['grids']
                ]
            # if 'points' in results:
            #     results['points'] *= np.array([new_w, new_h])

        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        if 'ref_seg_map' in results:
            if results['ref_seg_map'].dtype == np.uint8:
                results['ref_seg_map'] = mmcv.imresize(
                    results['ref_seg_map'], (new_w, new_h),
                    interpolation='nearest',
                    backend='pillow')
            else:
                results['ref_seg_map'] = mmcv.imresize(
                    results['ref_seg_map'], (new_w, new_h),
                    interpolation='bilinear',
                    backend='cv2')
                
                if len(results['ref_seg_map'].shape) == 2: 
                    results['ref_seg_map'] = results['ref_seg_map'][:,:,None]
                

                # if results['ref_seg_map'].n_dim() == 3:
                results['ref_seg_map'] = results['ref_seg_map'].transpose(2,0,1)
   

        if 'crop_bbox_ratio' in results:
            ratios = results['crop_bbox_ratio']
            results['bbox_mask'] = []
            results['mask_query_idx'] = []
            for idx, ratio in enumerate(ratios):
                img = results[self.keys][0]
                mask = np.zeros((new_h, new_w)).astype(np.uint8)
                mask[int(new_h * ratio[1]):int(new_h * ratio[3]), int(new_w * ratio[0]):int(new_w * ratio[2])] = 1
                if results.get('mask_sample_size', None):
                    sample_mask = mmcv.imresize(mask, results['mask_sample_size'], interpolation='nearest')
                    results['mask_query_idx'].append(sample_mask)
                
                results['bbox_mask'].append(mask)
        
        if 'masks' in results:
            if results['masks'][0].dtype == np.uint8:
                results['masks'] = [ mmcv.imresize(
                    mask, (32, 32),
                    interpolation='nearest',
                    backend='pillow') for mask in results['masks']]
            else:
                results['masks'] = [ mmcv.imresize(
                    mask, (32, 32),
                    interpolation='bilinear',
                    backend='cv2') for mask in results['masks'] ]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str

@PIPELINES.register_module()
class Flip(object):
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.
    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "lazy" and "flip_direction". Required keys in "lazy" is
    None, added or modified key are "flip" and "flip_direction".

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 lazy=False,
                 same_on_clip=True,
                 same_across_clip=False,
                 same_clip_indices=None,
                 same_frame_indices=None,
                 keys='imgs',
                 with_bbox=False):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.lazy = lazy
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip
        if same_clip_indices is not None:
            assert isinstance(same_clip_indices, Sequence)
        self.same_clip_indices = same_clip_indices
        if same_frame_indices is not None:
            assert isinstance(same_frame_indices, Sequence)
        self.same_frame_indices = same_frame_indices
        self.keys = keys
        self.with_bbox = with_bbox

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy, self.keys)
        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        if np.random.rand() < self.flip_ratio:
            flip = True
        else:
            flip = False

        results['flip'] = flip
        results['flip_direction'] = self.direction
        
        imgs = []
        if not self.lazy:
            results['bboxs'] = []
            for i, img in enumerate(results[self.keys]):
                is_new_clip = not self.same_across_clip and i % results[
                    'clip_len'] == 0 and i > 0
                generate_new = not self.same_on_clip or is_new_clip
                if self.same_clip_indices is not None:
                    assert min(self.same_clip_indices) >= 0
                    assert max(self.same_clip_indices) < results['num_clips']
                    keep_same = i % results[
                        'num_clips'] in self.same_clip_indices
                    generate_new = generate_new and not keep_same
                if self.same_frame_indices is not None:
                    assert min(self.same_frame_indices) >= 0
                    assert max(self.same_frame_indices) < results['clip_len']
                    keep_same = i % results[
                        'clip_len'] in self.same_frame_indices
                    generate_new = generate_new and not keep_same
                if generate_new:
                    flip = npr.rand() < self.flip_ratio
                if flip:
                    mmcv.imflip_(img, self.direction)
                    if results.get('bbox_mask', None) and self.with_bbox:
                        mmcv.imflip_(results['bbox_mask'][i], self.direction)
                        mmcv.imflip_(results['mask_query_idx'][i], self.direction)
                    if 'grids' in results:
                        mmcv.imflip_(results['grids'][i], self.direction)
                        
                    if results.get('flows', False):
                        mmcv.imflip_(results['flows'][i], self.direction)

                if results.get('bbox_mask', None) and self.with_bbox:
                    size = results['mask_sample_size']
                    ratio = results['crop_bbox_ratio'][i]

                    if not flip:
                        bbox = np.array([size[1] * ratio[0], size[0] * ratio[1], \
                                                size[1] * ratio[2], size[0] * ratio[3]])
                    else:
                        ratio_x = sorted([1- ratio[0], 1-ratio[2]])
                        ratio_y = sorted([1- ratio[1], 1-ratio[3]])
                        bbox = np.array([size[1] * ratio_x[0], size[0] * ratio_y[0], \
                                                size[1] * ratio_x[1], size[0] * ratio_y[1]])
                    results['bboxs'].append(bbox)
                
                if 'masks' in results:
                    if flip:
                        mmcv.imflip_(results['masks'][i])
            try:
                if results.get('flow', False):
                    mmcv.imflip_(results['flow'], self.direction)
            except Exception as e:
                pass
               
            if flip:
                lt = len(results[self.keys])
                for i in range(0, lt, 2):
                    # flow with even indexes are x_flow, which need to be
                    # inverted when doing horizontal flip
                    if modality == 'Flow':
                        results[self.keys][i] = mmcv.iminvert(results[self.keys][i])
            else:
                results[self.keys] = list(results[self.keys])

            if results.get('bbox_mask', None) and self.with_bbox:
                results['frames_mask'] = results['mask_query_idx']
                
                if results.get('return_first_query', False):
                    results['mask_query_idx'] = results['mask_query_idx'][0].reshape(-1)
                    
                else:
                    results['mask_query_idx'] = results['mask_query_idx'][-1].reshape(-1)
            
                            
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            lazyop['flip'] = flip
            lazyop['flip_direction'] = self.direction
        
        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False, keys='imgs'):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude
        self.keys = keys

    def __call__(self, results):
        modality = results.get('modality', 'RGB')

        if modality == 'RGB':
            n = len(results[self.keys])
            h, w, c = results[self.keys][0].shape
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results[self.keys]):
                imgs[i] = img

            for img in imgs:
                mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)

            results[self.keys] = imgs
            results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_bgr=self.to_bgr)
            return results
        elif modality == 'Flow':
            num_imgs = len(results[self.keys])
            assert num_imgs % 2 == 0
            assert self.mean.shape[0] == 2
            assert self.std.shape[0] == 2
            n = num_imgs // 2
            h, w = results[self.keys][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results[self.keys][2 * i]
                y_flow[i] = results[self.keys][2 * i + 1]
            x_flow = (x_flow - self.mean[0]) / self.std[0]
            y_flow = (y_flow - self.mean[1]) / self.std[1]
            if self.adjust_magnitude:
                x_flow = x_flow * results['scale_factor'][0]
                y_flow = y_flow * results['scale_factor'][1]
            imgs = np.stack([x_flow, y_flow], axis=-1)
            results[self.keys] = imgs
            args = dict(
                mean=self.mean,
                std=self.std,
                to_bgr=self.to_bgr,
                adjust_magnitude=self.adjust_magnitude)
            results['img_norm_cfg'] = args
            return results
        else:
            raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


@PIPELINES.register_module()
class CenterCrop(object):
    """Crop the center area from images.

    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox", "lazy" and "img_shape". Required keys in "lazy" is
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, crop_size, lazy=False, keys='imgs'):
        self.crop_size = _pair(crop_size)
        self.lazy = lazy
        self.keys = keys
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy, self.keys)

        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results[self.keys] = [
                img[top:bottom, left:right] for img in results[self.keys]
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class ThreeCrop(object):
    """Crop images into three crops.

    Crop the images equally into three crops with equal intervals along the
    shorter side.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size, keys='imgs'):
        self.crop_size = _pair(crop_size)
        self.keys = keys
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the ThreeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False, self.keys)

        imgs = results[self.keys]
        img_h, img_w = results[self.keys][0].shape[:2]
        crop_w, crop_h = self.crop_size
        assert crop_h == img_h or crop_w == img_w

        if crop_h == img_h:
            w_step = (img_w - crop_w) // 2
            offsets = [
                (0, 0),  # left
                (2 * w_step, 0),  # right
                (w_step, 0),  # middle
            ]
        elif crop_w == img_w:
            h_step = (img_h - crop_h) // 2
            offsets = [
                (0, 0),  # top
                (0, 2 * h_step),  # down
                (0, h_step),  # middle
            ]

        cropped = []
        crop_bboxes = []
        for x_offset, y_offset in offsets:
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            cropped.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results[self.keys] = cropped
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results[self.keys][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class TenCrop(object):
    """Crop the images into 10 crops (corner + center + flip).

    Crop the four corners and the center part of the image with the same
    given crop_size, and flip it horizontally.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size, keys='imgs'):
        self.crop_size = _pair(crop_size)
        self.keys = keys
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the TenCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False, self.keys)

        imgs = results[self.keys]

        img_h, img_w = results[self.keys][0].shape[:2]
        crop_w, crop_h = self.crop_size

        w_step = (img_w - crop_w) // 4
        h_step = (img_h - crop_h) // 4

        offsets = [
            (0, 0),  # upper left
            (4 * w_step, 0),  # upper right
            (0, 4 * h_step),  # lower left
            (4 * w_step, 4 * h_step),  # lower right
            (2 * w_step, 2 * h_step),  # center
        ]

        img_crops = list()
        crop_bboxes = list()
        for x_offset, y_offsets in offsets:
            crop = [
                img[y_offsets:y_offsets + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            flip_crop = [np.flip(c, axis=1).copy() for c in crop]
            bbox = [x_offset, y_offsets, x_offset + crop_w, y_offsets + crop_h]
            img_crops.extend(crop)
            img_crops.extend(flip_crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs) * 2)])

        crop_bboxes = np.array(crop_bboxes)
        results[self.keys] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results[self.keys][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class MultiGroupCrop(object):
    """Randomly crop the images into several groups.

    Crop the random region with the same given crop_size and bounding box
    into several groups.
    Required keys are "imgs", added or modified keys are "imgs", "crop_bbox"
    and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
        groups(int): Number of groups.
    """

    def __init__(self, crop_size, groups, keys='imgs'):
        self.crop_size = _pair(crop_size)
        self.groups = groups
        self.keys = keys
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(
                'Crop size must be int or tuple of int, but got {}'.format(
                    type(crop_size)))

        if not isinstance(groups, int):
            raise TypeError(f'Groups must be int, but got {type(groups)}.')

        if groups <= 0:
            raise ValueError('Groups must be positive.')

    def __call__(self, results):
        """Performs the MultiGroupCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        imgs = results[self.keys]
        img_h, img_w = imgs[0].shape[:2]
        crop_w, crop_h = self.crop_size

        img_crops = []
        crop_bboxes = []
        for _ in range(self.groups):
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)

            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            img_crops.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results[self.keys] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results[self.keys][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}'
                    f'(crop_size={self.crop_size}, '
                    f'groups={self.groups})')
        return repr_str


@PIPELINES.register_module()
class RGB2LAB(object):

    def __init__(self, 
                 norm=True,
                 keys='imgs',
                 output_keys='imgs'):
        self.norm = norm
        self.keys = keys
        self.output_keys = output_keys

    def __call__(self, results):
        if self.keys is not self.output_keys: 
            results[self.output_keys]  = copy.deepcopy(results[self.keys])
        for i, img in enumerate(results[self.keys]):
            # results[self.output_keys][i] = mmcv.imconvert(img, 'rgb', 'lab')
            if self.norm:
                img = np.float32(img) / 255.0
            results[self.output_keys][i] = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

        return results

@PIPELINES.register_module()
class RGB2GRAY(object):

    def __init__(self, 
                 keys='imgs',
                 output_keys='imgs'):
        self.keys = keys
        self.output_keys = output_keys

    def __call__(self, results):
        if self.keys is not self.output_keys: 
            results[self.output_keys]  = copy.deepcopy(results[self.keys])
        for i, img in enumerate(results[self.keys]):
            # results[self.output_keys][i] = mmcv.imconvert(img, 'rgb', 'lab')
            results[self.output_keys][i] = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0)[:,:,None]
            
        return results


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 p=0.5,
                 same_on_clip=True,
                 same_across_clip=True,
                 keys='imgs'):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.p = p
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip
        self.keys = keys

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img, beta):
        """Brightness distortion."""
        return self.convert(img, beta=beta)

    def contrast(self, img, alpha):
        """Contrast distortion."""
        return self.convert(img, alpha=alpha)

    def saturation(self, img, alpha):
        """Saturation distortion."""
        img = mmcv.bgr2hsv(img)
        img[:, :, 1] = self.convert(img[:, :, 1], alpha=alpha)
        img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img, delta):
        """Hue distortion."""
        img = mmcv.bgr2hsv(img)
        img[:, :, 0] = (img[:, :, 0].astype(int) + delta) % 180
        img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        apply_bright = npr.rand() < self.p
        bright_beta = npr.uniform(-self.brightness_delta,
                                  self.brightness_delta)
        apply_contrast = npr.rand() < self.p
        contrast_alpha = npr.uniform(self.contrast_lower, self.contrast_upper)
        apply_saturation = npr.rand() < self.p
        saturation_alpha = npr.uniform(self.saturation_lower,
                                       self.saturation_upper)
        apply_hue = npr.rand() < self.p
        hue_delta = npr.randint(-self.hue_delta, self.hue_delta)
        apply_mode = npr.rand() < self.p

        for i, img in enumerate(results[self.keys]):
            is_new_clip = not self.same_across_clip and i % results[
                'clip_len'] == 0 and i > 0
            if not self.same_on_clip or is_new_clip:
                apply_bright = npr.rand() < self.p
                bright_beta = npr.uniform(-self.brightness_delta,
                                          self.brightness_delta)
                apply_contrast = npr.rand() < self.p
                contrast_alpha = npr.uniform(self.contrast_lower,
                                             self.contrast_upper)
                apply_saturation = npr.rand() < self.p
                saturation_alpha = npr.uniform(self.saturation_lower,
                                               self.saturation_upper)
                apply_hue = npr.rand() < self.p
                hue_delta = npr.randint(-self.hue_delta, self.hue_delta)
                apply_mode = npr.rand() < self.p
            # random brightness
            if apply_bright:
                img = self.brightness(img, beta=bright_beta)

            if apply_mode and apply_contrast:
                img = self.contrast(img, alpha=contrast_alpha)

            # random saturation
            if apply_saturation:
                img = self.saturation(img, alpha=saturation_alpha)

            # random hue
            if apply_hue:
                img = self.hue(img, delta=hue_delta)

            # random contrast
            if not apply_mode and apply_contrast:
                img = self.contrast(img, alpha=saturation_alpha)

            results[self.keys][i] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@PIPELINES.register_module()
class RandomGaussianBlur(object):

    def __init__(self,
                 sigma_range=(0.1, 0.2),
                 p=0.5,
                 same_on_clip=True,
                 same_across_clip=True,
                 keys='imgs'):
        self.sigma_range = sigma_range
        self.p = p
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip
        self.keys = keys

    def __call__(self, results):
        apply = npr.rand() < self.p
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        for i, img in enumerate(results[self.keys]):
            is_new_clip = not self.same_across_clip and i % results[
                'clip_len'] == 0 and i > 0
            if not self.same_on_clip or is_new_clip:
                apply = npr.rand() < self.p
                sigma = random.uniform(self.sigma_range[0],
                                       self.sigma_range[1])
            if apply:
                pil_image = Image.fromarray(img)
                pil_image = pil_image.filter(
                    ImageFilter.GaussianBlur(radius=sigma))
                img = np.array(pil_image)
                results[self.keys][i] = img

        return results


@PIPELINES.register_module()
class RandomGrayScale(object):

    def __init__(self, p=0.5, same_on_clip=True, same_across_clip=True, keys='imgs'):
        self.p = p
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip
        self.keys = keys

    def __call__(self, results):
        apply = npr.rand() < self.p
        for i, img in enumerate(results[self.keys]):
            is_new_clip = not self.same_across_clip and i % results[
                'clip_len'] == 0 and i > 0
            if not self.same_on_clip or is_new_clip:
                apply = npr.rand() < self.p
            if apply:
                img = mmcv.rgb2gray(img, keepdim=True)
                img = np.repeat(img, 3, axis=-1)
                results[self.keys][i] = img

        return results

@PIPELINES.register_module()
class Grid(object):

    def __init__(self, normalize=False, keys='imgs'):
        self.normalize = normalize
        self.keys = keys

    def __call__(self, results):
        h, w = results['original_shape']
        y_grid, x_grid = np.meshgrid(
            range(h), range(w), indexing='ij', sparse=False)
        if self.normalize:
            y_grid = 2 * y_grid / h - 1
            x_grid = 2 * x_grid / w - 1
        grids = [
            np.stack((y_grid, x_grid), axis=-1).astype(np.float)
            for _ in range(len(results[self.keys]))
        ]

        results['grids'] = grids

        return results


# TODO not tested
@PIPELINES.register_module()
class Image2Patch(object):

    def __init__(self, patch_size=(64, 64, 3), stride=[0.5, 0.5], scale_jitter=(0.7, 0.9), keys='imgs'):
        self.patch_size = patch_size
        self.stride = stride
        self.crop_trans = _RandomResizedCrop(patch_size[:-1], scale=scale_jitter)
        self.keys = keys
        
        stride = np.random.random() * (stride[1] - stride[0]) + stride[0]
        self.stride = [int(patch_size[0]*stride), int(patch_size[1]*stride), patch_size[2]]

    def __call__(self, results):

        patches = []
        for img in results[self.keys]:
            patch = view_as_windows(img, self.patch_size, step=self.stride)
            patches.extend(list(patch.reshape(-1, *patch.shape[-3:])))
        # for i in range(len(patches)):
        #     patches[i] = self.crop_trans(patches[i])
        results[self.keys] = patches

        return results


@PIPELINES.register_module()
class HidePatch(object):
    """after normalization."""

    def __init__(self, patch_size, hide_prob, keys='imgs'):
        if not isinstance(patch_size, (list, tuple)):
            patch_size = [patch_size]
        self.patch_size = patch_size
        self.hide_prob = hide_prob
        self.keys = keys

    def __call__(self, results):
        patch_size = np.random.choice(self.patch_size)
        h, w = results[self.keys][0].shape[:2]
        for i, img in enumerate(results[self.keys]):
            for y in range(0, h, patch_size):
                for x in range(0, w, patch_size):
                    apply = npr.rand() < self.hide_prob
                    if apply:
                        results[self.keys][i][y:y + patch_size,
                                           x:x + patch_size] = 0

        return results


@PIPELINES.register_module()
class RandomAffine(object):

    def __init__(self,
                 degrees,
                 p=0.5,
                 same_on_clip=True,
                 same_across_clip=True,
                 translate=None,
                 scale=None,
                 shear=None,
                 resample=2,
                 fillcolor=0,
                 keys='imgs'):
        trans = _RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            resample=resample,
            fillcolor=fillcolor)
        self.degrees = trans.degrees
        self.translate = trans.translate
        self.scale = trans.scale
        self.shear = trans.shear
        self.resample = trans.resample
        self.fillcolor = trans.fillcolor
        self.p = p
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip
        self.keys = keys

    def __call__(self, results):
        apply = npr.rand() < self.p
        h, w = results[self.keys][0].shape[:2]
        ret = _RandomAffine.get_params(self.degrees, self.translate,
                                       self.scale, self.shear, (w, h))
        for i, img in enumerate(results[self.keys]):
            is_new_clip = not self.same_across_clip and i % results[
                'clip_len'] == 0 and i > 0
            if not self.same_on_clip or is_new_clip:
                apply = npr.rand() < self.p
                ret = _RandomAffine.get_params(self.degrees, self.translate,
                                               self.scale, self.shear, (w, h))
            if apply:
                img = np.array(
                    F.affine(
                        Image.fromarray(img),
                        *ret,
                        resample=self.resample,
                        fillcolor=self.fillcolor))
                results[self.keys][i] = img

        return results


@PIPELINES.register_module()
class RandomChoiceRotate(object):

    def __init__(self, p, degrees, same_on_clip=True, same_across_clip=True, keys='imgs'):
        self.p = p
        if not isinstance(degrees, (list, tuple)):
            degrees = [degrees]
        self.degrees = degrees
        self.label_map = {d: i for i, d in enumerate(degrees)}
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip
        self.keys = keys

    def __call__(self, results):
        apply = npr.rand() < self.p
        degree = np.random.choice(self.degrees)
        labels = []
        for i, img in enumerate(results[self.keys]):
            is_new_clip = not self.same_across_clip and i % results[
                'clip_len'] == 0 and i > 0
            if not self.same_on_clip or is_new_clip:
                apply = npr.rand() < self.p
                degree = np.random.choice(self.degrees)
            if apply:
                img = np.array(mmcv.imrotate(img, angle=degree))
                results[self.keys][i] = img
                labels.append(self.label_map[degree])
            else:
                labels.append(0)
        results['rotation_labels'] = np.array(labels)

        return results


@PIPELINES.register_module()
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        This variant of RandomErasing is intended to be applied to either a
        batch or single image tensor after it has been normalized by dataset
        mean and std.
    Args:
         p: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is
            scaled by count. per-image count is randomly chosen between 1 and
            this value.
    """

    def __init__(self,
                 p=0.5,
                 area_range=(0.02, 1 / 3),
                 aspect_ratio_range=(1 / 3, 3),
                 count_range=(1, 1),
                 mode='const',
                 keys='imgs'):
        self.p = p
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.count_range = count_range
        assert mode in ['rand', 'pixel', 'const']
        self.mode = mode
        self.keys = keys

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def get_pixels(self, patch_shape):
        if self.mode == 'pixel':
            return np.random.randn(*patch_shape)
        elif self.mode == 'rand':
            return np.random.randn(1, 1, patch_shape[-1])
        else:
            return np.zeros(patch_shape, dtype=np.float)

    def erase(self, img):
        count = random.randint(*self.count_range)
        img_h, img_w = img.shape[:2]
        for _ in range(count):
            left, top, right, bottom = self.get_crop_bbox(
                (img_h, img_w),
                (self.area_range[0] / count, self.area_range[1] / count),
                self.aspect_ratio_range)
            new_h, new_w = bottom - top, right - left
            img[top:bottom, left:right] = self.get_pixels(
                (new_h, new_w, img.shape[2]))

        return img

    def __call__(self, results):
        for i, img in enumerate(results[self.keys]):
            apply = npr.rand() < self.p
            if apply:
                results[self.keys][i] = self.erase(img)

        return results


@PIPELINES.register_module()
class ColorJitter(torch.nn.Module):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, p=0.5,
                 same_on_clip=True,
                 same_across_clip=True,
                 brightness=0,
                 contrast=0,
                 saturation=0,
                 hue=0,
                 keys='imgs',
                 output_keys='imgs',
                 keep_c=None):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.p = p
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip
        self.keys = keys
        self.output_keys = output_keys
        self.keep_c = keep_c

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness: Optional[List[float]],
                   contrast: Optional[List[float]],
                   saturation: Optional[List[float]],
                   hue: Optional[List[float]]
                   ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:

        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h


    def convert(self, img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        return img

    def __call__(self, results):
        apply = npr.rand() < self.p
        trans = self.get_params(self.brightness, self.contrast,
                                                self.saturation, self.hue)
        if self.keys is not self.output_keys: 
            results[self.output_keys] = results[self.output_keys] = copy.deepcopy(results[self.keys])

        if self.keep_c is not None:
            results['ori_imgs'] = copy.deepcopy(results['imgs'])
            keep_ch = random.randint(0,2)
            results['keep_ch'] = keep_ch
        
        for i, img in enumerate(results[self.keys]):
            is_new_clip = not self.same_across_clip and i % results[
                'clip_len'] == 0 and i > 0
            if not self.same_on_clip or is_new_clip:
                apply = npr.rand() < self.p
                trans = self.get_params(self.brightness, self.contrast,
                                                self.saturation, self.hue)
            if apply:
                img = np.array(self.convert(Image.fromarray(img), *trans))
                results[self.output_keys][i] = img
        
        if self.keep_c:
            for i, img in enumerate(results[self.output_keys]):
                results[self.output_keys][i][keep_ch] = results['ori_imgs'][i][keep_ch]
                
        return results
    
@PIPELINES.register_module()
class ColorDropout(object):

    def __init__(self, 
                 keys='imgs',
                 drop_rate=0.8
                 ):
        self.keys = keys
        self.droprate = drop_rate

    def __call__(self, results):
        
        if random.random() <= self.droprate:
            drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
            drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)
            
            for i, img in enumerate(results[self.keys]):
                for dropout_ch in drop_ch_ind:
                    img[:, :, dropout_ch] = 0
                img *= (3 / (3 - drop_ch_num))
                results[self.keys][i] = img
                
            return results
    
        else:
            return results
        
@PIPELINES.register_module()
class FrameDup(object):
    
    def __init__(self, 
                 keys_list=['imgs'],
                 out_keys_list=['imgs']
                 
                 ):
        self.keys_list = keys_list
        self.out_keys_list = out_keys_list
    
    def __call__(self, results):
        for source, target in zip(self.keys_list, self.out_keys_list):
            if source == target:
                pass
            else:
                results[target] = copy.deepcopy(results[source])
        return results
                
@PIPELINES.register_module()
class GetAffanity(object):
    
    def __init__(
        self,
        keys='imgs',
        size=(256, 256),
        get_inverse=True
    ):
        self.keys = keys
        self.size = size
        self.get_inverse= get_inverse
        
    def _get_affine(self, params):
    
        N = len(params)

        # construct affine operator
        affine = torch.zeros(N, 2, 3)

        aspect_ratio = float(self.size[0]) / \
                            float(self.size[1])

        for i, (dy,dx,alpha,scale,flip) in enumerate(params):

            # R inverse
            sin = math.sin(alpha * math.pi / 180.)
            cos = math.cos(alpha * math.pi / 180.)

            # inverse, note how flipping is incorporated
            affine[i,0,0], affine[i,0,1] = flip * cos, sin * aspect_ratio
            affine[i,1,0], affine[i,1,1] = -sin / aspect_ratio, cos

            # T inverse Rinv * t == R^T * t
            affine[i,0,2] = -1. * (cos * dx + sin * dy)
            affine[i,1,2] = -1. * (-sin * dx + cos * dy)

            # T
            affine[i,0,2] /= float(self.size[1] // 2)
            affine[i,1,2] /= float(self.size[0] // 2)

            # scaling
            affine[i] *= scale

        return affine

    def _get_affine_inv(self, affine, params):

        aspect_ratio = float(self.size[0]) / \
                            float(self.size[1])

        affine_inv = affine.clone()
        affine_inv[:,0,1] = affine[:,1,0] * aspect_ratio**2
        affine_inv[:,1,0] = affine[:,0,1] / aspect_ratio**2
        affine_inv[:,0,2] = -1 * (affine_inv[:,0,0] * affine[:,0,2] + affine_inv[:,0,1] * affine[:,1,2])
        affine_inv[:,1,2] = -1 * (affine_inv[:,1,0] * affine[:,0,2] + affine_inv[:,1,1] * affine[:,1,2])

        # scaling
        affine_inv /= torch.Tensor(params)[:,3].view(-1,1,1)**2

        return affine_inv
    
    def __call__(self, results):
        
        results[f'affine_{self.keys}'] = self._get_affine(results[f'affine_params_{self.keys}'])
        if self.get_inverse:
            results[f'affine_{self.keys}'] = self._get_affine_inv(results[f'affine_{self.keys}'], results[f'affine_params_{self.keys}'])
        
        return results
    
@PIPELINES.register_module()
class Flow_Sampler(object):
    def __init__(self,
        strategy=['watershed'], 
        bg_ratio=0.00015625, 
        nms_ks=15, 
        max_num_guide=-1, 
        guidepoint=None
    ):
        self.strategy = strategy
        self.bg_ratio = bg_ratio
        self.nms_ks = nms_ks
        self.max_num_guide = max_num_guide
        self.guidepoint = guidepoint


    def __call__(self, results):
        
        flow = results['flows'][0][:,:,:2]
        h = flow.shape[0]
        w = flow.shape[1]
        ds = max(1, max(h, w) // 400) # reduce computation

        pts_h = []
        pts_w = []

        stride = int(np.sqrt(1./self.bg_ratio))
        mesh_start_h = int((h - h // stride * stride) / 2)
        mesh_start_w = int((w - w // stride * stride) / 2)
        mesh = np.meshgrid(np.arange(mesh_start_h, h, stride), np.arange(mesh_start_w, w, stride))
        pts_h.append(mesh[0].flat)
        pts_w.append(mesh[1].flat)
       
        edge = self.get_edge(flow[::ds,::ds,:])
        edge /= max(edge.max(), 0.01)
        edge = (edge > 0.1).astype(np.float32)
        watershed = ndimage.distance_transform_edt(1-edge)
        nms_res = self.nms(watershed, self.nms_ks)
        self.remove_border(nms_res)
        pth, ptw = np.where(nms_res > 0)
        pth, ptw = self.neighbor_elim(pth, ptw, (self.nms_ks-1)/2)
        pts_h.append(pth * ds)
        pts_w.append(ptw * ds)
      
        pts_h = np.concatenate(pts_h)
        pts_w = np.concatenate(pts_w)

        if self.max_num_guide == -1:
           self.max_num_guide = np.inf

        randsel = np.random.permutation(len(pts_h))[:len(pts_h)]
        selidx = randsel[np.arange(min(self.max_num_guide, len(randsel)))]
        pts_h = pts_h[selidx]
        pts_w = pts_w[selidx]

        sparse = np.zeros(flow.shape, dtype=flow.dtype)
        mask = np.zeros(flow.shape, dtype=np.int)
        
        sparse[:, :, 0][(pts_h, pts_w)] = flow[:, :, 0][(pts_h, pts_w)]
        sparse[:, :, 1][(pts_h, pts_w)] = flow[:, :, 1][(pts_h, pts_w)]
        
        mask[:,:,0][(pts_h, pts_w)] = 1
        mask[:,:,1][(pts_h, pts_w)] = 1
        
        results['sparse'] = sparse
        results['mask'] = mask
        
        return results
    
    def get_edge(self, data, blur=False):
        if blur:
            data = cv2.GaussianBlur(data, (3, 3), 1.)
        sobel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]).astype(np.float32)
        ch_edges = []
        for k in range(data.shape[2]):
            edgex = signal.convolve2d(data[:,:,k], sobel, boundary='symm', mode='same')
            edgey = signal.convolve2d(data[:,:,k], sobel.T, boundary='symm', mode='same')
            ch_edges.append(np.sqrt(edgex**2 + edgey**2))
            
        return sum(ch_edges)
    
    def nms(self, score, ks):
        assert ks % 2 == 1
        ret_score = score.copy()
        maxpool = maximum_filter(score, footprint=np.ones((ks, ks)))
        ret_score[score < maxpool] = 0.
        return ret_score
    
    def neighbor_elim(self, ph, pw, d):
        valid = np.ones((len(ph))).astype(np.int)
        h_dist = np.fabs(np.tile(ph[:,np.newaxis], [1,len(ph)]) - np.tile(ph.T[np.newaxis,:], [len(ph),1]))
        w_dist = np.fabs(np.tile(pw[:,np.newaxis], [1,len(pw)]) - np.tile(pw.T[np.newaxis,:], [len(pw),1]))
        idx1, idx2 = np.where((h_dist < d) & (w_dist < d))
        for i,j in zip(idx1, idx2):
            if valid[i] and valid[j] and i != j:
                if np.random.rand() > 0.5:
                    valid[i] = 0
                else:
                    valid[j] = 0
        valid_idx = np.where(valid==1)
        return ph[valid_idx], pw[valid_idx]

    def remove_border(self, mask):
        mask[0,:] = 0
        mask[:,0] = 0
        mask[mask.shape[0]-1,:] = 0
        mask[:,mask.shape[1]-1] = 0
        
        


@PIPELINES.register_module()
class BlockwiseMaskGenerator(object):
    """Generate random block for the image.
    Args:
        input_size (int): Size of input image. Defaults to 192.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.6.
        mask_color (str): Filling color of the MIM mask in {'mean', 'zero'}.
            Defaults to 'zero'.
    """

    def __init__(self,
                 frame_idx=-1,
                 input_size=256,
                 mask_patch_size=32,
                 model_patch_size=4,
                 mask_ratio=0.5,
                 mask_only=False,
                 mask_color='mean',
                 keys='imgs',
                 output_keys='imgs'
                ):
        self.frame_idx = frame_idx
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.mask_only = mask_only
        self.mask_color = mask_color
        assert self.mask_color in ['mean', 'zero', 'rand',]
        if self.mask_color != 'zero':
            assert mask_only == False

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

        self.keys = keys
        self.output_keys = output_keys
        
    def __call__(self, results):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        # mask = torch.from_numpy(mask)  # [H, W]
        if self.keys is not self.output_keys: 
            results[self.output_keys]  = copy.deepcopy(results[self.keys])

        if self.mask_color == 'mean':
            img = results[self.keys][self.frame_idx]
            
            h, w, c = img.shape
            mask_ = mask.reshape((self.rand_size * self.scale, -1, 1))
            mask_ = mask_.repeat(
                self.model_patch_size, axis=0).repeat(self.model_patch_size, axis=1)
            mean = img.reshape(-1, img.shape[2]).mean(axis=0)
            img = np.where(mask_ == 1, img, mean).astype(np.uint8)
            
            results[self.output_keys][self.frame_idx] = img
        
        results['mask'] = mask
        
        return results
    
    

        


@PIPELINES.register_module()
class Flow2Heat(object):
    """Generate random block for the image.
    Args:
        input_size (int): Size of input image. Defaults to 192.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.6.
        mask_color (str): Filling color of the MIM mask in {'mean', 'zero'}.
            Defaults to 'zero'.
    """

    def __init__(self,
                 frame_idx=-1,
                 input_size=256,
                 mask_patch_size=32,
                 model_patch_size=4,
                 mask_ratio=0.5,
                 mask_only=False,
                 mask_color='mean',
                 keys='imgs',
                 output_keys='imgs'
                ):
        pass
    
    pass