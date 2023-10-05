import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import random
import cv2
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
import warnings
import math
import torch
from scipy import ndimage
import mmcv

from ..registry import PIPELINES


@PIPELINES.register_module()
class ClipRandomSizedCrop_stm(object):
    def __init__(self, scale, crop_size):
        self.scale = scale
        self.crop_size = crop_size

    def __call__(self, results):
        images =  results['images']
        labels = results['labels']
        scale_factor = random.uniform(self.scale[0],self.scale[1])
        x1_ = []
        x2_ = []
        y1_ = []
        y2_ = []
        for i in range(len(images)): 
            h, w = labels[i].shape
            h, w = (max(384,int(h * scale_factor)), max(384,int(w * scale_factor)))
            images[i] = (cv2.resize(images[i], (w, h), interpolation=cv2.INTER_LINEAR))
            labels[i] = Image.fromarray(labels[i]).resize((w, h), resample=Image.NEAREST)
            labels[i] = np.asarray(labels[i], dtype=np.int8)
        ob_loc = ((sum(labels)) > 0).astype(np.uint8)
        box = cv2.boundingRect(ob_loc)

        x_min = box[0]
        x_max = box[0] + box[2]
        y_min = box[1]
        y_max = box[1] + box[3]

        if x_max - x_min >384:
            start_w = random.randint(x_min,x_max - 384)
        elif x_max - x_min == 384:
            start_w = x_min
        else:
            start_w = random.randint(max(0,x_max-384), min(x_min,w - 384))

        if y_max - y_min >384:
            start_h = random.randint(y_min,y_max - 384)
        elif y_max - y_min == 384:
            start_h = y_min
        else:
            start_h = random.randint(max(0,y_max-384), min(y_min,h - 384))
        # Cropping

        end_h = start_h + 384
        end_w = start_w + 384
        for i in range(len(images)):
            start_h = random.randint(start_h-20,start_h+20)
            start_h = max(0,start_h)
            start_h = min(h - 384,start_h)
            start_w = random.randint(start_w-20,start_w+20)
            start_w = max(0,start_w)
            start_w = min(w - 384,start_w)
            end_h = start_h + 384
            end_w = start_w + 384
            images[i] = images[i][start_h:end_h, start_w:end_w]/255.
            labels[i] = labels[i][start_h:end_h, start_w:end_w]

        results['images'] = images
        results['labels'] = labels

        return results

@PIPELINES.register_module()
class ClipRandomSizedCrop(object):
    def __init__(self, size,  scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, backend='cv2'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

        self.backend = backend

    @staticmethod
    def get_params(img, scale, ratio, backend):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        if backend == 'cv2':
            height, width, _ = img.shape
        else:
            width, height = img.size()

        area = height * width

        for _ in range(20):
            sc = random.uniform(*scale)
            target_area = sc * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height

        i = (height - h) // 2
        j = (width - w) // 2

        return i, j, h, w

    def __call__(self, results):
        i, j, h, w = self.get_params(results['imgs'][0], self.scale, self.ratio, self.backend)

        if self.backend == 'pillow':
            results['imgs'] = list([F.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in results['imgs']])
            if results['with_flow']:
                results['flows'] = list([F.resized_crop(flow, i, j, h, w, self.size, self.interpolation) for flow in results['flows']])
        else:
            results['imgs'] = list([cv2.resize(img[i:i+h, j:j+w], self.size, self.interpolation) for img in results['imgs']])
            if results['with_flow']:
                results['flows'] = list([cv2.resize(flow[i:i+h, j:j+w], self.size, self.interpolation) for flow in results['flows']])
        
        return results

@PIPELINES.register_module()
class ClipRandomResizedCropObject(object):

    def __init__(self, size,  scale=(0.08, 1.0), ratio=(3/4, 4/3), backend='cv2'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio
        self.backend = backend

    @staticmethod
    def get_params(imgs, labels, obj_num, scale, ratio):

        height, width, _ = imgs[0].shape
        area = height * width

        obj_loc = (np.array(labels) == obj_num).astype(np.uint8).sum(0)

        assert obj_loc.sum() > 0
        obj_loc = (obj_loc > 0).astype(np.uint8)

        obj_box = cv2.boundingRect(obj_loc)

        w_min = obj_box[0]
        w_max = obj_box[0] + obj_box[2]
        h_min = obj_box[1]
        h_max = obj_box[1] + obj_box[3]

        for _ in range(30):
            sc = random.uniform(*scale)
            target_area = sc * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:

                if obj_box[2] <= w and obj_box[3] <= h:

                    for i in range(10):
                        i = random.randint(0, h_min)
                        j = random.randint(0, w_min)

                        if h_max <= i + h <= height and  w_max <= j + w <= width:
                            return i, j, h, w
                        else:
                            continue
                else:
                    continue
        
        # fall back
        h = random.randint(h_max, height) - h_min
        w = random.randint(w_max, width) - w_min

        return h_min, w_min, h, w
                   
               

    def __call__(self, results):

        imgs = results['images']
        labels = results['labels']
        obj_num = results['obj_num']
        
        i, j, h, w = self.get_params(imgs, labels, obj_num, self.scale, self.ratio)

        imgs = list([ img[i:i+h, j:j+w] for img in imgs])
        labels = list([ label[i:i+h, j:j+w] for label in labels])

        results['images'] = list([ cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR) for img in imgs])
        results['labels'] = list([ cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST) for label in labels])

        return results


@PIPELINES.register_module()
class ClipColorJitter(transforms.ColorJitter):

    def __init__(self, from_numpy, **kwargs):
        self.from_numpy = from_numpy

    def __call__(self, results):
        """
        Args:
            clip (List of PIL Image or Numpy): Input clip.

        Returns:
            List of PIL Image or Tensor: Color jittered clip.
        """
        images = results['images']
        if self.from_numpy:
            images = list([ Image.fromarray(x) for x in images ])

        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                images = [F.adjust_brightness(img, brightness_factor) for img in images]

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                images = [F.adjust_contrast(img, contrast_factor) for img in images]

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                images = [F.adjust_saturation(img, saturation_factor) for img in images]

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                images = [F.adjust_hue(img, hue_factor) for img in images]
        
        results['images'] = images

        return results

@PIPELINES.register_module()
class ClipRandomGrayscale(transforms.RandomGrayscale):

    def __init__(self, from_numpy, **kwargs):
        self.from_numpy = from_numpy

    def __call__(self, results):
        """
        Args:
            clip (List of PIL Image or Tensor): Clip to be converted to grayscale.

        Returns:
            List of PIL Image or Tensor: Randomly grayscaled clip.
        """
        images = results['images']
        if self.from_numpy:
            images = list([ Image.fromarray(x) for x in images ])

        num_output_channels = 1 if clip[0].mode == 'L' else 3
        if random.random() < self.p:
            results['images'] = [F.to_grayscale(img, num_output_channels=num_output_channels) for img in images]
            return results
        return results


@PIPELINES.register_module()
class ToClipTensor():
    def __call__(self, results):
        """
        Args:
            clip (List of PIL Image or numpy.ndarray): Clip to be converted to tensor.
        Returns:
            Tensor: Converted clip.
        """
        results['images'] = list([F.to_tensor(img) for img in results['images']])
        return results

@PIPELINES.register_module()
class ClipOrganize():
    def __init__(self, time_dim='T'):
        self.time_dim = time_dim
    
    def __call__(self, results):
        if self.time_dim == 'T':
            results['images'] = torch.stack(results['images'], 1)
        else:
            results['images'] = torch.cat(results['images'],0)
        return results


@PIPELINES.register_module()
class ClipMotionStatistic():
    def __init__(self, input_size, mag_size):
        self.input_size = input_size
        self.mag_size = mag_size

    @staticmethod
    def motion_mag_downsample(mag, size, input_size):
        block_size = input_size // size
        mask = np.zeros((size,size))
        for i in range(size):
            for j in range(size):
                x_start = i * block_size
                x_end = x_start + block_size
                y_start = j * block_size
                y_end = y_start + block_size

                tmp_block = mag[x_start:x_end, y_start:y_end]

                block_mean = np.mean(tmp_block)
                mask[i, j] = block_mean
        return mask

    @staticmethod
    def compute_motion_boudary(flow_clip):
        mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        my = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        dx_all = []
        dy_all = []
        mb_x = 0
        mb_y = 0

        for flow_img in flow_clip:
            d_x = ndimage.convolve(flow_img, mx)
            d_y = ndimage.convolve(flow_img, my)

            dx_all.append(d_x)
            dy_all.append(d_y)

            mb_x += d_x
            mb_y += d_y

        dx_all = np.array(dx_all)
        dy_all = np.array(dy_all)

        return dx_all, dy_all, mb_x, mb_y

    def __call__(self, flows):

        u, v = list([flow[:,:,0].astype(np.float32) for flow in flows]), list([flow[:,:,1].astype(np.float32) for flow in flows])
        _, _ , mb_x_u, mb_y_u = self.compute_motion_boudary(u)
        mag_u, ang = cv2.cartToPolar(mb_x_u, mb_y_u, angleInDegrees=True)

        _, _ , mb_x_v, mb_y_v = self.compute_motion_boudary(v)
        mag_v, ang = cv2.cartToPolar(mb_x_v, mb_y_v, angleInDegrees=True)

        mag = (mag_u + mag_v) / 2

        mag_down = self.motion_mag_downsample(mag, self.mag_size, self.input_size)

        return mag_down

class ClipRandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, results):
        """
        Args:
            clip (List of PIL Image or Tensor): Clip to be flipped.

        Returns:
            List of PIL Image or Tensor: Randomly flipped clip.
        """

        if torch.rand(1) < self.p:
            results['imgs'] =  list([mmcv.imflip_(img, 'horizontal') for img in results['imgs']])

            if results['with_flow']:
                results['flows'] =  list([mmcv.imflip_(flow, 'horizontal') for flow in results['flows']])
        return results