from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
from tensorboardX import SummaryWriter

import glob
import json
import imageio
import cv2
import re
from torchvision.transforms import ColorJitter, GaussianBlur

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

from enum import Enum
from mmcv.utils import print_log


from mmcv import scandir
import mmcv

from .base_dataset import BaseDataset
from .video_dataset import *
from .registry import DATASETS

from .pipelines import Compose
from mmpt.utils import *


IGNORE_ANIMALS = [
    # "bear.json",
    # "camel.json",
    "cat_jump.json"
    # "cows.json",
    # "dog.json",
    # "dog-agility.json",
    # "horsejump-high.json",
    # "horsejump-low.json",
    # "impala0.json",
    # "rs_dog.json"
    "tiger.json"
]


class SMALJointCatalog(Enum):
    # body_0 = 0
    # body_1 = 1
    # body_2 = 2
    # body_3 = 3
    # body_4 = 4
    # body_5 = 5
    # body_6 = 6
    # upper_right_0 = 7
    upper_right_1 = 8
    upper_right_2 = 9
    upper_right_3 = 10
    # upper_left_0 = 11
    upper_left_1 = 12
    upper_left_2 = 13
    upper_left_3 = 14
    neck_lower = 15
    # neck_upper = 16
    # lower_right_0 = 17
    lower_right_1 = 18
    lower_right_2 = 19
    lower_right_3 = 20
    # lower_left_0 = 21
    lower_left_1 = 22
    lower_left_2 = 23
    lower_left_3 = 24
    tail_0 = 25
    # tail_1 = 26
    # tail_2 = 27
    tail_3 = 28
    # tail_4 = 29
    # tail_5 = 30
    tail_6 = 31
    jaw = 32
    nose = 33 # ADDED JOINT FOR VERTEX 1863
    # chin = 34 # ADDED JOINT FOR VERTEX 26
    right_ear = 35 # ADDED JOINT FOR VERTEX 149
    left_ear = 36 # ADDED JOINT FOR VERTEX 2124

class SMALJointInfo():
    def __init__(self):
        # These are the 
        self.annotated_classes = np.array([
            8, 9, 10, # upper_right
            12, 13, 14, # upper_left
            15, # neck
            18, 19, 20, # lower_right
            22, 23, 24, # lower_left
            25, 28, 31, # tail
            32, 33, # head
            35, # right_ear
            36]) # left_ear

        self.annotated_markers = np.array([
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR,
            cv2.MARKER_CROSS,
            cv2.MARKER_CROSS])

        self.joint_regions = np.array([ 
            0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 
            2, 2, 2, 2, 
            3, 3, 
            4, 4, 4, 4, 
            5, 5, 5, 5, 
            6, 6, 6, 6, 6, 6, 6,
            7, 7, 7,
            8, 
            9])

        self.annotated_joint_region = self.joint_regions[self.annotated_classes]
        self.region_colors = np.array([
            [250, 190, 190], # body, light pink
            [60, 180, 75], # upper_right, green
            [230, 25, 75], # upper_left, red
            [128, 0, 0], # neck, maroon
            [0, 130, 200], # lower_right, blue
            [255, 255, 25], # lower_left, yellow
            [240, 50, 230], # tail, majenta
            [245, 130, 48], # jaw / nose / chin, orange
            [29, 98, 115], # right_ear, turquoise
            [255, 153, 204]]) # left_ear, pink
        
        self.joint_colors = np.array(self.region_colors)[self.annotated_joint_region]

@DATASETS.register_module()
class BadjaDataset(Video_dataset_base):
    
    def __init__(self, size=(320, 512), sigma=-1, scale=1, length=-1, vis_traj=False, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.vis_traj = vis_traj
        self.sigma = sigma
        self.size = size
        self.scale = scale
        self.length = length
        self.load_annotations()
          
    def load_annotations(self):
        
        logger = get_root_logger()
        
        annotations_path = os.path.join(self.list_path, "joint_annotations")

        self.animal_dict = {}
        self.animal_count = 0
        self.smal_joint_info = SMALJointInfo()
        num_frames = 0
        
        for animal_id, animal_json in enumerate(sorted(os.listdir(annotations_path))):
            if animal_json not in IGNORE_ANIMALS:
                json_path = os.path.join(annotations_path, animal_json)
                with open(json_path) as json_data:
                    animal_joint_data = json.load(json_data)

                filenames = []
                segnames = []
                joints = []
                visible = []

                first_path = animal_joint_data[0]['segmentation_path']
                last_path = animal_joint_data[-1]['segmentation_path']
                first_frame = first_path.split('/')[-1]
                last_frame = last_path.split('/')[-1]

                if not 'extra_videos' in first_path:


                    animal = first_path.split('/')[-2]

                    first_frame_int = int(first_frame.split('.')[0])
                    last_frame_int = int(last_frame.split('.')[0])

                    for idx, fr in enumerate(range(first_frame_int, last_frame_int+1)):
                        
                        ref_file_name = os.path.join(self.root, 'JPEGImages/Full-Resolution/%s/%05d.jpg' % (animal, fr))
                        ref_seg_name = os.path.join(self.root, 'Annotations/Full-Resolution/%s/%05d.png' % (animal, fr))
                        # print('ref_file_name', ref_file_name)

                        foundit = False
                        for ind, image_annotation in enumerate(animal_joint_data):
                            file_name = os.path.join(self.root, image_annotation['image_path'][6:])
                            seg_name = os.path.join(self.root, image_annotation['segmentation_path'][6:])


                            if file_name == ref_file_name:
                                foundit = True
                                label_ind = ind

                        if foundit:
                            image_annotation = animal_joint_data[label_ind]
                            file_name = os.path.join(self.root, image_annotation['image_path'][6:])
                            seg_name = os.path.join(self.root, image_annotation['segmentation_path'][6:])
                            joint = np.array(image_annotation['joints'])
                            vis = np.array(image_annotation['visibility'])
                        else:
                            file_name = ref_file_name
                            seg_name = ref_seg_name

                            joint = None
                            vis = None

                        filenames.append(file_name)
                        segnames.append(seg_name)
                        joints.append(joint)
                        visible.append(vis)
                    
                if len(filenames):
                    self.animal_dict[self.animal_count] = (filenames, segnames, joints, visible)
                    self.animal_count += 1
                    
        logger.info("Loaded BADJA dataset")
        logger.info('found %d unique videos in %s' % (self.animal_count, self.list_path))
        
        return 0
    
    def get_loader(self):
        for idx in range(int(1e6)):
            animal_id = np.random.choice(len(self.animal_dict.keys()))
            # print('choosing animal_id', animal_id)
            filenames, segnames, joints, visible = self.animal_dict[animal_id]
            # print('filenames', filenames)

            image_id = np.random.randint(0, len(filenames))

            seg_file = segnames[image_id]
            image_file = filenames[image_id]
            
            joints = joints[image_id].copy()
            joints = joints[self.smal_joint_info.annotated_classes]
            visible = visible[image_id][self.smal_joint_info.annotated_classes]

            rgb_img = imageio.imread(image_file)#, mode='RGB')
            sil_img = imageio.imread(seg_file)#, mode='RGB')

            rgb_h, rgb_w, _ = rgb_img.shape
            sil_img = cv2.resize(sil_img, (rgb_w, rgb_h), cv2.INTER_NEAREST)

            yield rgb_img, sil_img, joints, visible, image_file

    def get_video(self, animal_id):
        # print('choosing animal_id', animal_id)
        filenames, segnames, joint, visible = self.animal_dict[animal_id]
        # print('filenames', filenames)

        rgbs = []
        segs = []
        joints = []
        visibles = []

        for s in range(len(filenames)):
            image_file = filenames[s]
            rgb_img = mmcv.imread(image_file, backend='cv2', channel_order='rgb')#, mode='RGB')
            rgb_h, rgb_w, _ = rgb_img.shape
            
            seg_file = segnames[s]
            sil_img = mmcv.imread(seg_file, backend='pillow', flag='unchanged')
            sil_img = cv2.resize(sil_img, (rgb_w, rgb_h), cv2.INTER_NEAREST)
            
            jo = joint[s]

            # print('image_file', image_file)
            # print('seg_file', seg_file)
            if jo is not None:
                joi = joint[s].copy()
                joi = joi[self.smal_joint_info.annotated_classes]
                vis = visible[s][self.smal_joint_info.annotated_classes]
            else:
                joi = None
                vis = None

            rgbs.append(rgb_img)
            segs.append(sil_img)
            joints.append(joi)
            visibles.append(vis)

        return rgbs, segs, joints, visibles, filenames[0]
        
        
    def draw_label_map(self, img, pt, sigma):
        # Draw a 2D gaussian

        # Check that any part of the gaussian is in-bounds
        ul = [int(pt[1] - 3 * sigma), int(pt[0] - 3 * sigma)]
        br = [int(pt[1] + 3 * sigma + 1), int(pt[0] + 3 * sigma + 1)]
        if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0
                or br[1] < 0):
            # If not, just return the image as is
            return img

        # Generate gaussian
        size = 6 * sigma + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return img
    
    
    def img2coord(self, imgs, num_poses, topk=5):
        clip_len = len(imgs)
        height, width = imgs.shape[2:]
        assert imgs.shape[:2] == (clip_len, num_poses)
        coords = np.zeros((2, num_poses, clip_len), dtype=np.float)
        imgs = imgs.reshape(clip_len, num_poses, -1)
        assert imgs.shape[-1] == height * width
        # [clip_len, NUM_KEYPOINTS, topk]
        topk_indices = np.argsort(imgs, axis=-1)[..., -topk:]
        topk_values = np.take_along_axis(imgs, topk_indices, axis=-1)
        topk_values = topk_values / (np.sum(topk_values, keepdims=True, axis=-1)+1e-9)
        topk_x = topk_indices % width
        topk_y = topk_indices // width
        # [clip_len, NUM_KEYPOINTS]
        coords[0] = np.sum(topk_x * topk_values, axis=-1).T
        coords[1] = np.sum(topk_y * topk_values, axis=-1).T
        coords[:, np.sum(imgs.transpose(1, 0, 2), axis=-1) == 0] = -1 

        return coords
    
    
    
    def prepare_test_data(self, index):
        
        frames, segs, joints, visibles, video_path = self.get_video(index)
        
        origin_shape = frames[0].shape[0:2]
        
        sy = self.size[0] / origin_shape[0]
        sx = self.size[1] / origin_shape[1]
        
        frames = [ mmcv.imresize(x, size=(self.size[1], self.size[0])) for x in frames]

        # num_poses * (y,x)
        ref = joints[0]
        ref[:,0] = ref[:,0] * sy
        ref[:,1] = ref[:,1] * sx

        
        if self.length != -1:
            frames = frames[0:self.length]

        
        num_poses = ref.shape[0]
        num_frames = len(frames)
        video_path = os.path.dirname(video_path)
        origin_shape = frames[0].shape[:2]


        height, width = frames[0].shape[:2]
        
        height_ = height // self.scale
        width_ = width // self.scale
        
        pose_map = np.zeros((height_, width_, num_poses), dtype=np.float)
        pose_coord_raw = ref.transpose(1,0)
        pose_coord = pose_coord_raw / self.scale
        
        for j in range(num_poses):
            if self.sigma > 0:
                pose_map[:, :, j] = self.draw_label_map(pose_map[:, :, j], pose_coord[:, j], self.sigma)
                # print(pose_map[:,:,j].sum())
            else:
                ty = int(pose_coord[0, j])
                tx = int(pose_coord[1, j])
                if 0 <= tx < width_ and 0 <= ty < height_:
                    pose_map[ty, tx, j] = 1.0
                    
        # pose_map = pose_map.astype(np.uint8)

        data = {
            'imgs': frames,
            'ref':pose_coord_raw.astype(np.float32),
            'ref_seg_map': pose_map,
             'video_path': video_path,
            'original_shape': self.size,
            'modality': 'RGB',
            'num_clips': 1,
            'clip_len': num_frames,
        }
        return self.pipeline(data)
    
    
    def evaluate(self, results, metrics='pck', output_dir=None, logger=None):
        print('evaluate')
        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['pck']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        eval_results = dict()
        if mmcv.is_seq_of(results, list):
            num_feats = len(results[0])
            for feat_idx in range(num_feats):
                cur_results = [result[feat_idx] for result in results]
                eval_results.update(
                    add_prefix(
                        self.pck_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        else:
            eval_results.update(self.pck_evaluate(results, output_dir, logger))
        copypaste = []
        for k, v in list(eval_results.items())[:2]:
            copypaste.append(f'{float(v):.2f}')
        print_log(f'Results copypaste  {",".join(copypaste)}', logger=logger)
        return eval_results
    
    def pck_evaluate(self, results, output_dir, logger=None):
        # output_dir = './'
        writer = SummaryWriter(os.path.join(output_dir, 'traj'), max_queue=10, flush_secs=60)
        log_freq = 99999
        sw = pips_vis.Summ_writer(  
                                  writer=writer,
                                log_freq=log_freq,
                                fps=24,
                                scalar_freq=int(log_freq/2),
                                just_gif=True
                                )
    
        # dist_all = [np.zeros((0, 0)) for _ in range(num_poses)]
        if terminal_is_available():
            prog_bar = mmcv.ProgressBar(len(self))
            
        pck_range = [0.1,0.2,0.3,0.4]
        pck_result = {}
        for ratio in pck_range:
            pck_result[str(ratio)] = []
            
        avg = []
        
        for vid_idx in range(len(results)):
            
            pck_result_per_video = {}
            for ratio in pck_range:
                pck_result_per_video[str(ratio)] = []
            
            cur_results = results[vid_idx]
            if isinstance(cur_results, str):
                file_path = cur_results
                cur_results = np.load(file_path)
                os.remove(file_path)
                 # [2, num_poses, clip_len]
                pred_poses = self.img2coord(cur_results, num_poses)
            else:
                pred_poses = cur_results
                
            
            frames, segs, joints, visibles, video_path = self.get_video(vid_idx)
            
            
            origin_shape = frames[0].shape[0:2]
        
            sy = self.size[0] / origin_shape[0]
            sx = self.size[1] / origin_shape[1]
            
            frames = [ mmcv.imresize(x, size=(self.size[1], self.size[0])) for x in frames ]
            segs = [ mmcv.imresize(x, size=(self.size[1], self.size[0]), interpolation="nearest") for x in segs]
            
            
            for idx, joint in enumerate(joints):
                if joint is None:
                    pass
                else:
                    joints[idx][:,0] = joint[:,0] * sy
                    joints[idx][:,1] = joint[:,1] * sx
                
            if self.length != -1:
                frames = frames[0:self.length]
                segs = segs[0:self.length]
                joints = joints[0:self.length]
                visibles = visibles[0:self.length]
                
            clip_len = len(frames)   
            num_poses = joints[0].shape[0]
            
            for s in range(0,len(frames)):
                if joints[s] is None:
                    # segs[s] = np.zeros_like(segs[0])
                    joints[s] = np.zeros_like(joints[0])
                    visibles[s] = np.zeros_like(visibles[0])
             
            # B x T x C x H x W
            frames = torch.from_numpy(np.stack(frames, axis=0))[None].permute(0,1,4,2,3).cuda()
            frames  = frames.float() * 1./255 - 0.5
            gray_rgbs = torch.mean(frames, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
            

            

            # [num_poses, clip_len]
            joint_visible = pred_poses[0] > 0
            
            trajs_e = torch.zeros((1, clip_len, num_poses, 2)).cuda()
            
            for img_idx in range(clip_len):
                for t in range(num_poses):
                    # if not joint_visible[t, img_idx]:
                    #     print('haha')
                    
                    vis = visibles[img_idx][t]
                    
                    predx = pred_poses[0, t, img_idx]
                    predy = pred_poses[1, t, img_idx]
                     
                    gtx = joints[img_idx][t,1]
                    gty = joints[img_idx][t,0]
                    
                    trajs_e[0, img_idx, t, 0] = predx
                    trajs_e[0, img_idx, t, 1] = predy

                    dist = math.sqrt((gtx-predx)**2+(gty-predy)**2)

                    if vis > 0:
                        seg = (segs[img_idx] > 0)
                        area = seg.sum()
                        for ratio in pck_range:
                            thr = ratio * math.sqrt(area)
                            correct = (dist < thr)
                            pck_result[str(ratio)].append(correct)
                            
                            pck_result_per_video[str(ratio)].append(correct)
            
            
            
            with open(os.path.join(output_dir, 'result_per_video.txt'), 'a') as f:
                pck_all_per_video = []
                
                pck = np.mean(np.array(pck_result_per_video[str(0.2)])) * 100.0
                
                f.write(f'{vid_idx}_{pck}' + '\n')
                
                avg.append(pck)
            
            if output_dir is not None and self.vis_traj:
                os.makedirs(output_dir, exist_ok=True)
                if vid_idx % 1 == 0:
                    for n in range(num_poses):
                        if n > 8: break
                        if visibles[0][n] > 0:
                            sw.summ_traj2ds_on_rgbs(f'{output_dir}/kp{vid_idx}{n}_trajs_e_on_rgbs', trajs_e[0:1,:,n:n+1], gray_rgbs[0:1,:clip_len], cmap='spring', linewidth=2)   
                    
            if terminal_is_available():
                prog_bar.update()
                

        pck_all = []
        for ratio in pck_range:
            pck = np.mean(np.array(pck_result[str(ratio)])) * 100.0
            pck_all.append(pck)
        
        avg_pck = np.mean(np.array(avg))
            
        eval_results = {}
        
        
        with open(os.path.join(output_dir, 'result.txt'), 'a') as f:
            for alpha, pck in zip(pck_range, pck_all):
                eval_results[f'PCK@{alpha}'] = np.mean(pck)
                f.write(f'PCK@{alpha}: {np.mean(pck)}')
            f.write(f'PCK@0.1 AVG: {avg_pck}')
        return eval_results
    
        
    def __len__(self):

        return self.animal_count