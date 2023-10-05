import copy
import glob
import os
import os.path as osp

import cv2
import mmcv
import numpy as np
import scipy.io as sio
from mmcv.utils import print_log

from mmpt.utils import *
from mmpt.utils import add_prefix, terminal_is_available

from .registry import DATASETS
from .video_dataset import Video_dataset_base


@DATASETS.register_module()
class jhmdb_dataset_rgb(Video_dataset_base):

    NUM_KEYPOINTS = 15

    PALETTE = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
               [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
               [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
               [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255]]

    def __init__(self,
                 sigma=4,
                 filename_tmpl='{:05}.png',
                 **kwargs
                       ):
        super().__init__(filename_tmpl=filename_tmpl, **kwargs)
        self.sigma = sigma
        self.load_annotations()

    def vis_pose(self, img, coord):
        pa = np.zeros(15)
        pa[2] = 0
        pa[12] = 8
        pa[8] = 4
        pa[4] = 0
        pa[11] = 7
        pa[7] = 3
        pa[3] = 0
        pa[0] = 1
        pa[14] = 10
        pa[10] = 6
        pa[6] = 1
        pa[13] = 9
        pa[9] = 5
        pa[5] = 1

        canvas = img
        x = coord[0, :]
        y = coord[1, :]

        for n in range(len(x)):
            pair_id = int(pa[n])

            x1 = int(x[pair_id])
            y1 = int(y[pair_id])
            x2 = int(x[n])
            y2 = int(y[n])

            if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                cv2.line(canvas, (x1, y1), (x2, y2), self.PALETTE[n], 4)

        return canvas
    
    def load_annotations(self):
        
        self.samples = []
        self.video_dir = osp.join(self.root, 'JHMDB')
        list_path = osp.join(self.list_path, f'{self.split}_list.txt')
        num_frame= 0
        
        with open(list_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                sample = dict()
                anno, vname = line.strip('\n').split()
                sample['frames_path'] = sorted(glob.glob(osp.join(self.root, vname, '*.png')))
                sample['num_frames'] = len(sample['frames_path'])
                sample['anno_path'] = osp.join(self.root, anno)
                sample['video_path'] = osp.join(self.root, vname)
                
                
                if sample['num_frames'] < self.clip_length * self.step: continue
                
                self.samples.append(sample)
                num_frame += sample['num_frames']
                
        # self.samples = self.samples[:1]
                
        logger = get_root_logger()
        logger.info(" Load dataset with {} videos ".format(len(self.samples)))
        logger.info(" Load dataset with {} frames ".format(num_frame))
        

    def prepare_test_data(self, idx):
        sample = self.samples[idx]
        frames_path = sample['frames_path']
        anno_path = sample['anno_path']
        num_frames = sample['num_frames']
        frames = self._parser_rgb_rawframe([0], frames_path, num_frames)
        original_shape = frames[0].shape[:2]
       
        pose_mat = sio.loadmat(anno_path)
        # magic -1
        
        data = {
            'imgs': frames,
            'ref_seg_map': pose_mat,
            'video_path': sample['video_path'],
            'original_shape': original_shape,
            'modality': 'RGB',
            'num_clips': 1,
            'clip_len': num_frames
        }
        
        data['pose_coord'] = pose_mat['pos_img'][..., 0] - 1
        
        pose_coord = data['pose_coord']
        num_poses = pose_coord.shape[1]
        height, width = frames[0].shape[:2]
        pose_map = np.zeros((height, width, num_poses), dtype=np.float)

        for j in range(num_poses):
            if self.sigma > 0:
                self.draw_label_map(pose_map[:, :, j], pose_coord[:, j], self.sigma)
            else:
                tx = int(pose_coord[0, j])
                ty = int(pose_coord[1, j])
                if 0 <= tx < width and 0 <= ty < height:
                    pose_map[ty, tx, j] = 1.0
                    
        data['ref_seg_map'] = pose_map
        data['ref'] = np.flip(pose_coord, 0).astype(np.float32)
        
        return self.pipeline(data)

    @staticmethod
    def compute_pck(distAll, distThresh):

        pckAll = np.zeros((len(distAll), ))
        for pidx in range(len(distAll)):
            idxs = np.argwhere(distAll[pidx] <= distThresh)
            pck = 100.0 * len(idxs) / len(distAll[pidx])
            pckAll[pidx] = pck

        return pckAll

    def img2coord(self, imgs, topk=5):
        clip_len = len(imgs)
        height, width = imgs.shape[2:]
        assert imgs.shape[:2] == (clip_len, self.NUM_KEYPOINTS)
        coords = np.zeros((2, self.NUM_KEYPOINTS, clip_len), dtype=np.float)
        imgs = imgs.reshape(clip_len, self.NUM_KEYPOINTS, -1)
        assert imgs.shape[-1] == height * width
        # [clip_len, NUM_KEYPOINTS, topk]
        topk_indices = np.argsort(imgs, axis=-1)[..., -topk:]
        topk_values = np.take_along_axis(imgs, topk_indices, axis=-1)
        topk_values = topk_values / np.sum(topk_values, keepdims=True, axis=-1)
        topk_x = topk_indices % width
        topk_y = topk_indices // width
        # [clip_len, NUM_KEYPOINTS]
        coords[0] = np.sum(topk_x * topk_values, axis=-1).T
        coords[1] = np.sum(topk_y * topk_values, axis=-1).T
        coords[:, np.sum(imgs.transpose(1, 0, 2), axis=-1) == 0] = -1

        return coords

    def pck_evaluate(self, results, output_dir, logger=None):

        dist_all = [np.zeros((0, 0)) for _ in range(self.NUM_KEYPOINTS)]
        if terminal_is_available():
            prog_bar = mmcv.ProgressBar(len(self))
        for vid_idx in range(len(results)):
            cur_results = results[vid_idx]
            if isinstance(cur_results, str):
                file_path = cur_results
                cur_results = np.load(file_path)
                os.remove(file_path)
                coords = False
            else:
                coords = True
                
            pose_path = self.samples[vid_idx]['anno_path']
            gt_poses = sio.loadmat(pose_path)['pos_img'] - 1

            # get predict poses
            clip_len = self.samples[vid_idx]['num_frames']
            # truncate according to GT
            clip_len = min(clip_len, gt_poses.shape[-1])
                    
            # [2, 15, clip_len]
            if not coords:
                pred_poses = self.img2coord(cur_results)[:,:,:clip_len]
            else:
                pred_poses = cur_results[:,:,:clip_len]
            
            assert pred_poses.shape == gt_poses.shape, \
                f'{pred_poses.shape} vs {gt_poses.shape}'
            if output_dir is not None:
                for img_idx in range(clip_len):
                    mmcv.imwrite(
                        self.vis_pose(
                            mmcv.imread(self.samples[vid_idx]['frames_path'][img_idx]),
                            pred_poses[..., img_idx]),
                            osp.join(
                    output_dir, osp.relpath(self.samples[vid_idx]['video_path'], self.video_dir),
                    self.filename_tmpl.format(img_idx).replace(
                        'jpg', 'png'))
                        )
            # [15, clip_len]
            joint_visible = pred_poses[0] > 0
            # TODO verctorlized is slow or not? fast
            valid_max_gt_poses = gt_poses.copy()
            valid_max_gt_poses[:, ~joint_visible] = -1
            valid_min_gt_poses = gt_poses.copy()
            valid_min_gt_poses[:, ~joint_visible] = 1e6
            boxes = np.stack((valid_max_gt_poses[0].max(axis=0) -
                              valid_min_gt_poses[0].min(axis=0),
                              valid_max_gt_poses[1].max(axis=0) -
                              valid_min_gt_poses[1].min(axis=0)),
                             axis=0)
            # [clip_len]
            boxes = 0.6 * np.linalg.norm(boxes, axis=0)
            for img_idx in range(clip_len):
                for t in range(self.NUM_KEYPOINTS):
                    if not joint_visible[t, img_idx]:
                        continue
                    predx = pred_poses[0, t, img_idx]
                    predy = pred_poses[1, t, img_idx]
                    gtx = gt_poses[0, t, img_idx]
                    gty = gt_poses[1, t, img_idx]
                    dist = np.linalg.norm(
                        np.subtract([predx, predy], [gtx, gty]))
                    dist = dist / boxes[img_idx]

                    dist_all[t] = np.append(dist_all[t], [[dist]])
            if terminal_is_available():
                prog_bar.update()
        pck_ranges = (0.1, 0.2, 0.3, 0.4, 0.5)
        pck_all = []
        for pck_range in pck_ranges:
            pck_all.append(self.compute_pck(dist_all, pck_range))
        eval_results = {}
        
        with open(osp.join(output_dir, 'result.txt'), 'a') as f:
            for alpha, pck in zip(pck_ranges, pck_all):
                eval_results[f'PCK@{alpha}'] = np.mean(pck)
                f.write(f'PCK@{alpha}: {np.mean(pck)}')
        
        return eval_results

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
    
    def draw_label_map(self, img, pt, sigma):
        # Draw a 2D gaussian

        # Check that any part of the gaussian is in-bounds
        ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
        br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
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