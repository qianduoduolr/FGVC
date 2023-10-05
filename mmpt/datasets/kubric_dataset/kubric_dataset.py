import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import copy
import wandb
import json

import torch
from torch.utils import data

import random
import glob
import pickle

from collections import *
import pdb
from mmcv.utils import print_log
import mmcv
from mmpt.utils import *
from mmpt.datasets.flyingthingsplus.utils.figures import *
from tensorboardX import SummaryWriter

from ..base_dataset import BaseDataset
from ..video_dataset import *
from ..registry import DATASETS
from ..tapvid_evaluation_datasets import *




@DATASETS.register_module()
class KubricDataset(Video_dataset_base):
    """
    An iterator that loads a TAP-Vid dataset and yields its elements.
    The elements consist of videos of arbitrary length.
    """

    def __init__(self,  size=(24, 256, 256, 3), query_num=32, mode="", *args, **kwargs):
        super().__init__( *args, **kwargs)

        self.size = size
        self.mode = mode
        self.query_num = query_num
        
        self.load_annotations()
        
    def __len__(self):
        return len(self.samples)
          
    def load_annotations(self):
        
        logger = get_root_logger()
        
        self.samples = glob.glob(osp.join(self.root, '*.pkl'))
                      
        logger.info(f"Loaded Kubric dataset")
        logger.info('found %d unique videos' % (len(self.samples)))
        
        return 0
        
    
    
    def prepare_train_data(self, idx):
            
        video_path = self.samples[idx]
        sample = mmcv.load(video_path)
        sample['imgs'] = [ (sample['video'][0][i] + 1) * 255 / 2 for i in range(sample['video'].shape[1])]
        
        # a = sample['video'][0].astype(np.uint8)
        # b = sample['video'][1].astype(np.uint8)
        # c = sample['video'][2].astype(np.uint8)
        

        query_points = sample['query_points'][0]
        target_points = sample['target_points'][0]
        target_occ = sample['occluded'][0]
        
        ts = np.unique(query_points[:,0])
        ts = ts[(self.size[0] - ts - self.clip_length) >= 0]
        start = int(random.choice(ts.tolist()))
    
        point_mask = query_points[:, 0] == start
        target_points = target_points[point_mask]
        target_occ = target_occ[point_mask]
        
        if target_points.shape[0] >= self.query_num:
            select_idxs = random.choices(range(self.query_num), k=self.query_num)
            target_points = target_points[select_idxs]
            target_occ = target_occ[select_idxs]
        else:
            n_missing = self.query_num - target_points.shape[0]
            target_points = np.concatenate([target_points, target_points[-1:].repeat(n_missing, axis=0)], 0)
            target_occ = np.concatenate([target_occ, target_occ[-1:].repeat(n_missing, axis=0)], 0)
            
        sample['trajs'] = np.transpose(target_points, (1,0,2))[start:start+self.clip_length]
        sample['visibles'] = ~np.transpose(target_occ, (1,0))[start:start+self.clip_length]
        sample['imgs'] = sample['imgs'][start:start+self.clip_length]
        
        
        data = {
            **sample,
            'num_clips': 1,
            'modality': 'RGB',
            'clip_len': self.clip_length
        }
        
        return self.pipeline(data)        


    
    

    @staticmethod
    def preprocess_dataset_element(dataset_element):
        rgbs = torch.from_numpy(dataset_element['video']).permute(0, 1, 4, 2, 3)
        query_points = torch.from_numpy(dataset_element['query_points'])
        trajectories = torch.from_numpy(dataset_element['target_points']).permute(0, 2, 1, 3)
        visibilities = ~torch.from_numpy(dataset_element['occluded']).permute(0, 2, 1)

        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = query_points.shape[1]

        # Convert query points from (t, y, x) to (t, x, y)
        query_points = query_points[:, :, [0, 2, 1]]

        # Ad hoc fix for Kubric reporting invisible query points when close to the crop boundary, e.g., x=110, y=-1e-5
        for point_idx in range(n_points):
            query_point = query_points[0, point_idx]
            query_visible = visibilities[0, query_point[0].long(), point_idx]
            if query_visible:
                continue

            x, y = query_point[1:]
            x_at_boundary = min(abs(x - 0), abs(x - (width - 1))) < 1e-3
            y_at_boundary = min(abs(y - 0), abs(y - (height - 1))) < 1e-3
            x_inside_window = 0 <= x <= width - 1
            y_inside_window = 0 <= y <= height - 1

            if x_at_boundary and y_inside_window or x_inside_window and y_at_boundary or x_at_boundary and y_at_boundary:
                visibilities[0, query_point[0].long(), point_idx] = 1

        # Check dimensions are correct
        assert batch_size == 1
        assert rgbs.shape == (batch_size, n_frames, channels, height, width)
        assert query_points.shape == (batch_size, n_points, 3)
        assert trajectories.shape == (batch_size, n_frames, n_points, 2)
        assert visibilities.shape == (batch_size, n_frames, n_points)

        # Check that query points are visible
        assert torch.all(visibilities[0, query_points[0, :, 0].long(), torch.arange(n_points)] == 1), \
            "Query points must be visible"

        # Check that query points are correct
        assert torch.allclose(
            query_points[0, :, 1:].float(),
            trajectories[0, query_points[0, :, 0].long(), torch.arange(n_points)].float(),
            atol=1.0,
        )

        return {
            "rgbs": rgbs[0],
            "query_points": query_points[0],
            "trajectories": trajectories[0],
            "visibilities": visibilities[0],
        }
        
        
    def evaluate(self, results, metrics='tapvid', output_dir=None, logger=None):

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['pck','tapvid']
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
                        self.tapvid_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        else:
            eval_results.update(self.tapvid_evaluate(results, output_dir, logger))

        return eval_results
    
    def tapvid_evaluate(self, results, output_dir, logger=None):
        
        if terminal_is_available():
            prog_bar = mmcv.ProgressBar(len(self))
            
        summaries = []
        results_list = []
        
        for vid_idx in range(len(results)):
            
            trajectories_gt, visibilities_gt, trajectories_pred, visibilities_pred, query_points = results[vid_idx]
            num_points = trajectories_gt.shape[2]
            b = 1
            
            unpacked_results = []
            
            for n in range(num_points):
                unpacked_result = {
                        "idx": f'{vid_idx}_{n}',
                        "iter": vid_idx,
                        "video_idx": 0,
                        "point_idx_in_video": n,
                        "trajectory_gt": trajectories_gt[0, :, n, :].detach().clone().cpu(),
                        "trajectory_pred": trajectories_pred[0, :, n, :].detach().clone().cpu(),
                        "visibility_gt": visibilities_gt[0, :, n].detach().clone().cpu(),
                        "visibility_pred": visibilities_pred[0, :, n].detach().clone().cpu(),
                        "query_point": query_points[0, n, :].detach().clone().cpu(),
                    }
                unpacked_results.append(unpacked_result)
            
            summaries_batch = [compute_summary(res, self.query_mode) for res in unpacked_results] # query_mode ?
            summaries += summaries_batch

            summary_df = compute_summary_df(unpacked_results)
            selected_metrics = ["ade_visible", "average_jaccard", "average_pts_within_thresh", "occlusion_accuracy"]
            selected_metrics_shorthand = {
                "ade_visible": "ADE",
                "average_jaccard": "AJ",
                "average_pts_within_thresh": "<D",
                "occlusion_accuracy": "OA",
            }
            print(summary_df[selected_metrics].to_markdown())
        
        
        metadata = {
        "name": 'tapvid_evaluate',
        "model": 'none',
        "dataset": f"{self.tapvid_subset_name}",
        "query_mode": self.query_mode,
        }
        
        result = save_results(summaries, results_list, output_dir, 4, metadata)
        
        return result
            


def save_results(summaries, results_list, output_dir, mostly_visible_threshold, metadata):
    # Save summaries as a json file
    os.makedirs(output_dir, exist_ok=True)
    summaries_path = os.path.join(output_dir, "summaries.json")
    with open(summaries_path, "w", encoding="utf8") as f:
        json.dump(summaries, f)
    print(f"\nSummaries saved to:\n{summaries_path}\n")

    # Save results summary dataframe as a csv file
    results_df_path = os.path.join(output_dir, "results_df.csv")
    results_df = pd.DataFrame.from_records(summaries)
    results_df.to_csv(results_df_path)
    print(f"\nResults summary dataframe saved to:\n{results_df_path}\n")
    for k, v in metadata.items():
        results_df[k] = v

    # # Save results summary dataframe as a wandb artifact
    # artifact = wandb.Artifact(name=f"{wandb.run.name}__results_df", type="df", metadata=metadata)
    # artifact.add_file(results_df_path, "results_df.csv")
    # wandb.log_artifact(artifact)

    # Save results list as a pickle file
    if len(results_list) > 0:
        results_list_pkl_path = os.path.join(output_dir, "results_list.pkl")
        with open(results_list_pkl_path, "wb") as f:
            print(f"\nResults pickle file saved to:\n{results_list_pkl_path}")
            pickle.dump(results_list, f)

    # Make figures
    figures_dir = os.path.join(output_dir, "figures")
    ensure_dir(figures_dir)
    result = make_figures(results_df, figures_dir, mostly_visible_threshold)
    
    return result