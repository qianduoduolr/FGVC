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
from glob import glob
import pickle

from collections import *
import pdb
from mmcv.utils import print_log
import mmcv
from mmpt.utils import *
from mmpt.datasets.flyingthingsplus.utils.figures import *
from mmpt.datasets.flyingthingsplus.utils import improc as pips_vis

from tensorboardX import SummaryWriter

from .base_dataset import BaseDataset
from .video_dataset import *
from .registry import DATASETS
from .tapvid_evaluation_datasets import *
from .flyingthingsplus.utils import visualize




@DATASETS.register_module()
class TAPVidDataset(Video_dataset_base):
    """
    An iterator that loads a TAP-Vid dataset and yields its elements.
    The elements consist of videos of arbitrary length.
    """

    def __init__(self, tapvid_subset_name, query_mode, size=(24, 256, 256, 3), input_size=(256, 256), vis_traj=False, *args, **kwargs):
        super().__init__( *args, **kwargs)

        self.tapvid_subset_name = tapvid_subset_name
        self.query_mode = query_mode
        self.size = size
        self.input_size = input_size
        self.vis_traj = vis_traj
        self.load_annotations()
          
    def load_annotations(self):
        
        logger = get_root_logger()
        
        self.samples = glob(osp.join(self.root, '*.pkl'))
            
        # self.samples = self.samples[4:8]
                            
        logger.info(f"Loaded TAPVid-{self.tapvid_subset_name} dataset")
        logger.info('found %d unique videos' % (len(self.samples)))
        
        return 0
    
    def prepare_test_data(self, idx):
        
        video_path = self.samples[idx]
        sample = mmcv.load(video_path)
        
        
        if isinstance(sample['video'][0], bytes):
            # Tapnet is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            sample['video'] = np.array([decode(sample['video'][i]) for i in range(sample['video'].shape[0])])
        else:
            sample['video'] = [ sample['video'][i] for i in range(sample['video'].shape[0])]
        
        # resize and normalize video
        sample = self.pipeline(sample)
        frames = np.array(sample['video'])
        
        target_points = sample['points']
        target_occ = sample['occluded']
        target_points *= np.array([self.input_size[1], self.input_size[0]])

        if self.query_mode == 'strided':
            data = sample_queries_strided(target_occ, target_points, frames)
        elif self.query_mode == 'first':
            data = sample_queries_first(target_occ, target_points, frames)
        else:
            raise ValueError(f'Unknown query mode {self.query_mode}.')
        
        data = TAPVidDataset.preprocess_dataset_element(data)

        return data


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
        
        writer = SummaryWriter(os.path.join(output_dir, 'traj'), max_queue=10, flush_secs=60)
        log_freq = 99999
        sw = pips_vis.Summ_writer(  
                                writer=writer,
                                log_freq=log_freq,
                                fps=24,
                                scalar_freq=int(log_freq/2),
                                just_gif=True
                                )
        
        for vid_idx in range(len(results)):
            
            video_path = self.samples[vid_idx]
            sample = mmcv.load(video_path)
            
            if isinstance(sample['video'][0], bytes):
                # Tapnet is stored and JPEG bytes rather than `np.ndarray`s.
                def decode(frame):
                    byteio = io.BytesIO(frame)
                    img = Image.open(byteio)
                    return np.array(img)

                rgbs = np.array([decode(sample['video'][i]) for i in range(sample['video'].shape[0])])
            else:
                rgbs = np.array([sample['video'][i] for i in range(sample['video'].shape[0])])
                
            rgbs = [ mmcv.imresize(rgbs[i], size=(self.size[2], self.size[1])) for i in range(rgbs.shape[0])]
            rgbs = np.stack(rgbs, 0)
            
            trajectories_gt, visibilities_gt, trajectories_pred, visibilities_pred, query_points = results[vid_idx]
            num_points = trajectories_gt.shape[2]
            
            # back to 256x256 for inference on TAP-Vid
            trajectories_gt[..., 0] = trajectories_gt[..., 0] * self.size[2] / self.input_size[1] 
            trajectories_gt[..., 1] = trajectories_gt[..., 1] * self.size[1] / self.input_size[0] 
            trajectories_pred[..., 0] = trajectories_pred[..., 0] * self.size[2] / self.input_size[1] 
            trajectories_pred[..., 1] = trajectories_pred[..., 1] * self.size[1] / self.input_size[0] 
            
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
            
            video = visualize.paint_point_track(rgbs, trajectories_pred[0].transpose(0,1).cpu().numpy(), visibilities_gt[0].transpose(0,1).cpu().numpy())
            
            # generate_gif(video, f'vis/a{vid_idx}.gif')
            if self.vis_traj:
                generate_video(video, os.path.join(output_dir, f'{vid_idx}.mp4'))
                
                rgbs_input = torch.from_numpy(rgbs)[None].permute(0,1,4,2,3).cuda()
                rgbs_input  = rgbs_input.float() * 1./255 - 0.5
                # gray_rgbs = torch.mean(rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
                
                if vid_idx % 1 == 0:
                    vid_path = os.path.join(output_dir,f'{vid_idx}')
                    os.makedirs(vid_path, exist_ok=True)
                    
                    for n in range(num_points):
                        
                        start_idx = 0
                        for i in range(trajectories_gt.shape[1]):
                            if visibilities_gt[0,i,n]:
                                start_idx = i
                                break
                        
                         # color point/line
                        rgbs = sw.summ_traj2ds_on_rgbs(f'{output_dir}/kp{vid_idx}{n}_trajs_e_on_rgbs', trajectories_pred[0:1,start_idx:,n:n+1], rgbs_input[0:1,start_idx:], cmap='spring', linewidth=2, only_return=True)  
                        
                        rgbs = rgbs[0].permute(0,2,3,1).detach().cpu().numpy()
                        
                        generate_video(rgbs, os.path.join(output_dir, f'{vid_idx}_{n}.mp4'))
    
        
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
    dataset = metadata["dataset"]
    os.makedirs(output_dir, exist_ok=True)
    summaries_path = os.path.join(output_dir, f"summaries{dataset}.json")
    with open(summaries_path, "w", encoding="utf8") as f:
        json.dump(summaries, f)
    print(f"\nSummaries saved to:\n{summaries_path}\n")

    # Save results summary dataframe as a csv file
    results_df_path = os.path.join(output_dir, f"results_df{dataset}.csv")
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
        results_list_pkl_path = os.path.join(output_dir, f"results_list{dataset}.pkl")
        with open(results_list_pkl_path, "wb") as f:
            print(f"\nResults pickle file saved to:\n{results_list_pkl_path}")
            pickle.dump(results_list, f)

    # Make figures
    figures_dir = os.path.join(output_dir, f"figures{dataset}")
    ensure_dir(figures_dir)
    result = make_figures(results_df, figures_dir, mostly_visible_threshold)
    
    return result