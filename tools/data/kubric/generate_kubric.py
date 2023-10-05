import sys
import os
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm


sys.path.insert(0,  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


from mmpt.datasets.kubric_dataset.kubric.challenges.point_tracking import dataset

# root='/media/lr/dataset/tfds/movi_f/'
root='/home/sist/lirui/dataset/Kubric/'

max_dataset_size = 11000


def create_dataset(root):
        
     res = dataset.create_point_tracking_dataset(
          data_name='movi_f/512x512',
          data_dir=root,
          train_size=(512, 512),
          shuffle_buffer_size=None,
          split="train",
          batch_dims=[1],
          repeat=True,
          vflip=False,
          random_crop=True,
          tracks_to_sample=2048,
          sampling_stride=4,
          max_seg_id=25,
          max_sampled_frac=0.1,
          num_parallel_point_extraction_calls=16
          )

     samples = tfds.as_numpy(res)
     for idx, s in enumerate(samples):
          #     dst_path = f'/media/lr/dataset/tfds/movi_f/movi_f/512x512_split/{idx}.npy'
          #     dst_path = f'/home/sist/lirui/dataset/Kubric/movi_f/512x512_split/{idx}.npy'
          
          #     np.save(dst_path, s)
          
          yield s

   

ds = create_dataset(root)
    
for idx in range(max_dataset_size):
     s = ds.__next__()
     dst_path = f'/home/sist/lirui/dataset/Kubric/movi_f/512x512_split/{idx}.npy'
     np.save(dst_path, s)
    