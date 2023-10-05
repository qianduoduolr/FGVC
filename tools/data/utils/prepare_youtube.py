import numpy as np
import glob
import os.path as osp
import mmcv
import json
import os

year = '2018'
imset = ['valid_all_frames']

resolution = '480p'
root = '/home/lr/dataset/YouTube-VOS/2018'
dst_path = f'/home/lr/dataset/YouTube-VOS/2018/{imset[0]}'



def to_list():
    for mode in imset:
        anno_path = osp.join(root, mode, 'Annotations')
        frame_path = osp.join(root, mode, 'JPEGImages_s256')   
        with open(f'youtube{year}_{mode}_list.txt','a') as f:
        
            videos_subset = glob.glob(osp.join(frame_path, '*'))
            for video in videos_subset:

                video_name = video.split('/')[-1]

                frame_num = len(glob.glob(osp.join(video, '*.jpg')))
                f.write(f'{video_name} {frame_num}' + '\n')


def to_json():
    
    for mode in imset:
        data = dict()
        anno_path = osp.join(root, mode, 'Annotations')
        frame_path = osp.join(root, mode, 'JPEGImages_s256')   
        videos_subset = glob.glob(osp.join(frame_path, '*'))
        for video in videos_subset:
            video_name = video.split('/')[-1]
            frames = sorted(glob.glob(osp.join(video, '*.jpg')))
            data[video_name] = []
            for frame in frames:
                x = os.path.basename(frame)
                data[video_name].append(x)
        
        
        with open(f'{dst_path}/youtube{year}_{mode}.json','w') as f:
            _ = mmcv.dump(data, f, file_format='json')

def check():
    vs = glob.glob(os.path.join(root, 'train', 'JPEGImages', '*'))
    for v in vs:
        x = v.replace('JPEGImages', 'JPEGImages_s256')
        a1 = len(glob.glob(os.path.join(v, '*.jpg')))
        a2 = len(glob.glob(os.path.join(x, '*.jpg')))
        if a1 != a2: print(v)

to_json()
# check()