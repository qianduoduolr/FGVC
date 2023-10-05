import pickle
from tqdm import tqdm
import numpy as np
from glob import glob
from joblib import delayed, Parallel
import sys
import os
import cv2
import lmdb
import argparse


def get_video_frames_cv(v_path, dataset='ucf101'):

    target = 256 # for kinetics
    vidcap = cv2.VideoCapture(v_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if nb_frames == 0: return None
    w = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float


    short_size = min(w, h)
    success, image = vidcap.read()
    count = 1

    if w >= h:
        size = (int(target * w / h), int(target))
    else:
        size = (int(target), int(target * h / w))

    frames = []
    while success:
        if dataset == 'kinetics':
            if short_size <= 256:
                image = cv2.resize(image, size, cv2.INTER_CUBIC)
            else:
                image = cv2.resize(image, size, cv2.INTER_AREA)

        frames.append(image)

        success, image = vidcap.read()
        count += 1

    vidcap.release()
    return frames

def compute_TVL1(prev, curr, bound=20):
    """Compute the TV-L1 optical flow."""

    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()

    flow = TVL1.calc(prev, curr, None)
    flow = np.clip(flow, -bound, bound)

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype('uint8')

    return flow

def create_lmdb_video_dataset_rgb(dataset, root_path, dst_path, workers=-1, quality=95, skip=1, video_type='mp4'):

    if dataset == 'kinetics': video_type = 'mp4' 
    else: video_type = 'avi'
    
    videos = glob(os.path.join(root_path,'*/*.{}'.format(video_type)))
    print('begin')
    
    def make_video(video_path, dst_path):
        vid_names = '/'.join(video_path.split('/')[-2:])
        dst_file = os.path.join(dst_path, vid_names[:-4])
        os.makedirs(dst_file, exist_ok=True)
        try:
            frames = get_video_frames_cv(video_path, dataset)
        except Exception as e:
            return
        else:
            if frames == None: return 
        
        frames = frames[::skip]

        _, frame_byte = cv2.imencode('.jpg', frames[0],  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        env = lmdb.open(dst_file, frame_byte.nbytes * len(frames) * 50)
        frames_num = len(frames)

        for i in range(frames_num):
            txn = env.begin(write=True)
            key = os.path.join(vid_names[:-4], 'image_{:05d}.jpg'.format((i+1) * skip))
            frame = frames[i]
            _, frame_byte = cv2.imencode('.jpg', frame,  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            txn.put(key.encode(), frame_byte)
            txn.commit()
        # except Exception as e:
        #     with open(os.path.join(dst_path, 'err_list.txt'), 'a') as f:
        #         f.write(vid_names + '\n')
        #     print(vid_names)
        #     return
        with open(os.path.join(dst_file, 'split.txt'),'w') as f:
            f.write(str(frames_num))

    Parallel(n_jobs=workers)(delayed(make_video)(vp, dst_path) for vp in tqdm(videos, total=len(videos)))


def create_lmdb_video_dataset_optical_flow(dataset, root_path, dst_path, workers=-1, quality=95):

    videos = glob(os.path.join(root_path,'*/*'))
    print('begin')
    
    def make_video_optical_flow(video_path, dst_path):
        vid_names = '/'.join(video_path.split('/')[-2:])
        dst_file = os.path.join(dst_path, vid_names)
        os.makedirs(dst_file, exist_ok=True)
        
        # load rgb frames from lmdb. You can change the code to load it in another way
        frames = []
        env = lmdb.open(video_path, readonly=True)
        txn = env.begin(write=False)
        for k,v in txn.cursor():
            frame_decode = cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_COLOR) 
            frames.append(frame_decode)
        env.close()

        height, width, _ = frames[0].shape
        empty_img = 128 * np.ones((int(height),int(width),3)).astype(np.uint8)
        # extract flows
        flows = []
        for idx, frame in enumerate(frames):
            if idx == 0: 
                pre_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                continue
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = compute_TVL1(pre_frame, frame_gray)
            # create flow frame with 3 channel
            flow_img = empty_img.copy()
            flow_img[:,:,0:2] = flow
            flows.append(flow_img)
            pre_frame = frame_gray

        # save flows
        _, frame_byte = cv2.imencode('.jpg', flows[0],  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        env = lmdb.open(dst_file, frame_byte.nbytes * len(flows) * 50)
        frames_num = len(flows)

        for i in range(frames_num):
            txn = env.begin(write=True)
            key = 'image_{:05d}.jpg'.format(i+1)
            flow_img = flows[i]
            _, frame_byte = cv2.imencode('.jpg', flow_img,  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            txn.put(key.encode(), frame_byte)
            txn.commit()
        with open(os.path.join(dst_file, 'split.txt'),'w') as f:
            f.write(str(frames_num))

    Parallel(n_jobs=workers)(delayed(make_video_optical_flow)(vp, dst_path) for vp in tqdm(videos, total=len(videos)))



def parse_option():
    parser = argparse.ArgumentParser('training')

    # dataset
    parser.add_argument('--root-path', type=str, default='//var/dataset/Kinetics_raw_video/', help='path of original data')
    parser.add_argument('--dst-path', type=str, default='/var/dataset/kinetics_s256_skip5_lmdb', help='path to store generated data')
    parser.add_argument('--dataset', type=str, default='kinetics', choices=['kinetics','ucf101'], help='dataset to training')
    parser.add_argument('--data-type', type=str, default='rgb', choices=['rgb','flow'], help='which data')
    parser.add_argument('--video-type', type=str, default='mp4', choices=['mp4', 'avi'], help='which data')
    parser.add_argument('--num-workers', type=int, default=-1, help='num of workers to use')
    parser.add_argument('--clip-length', type=int, default=16, help='num of clip length')
    parser.add_argument('--num-steps', type=int, default=1, help='num of sampling steps')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args =  parse_option()

    if args.data_type == 'rgb':
        create_lmdb_video_dataset_rgb(args.dataset, args.root_path, args.dst_path, workers=args.num_workers, video_type=args.video_type, skip=args.num_steps)
    elif args.data_type == 'flow':
        create_lmdb_video_dataset_optical_flow(args.dataset, args.root_path, args.dst_path, workers=args.num_workers)
    