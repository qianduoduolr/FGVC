import numpy as np
import av
import imageio
import os
import os.path as osp
import mediapy as media
import cv2
from glob import glob

def generate_gif(rgbs, gif_path, duration=1/15):

    frames = [rgbs[i] for i in range(rgbs.shape[0])]

    imageio.mimsave(gif_path, frames, 'GIF', duration=duration)


def generate_video(rgbs, path, fps=20):

    media.write_video(path, rgbs, fps=fps)
    
    

def get_video_frames_cv(v_path, size, is_decode=False):
    
    if not is_decode:
        vidcap = cv2.VideoCapture(v_path)
        nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        if nb_frames == 0: return None
        w = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        h = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

        success, image = vidcap.read()
        count = 1

        frames = []
        while success:
            image = cv2.resize(image, size, cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
            success, image = vidcap.read()
            count += 1

        vidcap.release()
    else:
        paths = sorted(glob(osp.join(v_path, '*.jpg')))
        frames = []
        for path in paths:
            image = cv2.imread(path)
            image = cv2.resize(image, size, cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
        
    return frames