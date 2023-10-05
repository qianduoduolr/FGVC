import sys
sys.path.insert(0, '../raft/core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from torch.nn.parallel import DistributedDataParallel
import mmcv

from tqdm import tqdm

DEVICE = 'cuda'
bound = 20
target = 256

def load_image(imfile):
    img = Image.open(imfile)
    w, h = img.size
    ratio = w / h

    # if w != target and h != target:
    #     if ratio >= 1:
    #         img = img.resize((int(target * w/h), target))
    #     else:
    #         img = img.resize((target, int(target * h/w)))

    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def main(args):
    num_gpu = args.num_gpu
    
    model = RAFT(args).cuda()
    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    vid_files = glob.glob(os.path.join(args.path,'*'))


    if args.split == -1:
        per_samples = len(vid_files) // num_gpu
        sub_files = vid_files[args.local_rank * per_samples:(args.local_rank+1) * per_samples] if args.local_rank != num_gpu -1 else \
            vid_files[args.local_rank * per_samples:]
    else:
        per_samples = len(vid_files) // 2
        sub_files = vid_files[args.split * per_samples:(args.split+1) * per_samples] if args.split != 1 else \
            vid_files[args.split * per_samples:]
    
    with torch.no_grad():
        for vid_file in tqdm(sub_files, total=len(sub_files)):
            images = glob.glob(os.path.join(vid_file, '*.png')) + \
                    glob.glob(os.path.join(vid_file, '*.jpg'))

            images = sorted(images)
            # print(len(images))
            video_path = vid_file.replace('JPEGImages_s256','Flows_flo_s256').replace('gdata',  'gdata1')
            # if len(glob.glob(os.path.join(video_path, '*.jpg'))) == len(images) -1: continue

            os.makedirs(video_path, exist_ok=True)
            
            out_flows = []
            for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
                for mode in ['forward', 'backward']:
                    
                    image1 = load_image(imfile1)
                    image2 = load_image(imfile2)
                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)

                    if mode == 'forward':
                        flow_low, flow_up = model(image1, image2, iters=30, test_mode=True)
                    else:
                        flow_low, flow_up = model(image2, image1, iters=30, test_mode=True)
                    
                    flow_up = padder.unpad(flow_up)
                         
                        
                    empty_img = 128 * np.ones((int(image1.shape[2]),int(image1.shape[3]),3)).astype(np.uint8)

                    base_name = os.path.basename(imfile1)
                    dst_path = os.path.join(video_path, f'{mode}_{base_name}')

                    if args.norm == 'min-max':
                        flow = np.clip(flow_up.permute(0,2,3,1).cpu().numpy(), -bound, bound)
                        flow = (flow - flow.min()) * 255.0 / (flow.max() - flow.min())
                        flow = np.round(flow).astype('uint8')
                        flow_img = empty_img.copy()
                        flow_img[:,:,:2] = flow[:]
                        cv2.imwrite(dst_path, flow_img)

                    elif args.norm == '0-1':
                        flow = np.clip(flow_up.permute(0,2,3,1).cpu().numpy(), -bound, bound)
                        flow = (flow + bound) * (255.0 / (2*bound))
                        flow = np.round(flow).astype('uint8')
                        flow_img = empty_img.copy()
                        flow_img[:,:,:2] = flow[:]
                        cv2.imwrite(dst_path, flow_img)

                    else:
                        flow_img = flow_up[0].permute(1,2,0).cpu().numpy()
                        mmcv.flowwrite(flow_img, dst_path.replace('jpg', 'flo'))
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='/gdata/lirui/models/optical_flow/raft-things.pth')
    parser.add_argument('--path', help="dataset for evaluation", default='/gdata/lirui/dataset/YouTube-VOS/2018/train/JPEGImages_s256')
    parser.add_argument('--out', help="dataset for evaluation", default='/gdata1/lirui/dataset/YouTube-VOS/2018/train/Flows_flo_s256')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--local_rank',  type=int, help='use small model')
    parser.add_argument('--num-gpu',  type=int, default=1, help='use small model')
    parser.add_argument('--split',  type=int, default=-1, help='use small model')
    parser.add_argument('--norm',  type=str, default='none', help='use small model')

    

    args = parser.parse_args()

    print(args.local_rank)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    main(args)