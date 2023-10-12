import mmcv
from glob import glob
import os
count = 0
files = glob(os.path.join('data/tapvid_kinetics/pkl_datas/','*.pkl'))
for file in files:
    data = mmcv.load(file)
    for idx, v in enumerate(data):
        mmcv.dump(v, f'data/TAP-Vid/tapvid_kinetics/data_split/{count}.pkl')
        count += 1

count = 0     
files = glob(os.path.join('data/tapvid_davis/pkl_datas/','*.pkl'))
for file in files:
    data = mmcv.load(file)
    for idx, v in enumerate(data):
        mmcv.dump(v, f'data/TAP-Vid/tapvid_davis/data_split/{count}.pkl')
        count += 1