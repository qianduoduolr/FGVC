import mmcv
from glob import glob
import os
count = 0
files = glob(os.path.join('/var/dataset/tapvid_kinetics/all','*.pkl'))
for file in files:
    data = mmcv.load(file)
    for idx, v in enumerate(data):
        mmcv.dump(v, f'/var/dataset/tapvid_kinetics/all_split/{count}.pkl')
        count += 1