## Data Preparation
#### FlyingThings
We create FlyingThings dataset for training. Please first download FlyingThings from [this link](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). Note we only need the `frames_cleanpass_webp` and `optical_flow` in our training.

#### YouTube-VOS
Please download the zip file `train.zip` from the [official website](https://competitions.codalab.org/competitions/19544#participate-get-data). Then, unzip and place it to`data/ytv`. Besides, please move the `youtube2018_train.json` in `data/data_info/` to `data/YouTube-VOS`.

#### TAP-Vid
Please follow the instructions in [this link](https://github.com/google-deepmind/tapnet) to download the TAP-Vid-DAVIS and TAP-Vid-Kinetics. 
Once these dataset have been downloaded to `TAP-Vid/tapvid_davis/pkl_datas` and `TAP-Vid/tapvid_kinetics/pkl_datas`, we split these dataset (`*.pkl` files) with `tools/data/tapvid/split_pickle.py`

#### DAVIS-2017
DAVIS-2017 dataset could be downloaded from the [official website](https://davischallenge.org/davis2017/code.html). We use the 480p validation set for evaluation. Please move the `davis2017_val_list.json` in `data/data_info/` to `data/DAVIS/ImageSets`.
#### JHMDB
Please download the data (`Videos`, `Joint positions`) from [official website](http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets), unzip and place them in `data/jhmdb`. Please move the `val_list.txt` in `data/data_info/` to `data/JHMDB`.

#### BADJA
To evaluate the model in BAJDA, first follow the instructions at the [BADJA repo](https://github.com/benjiebob/BADJA). This will involve downloading DAVIS trainval full-resolution data. 

The overall data structure is as followed:

```shell
├── data
│   ├── FlyingThings3D
│   │   ├── frames_cleanpass_webp
│   │   │   ├── TRAIN
│   │   │   │   ├──A/
│   │   │   │   ├──...
│   │   ├── optical_flow
│   │   │   ├── TRAIN
│   │   │   │   ├──A/
│   │   │   │   ├──...
│   ├── YouTube-VOS
│   │   ├── train
│   │   │   ├── JPEGImages
│   │   │   │   ├──003234408d/
│   │   │   │   ├──...
│   │   │   ├── youtube2018_train.json
│   ├── TAP-Vid
│   │   ├── tapvid_davis
│   │   │   ├── pkl_datas
│   │   │   │   ├──*.pkl
│   │   │   │   ├──...
│   │   │   ├── data_split
│   │   │   │   ├──1.pkl
│   │   │   │   ├──2.pkl
│   │   │   │   ├──...
│   │   ├── tapvid_kinetics
│   │   │   ├── pkl_datas
│   │   │   │   ├──*.pkl
│   │   │   │   ├──...
│   │   │   ├── data_split
│   │   │   │   ├──1.pkl
│   │   │   │   ├──2.pkl
│   │   │   │   ├──...
│   ├── DAVIS
│   │   ├── Annotations
│   │   │   ├── 480p
│   │   │   │   ├── bike-packing/
│   │   │   │   ├── ...
│   │   ├── ImageSets
│   │   │   ├── davis2017_val_list.json
│   │   │   ├── ...
│   │   ├── JPEGImages
│   │   │   ├── 480p
│   │   │   │   ├── bike-packing/
│   │   │   │   ├── ...
│   ├── JHMDB
│   │   ├── Rename_Images
│   │   │   ├── brush_hair/
│   │   │   ├── ...
│   │   ├── joint_positions
│   │   │   ├── brush_hair/
│   │   │   ├── ...
│   │   ├── val_list.txt
│   ├── DAVIS_Full_Res
│   │   ├── Annotations
│   │   │   ├── Full-Resolution
│   │   │   │   ├── bike-packing/
│   │   │   │   ├── ...
│   │   ├── ImageSets
│   │   │   ├── 2016
│   │   │   │   ├── train.txt
│   │   │   │   ├── val.txt
│   │   │   ├── ...
│   │   ├── JPEGImages
│   │   │   ├── Full-Resolution
│   │   │   │   ├── bike-packing/
│   │   │   │   ├── ...
│   │   ├── joint_annotations
│   │   │   ├── bear.json
│   │   │   ├── ...
```