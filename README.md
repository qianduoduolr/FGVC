## [ICCV 2023] Learning Fine-Grained Features for Pixel-wise Video Correspondences

[Rui Li](https://qianduoduolr.github.io/)<sup>1</sup>, Shenglong Zhou<sup>1</sup>, and [Dong Liu](https://faculty.ustc.edu.cn/dongeliu/en/index/85593/list/index.htm)<sup>1</sup>, 


<sup>1</sup>University of Science and Technology of China, Hefei, China

##### [Paper](https://arxiv.org/pdf/) | [Video (comming soon)](https://www.youtube.com/)



<p float="left">
<img src="figure/pt1.gif" width = "230" height = "160">
<img src="figure/pt3.gif" width = "230" height = "160">
<!-- <img src="figure/pt2.gif" width = "230" height = "160"> -->
<img src="figure/vos1.gif" width = "230" height = "160">

This is the official code for  "**Learning Fine-Grained Features for Pixel-wise Video Correspondences**" (ICCV'23).

<!-- ![](figure/framework.png) -->

<div  align="center">    
<img src="figure/framework.png"  height="340px"/> 
</div>



### Citation
If you find this repository useful for your research, please cite our paper:

```latex
@inproceedings{lilearn,
  title={Learning Fine-Grained Features for Pixel-wise Video Correspondences},
  author={Li, Rui and Liu, Dong},
  booktitle={ICCV},
  year={2023}
}

```
### Prerequisites

* Python 3.8.8
* PyTorch 1.9.1
* mmcv-full == 1.5.2
* davis2017-evaluation


To get started, first please clone the repo
```
git clone https://github.com/qianduoduolr/FGVC
```
Then, please run the following commands:
```
conda create -n fgvc python=3.8.8
conda activate fgvc

pip install  torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install  mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install -r requirements.txt
pip install future tensorboard

# setup for davis evaluation
git clone https://github.com/davisvideochallenge/davis2017-evaluation.git && cd davis2017-evaluation
python setup.py develop
```