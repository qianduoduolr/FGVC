FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel


ENV TORCH_CUDA_ARCH_LIST="3.5 3.7 5.2 6.0 6.1 7.0+PTX"

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH

COPY ./DATA.zip .


RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
	APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple" && \
    GIT_CLONE="git clone --depth 10" && \
    CONDA_INSTALL="conda install -y" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        apt-utils \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        htop \
        tmux \
        openssh-client \
        openssh-server \
        libboost-dev \
        libjpeg-dev \
        zlib1g-dev \
        libboost-thread-dev \
        libboost-filesystem-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        zip \
        unzip \
        rsync \
        && \

# ==================================================================
# conda
# ------------------------------------------------------------------
   # conda config --set show_channel_urls yes && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --set show_channel_urls yes && \
    $CONDA_INSTALL \
        python=3.8.8 && \
    pip install  torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --upgrade pip && \
    $PIP_INSTALL \
        attr \
        DALL-E \
        timm \
        numpy \
        opencv_python \
        scikit-learn \
        scipy \
        matplotlib \
        plyfile \
        nvidia-ml-py \
        yacs \
        tqdm \
	tensorflow \
	tensorflow_datasets \
        tensorboardX \
        argparse \
        sacred \
        progressbar2 \
        path.py \
        h5py \
        blessings \
        scikit-image \
        lz4 \
        lmdb\
        fitlog\
        appdirs\
        cachetools\
        certifi\
        cffi\
        chardet\
        cycler\
        Cython\
        decorator\
        easydict\
        fire\
        google-auth\
        google-auth-oauthlib\
        termcolor\
        got10k\
        grpcio\
        idna\
        imgaug\
        resnest\
        wandb \
	imageio\
	einops\
	protobuf==3.19.* \
        fire\	 
        pytorch-lightning==1.7.0 \
        tensorflow-graphics \
	seaborn \
	mediapy \
	terminaltables \
	&& \
    conda clean -y --all && \
# ==================================================================
# Pillow-SIMD
# ------------------------------------------------------------------
    pip uninstall -y pillow && \
    pip install pillow-simd && \

# ==================================================================
# A  bug about cv2
# ------------------------------------------------------------------
    apt-get update &&\
    apt-get install -y libgl1-mesa-dev &&\


# ==================================================================
# mmcv installtion
# ------------------------------------------------------------------
    pip install  mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html &&\
    pip install future tensorboard &&\
   


# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/* &&\



# ==================================================================
# video loader
# ------------------------------------------------------------------
    pip install pytorchvideo &&\
    pip install decord &&\
    pip install pims &&\
    pip install moviepy &&\


# ==================================================================
# davis
# ------------------------------------------------------------------
    git clone https://github.com/davisvideochallenge/davis2017-evaluation.git &&\
    cd davis2017-evaluation &&\
    python setup.py develop

EXPOSE 6006
