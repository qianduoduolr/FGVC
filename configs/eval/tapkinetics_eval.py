import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
from mmpt.utils import *
from datetime import timedelta


exp_name = 'res18_d1_tapkinetics_eval'
docker_name = 'bit:5000/lirui_torch1.8_cuda11.1_corres'

# model settings
model = dict(
    type='HRVanillaTracker',
    backbone=dict(type='ResNet',depth=18, strides=(1, 1, 1, 4), out_indices=(2, ), pool_type='none'),
)

model_test = None

# model training and testing settings
train_cfg = dict(syncbn=True)

test_cfg = dict(
    precede_frames=5,
    topk=10,
    temperature=0.07,
    out_indices=(2, ),
    neighbor_range=30,
    with_first=True,
    with_first_neighbor=True,
    output_dir='eval_results')


# dataset settings
train_dataset_type = 'VOS_davis_dataset_test'

val_dataset_type =  'TAPVidDataset'

test_dataset_type =  'TAPVidDataset'


# train_pipeline = None
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
img_norm_cfg_lab = dict(mean=[50, 0, 0], std=[50, 127, 127], to_bgr=False)

train_pipeline = [
    dict(type='RandomResizedCrop', area_range=(0.6,1.0), aspect_ratio_range=(1.5, 2.0),),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='RGB2LAB', output_keys='images_lab'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Normalize', **img_norm_cfg_lab, keys='images_lab'),
    dict(type='FormatShape', input_format='NPTCHW'),
    dict(type='FormatShape', input_format='NPTCHW', keys='images_lab'),
    dict(type='Collect', keys=['imgs', 'images_lab'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'images_lab'])
]

val_pipeline = [
    dict(type='Resize', scale=(256, 256), keep_ratio=False, keys='video'),
    dict(type='RGB2LAB', keys='video', output_keys='video'),
    dict(type='Normalize', **img_norm_cfg_lab, keys='video'),
]

# demo_pipeline = None
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=8, drop_last=True),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train =  dict(
            type=train_dataset_type,
            root='/home/sist/lirui/dataset/TAP-Vid/tapvid_kinetics/all_split',
            list_path=None,
            tapvid_subset_name='kinetics',
            query_mode='first',
            pipeline=val_pipeline,
            test_mode=True
            ),

    test =  dict(
            type=test_dataset_type,
            root='/home/sist/lirui/dataset/TAP-Vid/tapvid_kinetics/all_split',
            list_path=None,
            tapvid_subset_name='kinetics',
            query_mode='first',
            pipeline=val_pipeline,
            test_mode=True
            ),
    
    val =  dict(
            type=val_dataset_type,
            root='/home/sist/lirui/dataset/TAP-Vid/tapvid_kinetics/all_split',
            list_path=None,
            tapvid_subset_name='kinetics',
            query_mode='first',
            pipeline=val_pipeline,
            test_mode=True
            ),
)
# optimizer
optimizers = dict(
    backbone=dict(type='Adam', lr=0.001, betas=(0.9, 0.999))
    )
# learning policy
# total_iters = 200000
runner_type='epoch'
max_epoch=1600
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.001,
    by_epoch=False,
    warmup_iters=10,
    warmup_ratio=0.1,
    warmup_by_epoch=True
    )

checkpoint_config = dict(interval=1600, save_optimizer=True, by_epoch=True)
# remove gpu_collect=True in non distributed training
# evaluation = dict(interval=1000, save_image=False, gpu_collect=False)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False, interval=10),
    ])

visual_config = None


# runtime settings
dist_params = dict(backend='nccl', timeout=timedelta(seconds=6400000))
log_level = 'INFO'
work_dir = f'/home/sist/lirui/expdir/mmpt/group_stsl/{exp_name}'


evaluation = dict(output_dir=f'{work_dir}/eval_output_val', interval=800, by_epoch=True
                  )

eval_config= dict(
                  output_dir=f'{work_dir}/eval_output',
                  checkpoint_path='/path/to/ckpt',
                  torchvision_pretrained=None
                )

eval_arc = 'none'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
test_mode = True



if __name__ == '__main__':

    make_local_config(exp_name, file='eval')