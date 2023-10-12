exp_name = 'mixed_train_res18_d1_l2_rec_ytv_fly'

# model settings
model = dict(
    type='Mixed_Tracker',
    backbone=dict(type='ResNet',depth=18, strides=(1, 1, 1, 4), out_indices=(2,), pool_type='none'),
    teacher=dict(type='ResNet',depth=18, strides=(1, 1, 1, 4), out_indices=(2,), pool_type='none', frozen_stages=4, pretrained='/home/lr/project/fgvc/ckpt/epoch_40.pth', torchvision_pretrain=False),
    loss=dict(type='Soft_Ce_Loss'),
    loss_weight=dict(l1_loss=1, sup_loss=1, da_loss=0, corr_da_loss=1),
    downsample_rate=[2,],
    radius=[24,],
    temperature=1,
    feat_size=[128,],
    scale=2,
    pretrained=None,
)

# model training and testing settings
train_cfg = dict(syncbn=True)

val_cfg = dict(
    precede_frames=5,
    topk=10,
    temperature=0.07,
    out_indices=(2, ),
    neighbor_range=30,
    step=128,
    with_first=True,
    with_first_neighbor=True,
    output_dir='eval_results')

test_cfg = dict(
    precede_frames=5,
    topk=10,
    temperature=0.07,
    out_indices=(2, ),
    neighbor_range=30,
    step=128,
    with_first=True,
    with_first_neighbor=True,
    output_dir='eval_results')

# dataset settings
train_dataset_type = 'Flyingthings_ytv_dataset_rgb'
val_dataset_type =  'TAPVidDataset'
test_dataset_type =  'TAPVidDataset'

# train_pipeline = None
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
img_norm_cfg_lab = dict(mean=[50, 0, 0], std=[50, 127, 127], to_bgr=False)

train_pipeline = [
    dict(type='RandomResizedCrop', area_range=(0.6,1.0), aspect_ratio_range=(1.5, 2.0),),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='RandomGaussianBlur',p=0.8,same_across_clip=True,same_on_clip=True),
    dict(type='RGB2LAB'),
    dict(type='Normalize', **img_norm_cfg_lab),
    dict(type='FormatShape', input_format='NPTCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

train_pipeline_sup = [
    dict(type='RandomCrop', size=256),
    dict(type='RandomGaussianBlur',p=0.8,same_across_clip=True,same_on_clip=True),
    dict(type='RGB2LAB'),
    dict(type='Normalize', **img_norm_cfg_lab),
    dict(type='FormatShape', input_format='NPTCHW'),
    dict(type='Collect', keys=['imgs', 'flow', 'flow_back'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'flow', 'flow_back'])
]

val_pipeline = [
    # dict(type='Resize', scale=(512, 384), keep_ratio=False, keys='video'),
    dict(type='RGB2LAB', keys='video', output_keys='video'),
    dict(type='Normalize', **img_norm_cfg, keys='video'),
]

test_pipeline = [
    # dict(type='Resize', scale=(512, 384), keep_ratio=False, keys='video'),
    dict(type='RGB2LAB', keys='video', output_keys='video'),
    dict(type='Normalize', **img_norm_cfg, keys='video'),
]

# demo_pipeline = None
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=
            dict(
                type=train_dataset_type,
                root='/home/lr/dataset/YouTube-VOS',
                root_flow='/home/lr/dataset/FlyingThings3D',
                list_path='/home/lr/dataset/YouTube-VOS/2018/train',
                data_prefix=dict(RGB='train/JPEGImages_s256', FLOW=None, ANNO=None),
                clip_length=2,
                pipeline=train_pipeline,
                pipeline_sup=train_pipeline_sup,
                test_mode=False
                ),


    val =  dict(
            type=val_dataset_type,
            root='/home/lr/dataset/TAP-Vid/tapvid_davis/data_split',
            list_path=None,
            tapvid_subset_name='davis',
            query_mode='first',
            pipeline=val_pipeline,
            test_mode=True,
            # input_size=(384, 512)
            ),
    
    test =  dict(
            type=test_dataset_type,
            root='/home/lr/dataset/TAP-Vid/tapvid_davis/data_split',
            list_path=None,
            tapvid_subset_name='davis',
            query_mode='first',
            pipeline=test_pipeline,
            test_mode=True,
            # input_size=(384, 512)
            ),
)

# optimizer
optimizers = dict(
    type='Adam', lr=0.001, betas=(0.9, 0.999)
    )
optimizer_config = {}

# learning policy
# total_iters = 200000
runner_type='epoch'
max_epoch=30
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.001,
    by_epoch=False,
    warmup_iters=10,
    warmup_ratio=0.1,
    warmup_by_epoch=True
    )

work_dir = f'/home/lr/expdir/VCL/group_stsl/{exp_name}'


checkpoint_config = dict(interval=max_epoch//2, save_optimizer=True, by_epoch=True)
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
dist_params = dict(backend='nccl')
log_level = 'INFO'


evaluation = dict(output_dir=f'{work_dir}/eval_output_val', interval=max_epoch//2, by_epoch=True
                  )

eval_config= dict(
                  output_dir=f'{work_dir}/eval_output',
                  checkpoint_path=f'{work_dir}/epoch_{max_epoch}.pth'
                )

eval_arc = 'none'
load_from = None
resume_from = None
ddp_shuffle = True
workflow = [('train', 1)]
find_unused_parameters = True