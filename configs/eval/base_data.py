img_norm_cfg_lab = dict(mean=[50, 0, 0], std=[50, 127, 127], to_bgr=False)

test_pipeline_davis = [
    dict(type='Resize', scale=(256, 256), keep_ratio=False, keys='video'),
    dict(type='RGB2LAB', keys='video', output_keys='video'),
    dict(type='Normalize', **img_norm_cfg_lab, keys='video'),
]

test_pipeline_kinetics = [
    dict(type='Resize', scale=(256, 256), keep_ratio=False, keys='video'),
    dict(type='RGB2LAB', keys='video', output_keys='video'),
    dict(type='Normalize', **img_norm_cfg_lab, keys='video'),
]

test_pipeline_jhmdb = [
    dict(type='Resize', scale=(320, 320), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='RGB2LAB'),
    dict(type='Normalize', **img_norm_cfg_lab),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'ref_seg_map'],
        meta_keys=('video_path', 'original_shape')),
    dict(type='ToTensor', keys=['imgs', 'ref_seg_map'])
]

test_pipeline_badja = [
    dict(type='Resize', scale=(-1, 320), keep_ratio=True),
    dict(type='Flip', flip_ratio=0),
    dict(type='RGB2LAB'),
    dict(type='Normalize', **img_norm_cfg_lab),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'ref_seg_map'],
        meta_keys=('video_path', 'original_shape')),
    dict(type='ToTensor', keys=['imgs', 'ref_seg_map'])
]

# demo_pipeline = None
data = dict(
    workers_per_gpu=2,
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    
    test_davis =  dict(
            type='TAPVidDataset',
            # root='/home/lr/dataset/TAP-Vid/tapvid_davis/data_split',
            root='/home/sist/lirui/dataset/TAP-Vid/tapvid_davis/data_split',
            list_path=None,
            tapvid_subset_name='davis',
            query_mode='first',
            pipeline=test_pipeline_davis,
            test_mode=True,
            ),

    test_kinetics =  dict(
            type='TAPVidDataset',
            root='/home/sist/lirui/dataset/TAP-Vid/tapvid_kinetics/all_split',
            list_path=None,
            tapvid_subset_name='kinetics',
            query_mode='first',
            pipeline=test_pipeline_kinetics,
            test_mode=True
            ),

    test_jhmdb =  dict(
            type='jhmdb_dataset_rgb',
            root='/data/',
            list_path='/data/jhmdb',
            split='val',
            pipeline=test_pipeline_jhmdb,
            test_mode=True
            ),
    
    test_badja =  dict(
            type='vip_dataset_rgb',
            root='/gdata1/lirui/dataset/DAVIS_Full_Res',
            list_path='/gdata1/lirui/dataset/DAVIS_Full_Res',
            length=-1,
            scale=2,
            size=(320,512),
            test_mode=True,
            sigma=3,
            pipeline=test_pipeline_badja,
            )
)