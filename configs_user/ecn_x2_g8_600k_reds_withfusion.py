exp_name = 'ecn_x2_test_600k_reds_withfusion'

# model settings
model = dict(
    type='ECN',
    generator=dict(
        type='ECNNet',
        with_fusion=True
        ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)
data_root = 'data/REDS/train/encode/32x32/recon_png'

# dataset settings
train_dataset_type = 'SRREDSDataset'
val_dataset_type = 'SRREDSDataset'
train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], frames_per_clip=99),
    dict(type='TemporalReverse', keys=['lq_path', 'gt_path'], reverse_ratio=0),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged',
        convert_to='Y'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        flag='unchanged',
        convert_to='Y'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    # dict(
    #     type='Normalize',
    #     keys=['lq', 'gt'],
    #     mean=[0, 0, 0],
    #     std=[1, 1, 1],
    #     to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='FramesToTensor', keys=['lq', 'gt'])
]

test_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection_circle'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged',
        convert_to='Y'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        flag='unchanged',
        convert_to='Y'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    # dict(
    #     type='Normalize',
    #     keys=['lq', 'gt'],
    #     mean=[0, 0, 0],
    #     std=[1, 1, 1],
    #     to_rgb=True),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key']),
    dict(type='FramesToTensor', keys=['lq', 'gt'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=20,
    train_dataloader=dict(samples_per_gpu=8, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder=f'{data_root}/bl',
            gt_folder=f'{data_root}/el',
            ann_file=f'{data_root}/meta_info_REDS_GT.txt',
            num_input_frames=5,
            pipeline=train_pipeline,
            scale=2,
            val_partition='REDS4',
            test_mode=False)),
    val=dict(
        type=val_dataset_type,
        lq_folder=f'{data_root}/bl',
        gt_folder=f'{data_root}/el',
        ann_file=f'{data_root}/meta_info_REDS_GT.txt',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=2,
        val_partition='REDS4',
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        lq_folder=f'{data_root}/bl',
        gt_folder=f'{data_root}/el',
        ann_file=f'{data_root}/meta_info_REDS_GT.txt',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=2,
        val_partition='REDS4',
        test_mode=True),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))
find_unused_parameters = False
# learning policy
total_iters = 600000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[50000, 100000, 150000, 150000, 150000],
    restart_weights=[1, 1, 1, 1, 1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=50000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr'))
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./train_dirs/{exp_name}'
load_from = None
# load_from = 'work_dirs/edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth'
resume_from = None
workflow = [('train', 1)]
