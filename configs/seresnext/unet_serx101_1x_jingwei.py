# model settings
model = dict(
    type='UNet',
    pretrained='open-mmlab://se_resnext101_32x4d',
    backbone=dict(
        type='SEResNext',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    multi_scale=dict(
        type='ASPP',
        in_channel=[64, 256, 512, 1024, 2048],
        out_channel=[32, 64, 128, 256, 512],
        size=[256, 128, 64, 32, 16]),
    decoder=dict(
        type='UnetSCSE',
        in_channels=[32, 64, 128, 256, 512],
        num_filters=16,
        num_classes=4))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'JingweiDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix=data_root + 'jingwei_round1_train_20190619/image',
        split_file='mix_train.txt',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        ann_file='data/jingwei_round1_train_20190619/label'),
    val=dict(
        type=dataset_type,
        img_prefix=data_root + 'jingwei_round1_train_20190619/image',
        split_file='mix_val.txt',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        val_mode=True,
        ann_file='data/jingwei_round1_train_20190619/label'),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + 'jingwei_round1_test_a_20190619/image',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        test_mode=True,
        ann_file='data/jingwei_round1_train_20190619/label'))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 15  # actual epoch = 4 * 3 = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/unet_x101_fpn_1x_jingwei_all'
load_from = None
resume_from = None
workflow = [('train', 1)]
