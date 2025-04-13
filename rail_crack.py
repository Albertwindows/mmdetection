# 文件名: configs/cracks/rtmdet_ins_s_1024_crack.py
# ./tools/dist_train.sh \
#     mymodel.py \
#     8 \
#     --cfg-options \
#         train_dataloader.batch_size=16 \
#         val_dataloader.batch_size=8 \
#         train_cfg.max_epochs=300 \
#     --amp \
#     --work-dir work_dirs/crack_det \
#     --seed 42 \
#     --gpu-ids 0-7  # 使用全部8张3090
_base_ = 'configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py'
# 数据配置
data_root = '/mmdetection/data/rail/coco_format/'
metainfo = {
    'classes': ('crack', 'falling_blocks', 'mortar_layer_separation', 'joint_separation'),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
    ]
}
# 基础配置
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='coco/segm_mAP',
        rule='greater',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# 模型配置
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0, 0, 0],  # 灰度图像专用均值
        std=[255, 255, 255],     # 灰度图像专用标准差
        bgr_to_rgb=False,
        # pad_mask=False,
        pad_size_divisor=32),
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.5,
        use_depthwise=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth')),
    neck=dict(
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=2,
        use_depthwise=False),
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        num_classes=4,
        in_channels=128,
        stacked_convs=2,
        share_conv=True,
        # mask_size=56,  # 提高mask分辨率
        # pred_mask=True,
        loss_mask=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=False,
            loss_weight=3.0),  # 增加mask损失权重
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    test_cfg=dict(
        nms_pre=2000,        # 增加候选框数量
        min_bbox_size=1,     # 允许检测1像素目标
        score_thr=0.01,      # 降低初始阈值
        nms=dict(type='nms', iou_threshold=0.4),  # 宽松NMS
        max_per_img=1000,    # 最大检测数量
        mask_thr_binary=0.35))

# 数据配置
dataset_type = 'CocoDataset'
img_scale = (1024, 1024)

# 增强配置
albu_train_transforms = [
    dict(type='RandomRotate90', p=0.6),
    dict(
        type='ElasticTransform',
        alpha=60,
        sigma=8,
        alpha_affine=20,
        p=0.4),
    dict(type='GridDistortion', num_steps=5, distort_limit=0.3, p=0.3),
    dict(type='CLAHE', p=0.5),
    dict(type='RandomGamma', gamma_limit=(80, 120), p=0.5),
    dict(type='GaussianBlur', blur_limit=3, p=0.3),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=15,
        border_mode=0,
        value=114,
        p=0.5)
]
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
#
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(
#         type='Albu',
#         transforms=albu_train_transforms,
#         bbox_params=dict(
#             type='BboxParams',
#             format='pascal_voc',
#             label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
#             min_visibility=0.1,
#             filter_lost_elements=True),
#         keymap={
#             'img': 'image',
#             'gt_masks': 'masks',
#             'gt_bboxes': 'bboxes'
#         },
#         #update_pad_shape=False,
#         skip_img_without_anno=True),
#     dict(
#         type='RandomResize',
#         scale=img_scale,
#         ratio_range=(0.7, 1.3),  # 更大尺度变化
#         keep_ratio=True),
#     dict(type='RandomCrop', crop_size=(768, 768), allow_negative_crop=True),
#     dict(type='YOLOXHSVRandomAug'),
#     dict(type='Pad', size=img_scale, pad_val=dict(img=114)),  # 灰度填充值
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
# ]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=0)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 数据加载配置
train_dataloader = dict(
    batch_size=16,  # 8卡x16=128 total
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img_path='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img_path='val/'),
        test_mode=True,
        pipeline=test_pipeline))

# 评估配置
val_evaluator = dict(
    ann_file=data_root+'annotations/val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True)

test_evaluator = val_evaluator

# 训练策略
max_epochs = 300
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'neck': dict(lr_mult=0.3),
            'bbox_head': dict(lr_mult=1.0)},
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True))

# 学习率调度
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        begin=50,
        end=max_epochs,
        T_max=250,
        by_epoch=True,
        convert_to_iter_based=True)
]

# 高级配置
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=test_pipeline)
]

# 可视化配置
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    line_width=2)