### Config(配置文件)

------

本文档详细介绍配置文件中常用的字段含义

config/文件夹下的config文件包含互相引用，对各个字段的展示不是很友好，可以使用如下脚本展示所有字段

```python
# python tools/misc/print_config.py config_filepath
# eg:
python tools/misc/print_config.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
```



1. 模型部分:model TODO

   ```python
   # 详细参见mmdetection构建流程
   # https://zhuanlan.zhihu.com/p/337375549
   model = dict(
       # 模型类型
       type='FasterRCNN',
       # backbone类型与参数
       backbone=dict(
           type='ResNet',
           depth=50,
           num_stages=4,
           out_indices=(0, 1, 2, 3),
           frozen_stages=1,
           norm_cfg=dict(type='BN', requires_grad=True),
           norm_eval=True,
           style='pytorch',
           init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),    
       neck=dict(
           type='FPN',
           in_channels=[256, 512, 1024, 2048],
           out_channels=256,
           num_outs=5),
       rpn_head=dict(
           type='RPNHead',
           in_channels=256,
           feat_channels=256,
           anchor_generator=dict(
               type='AnchorGenerator',
               scales=[8],
               ratios=[0.5, 1.0, 2.0],
               strides=[4, 8, 16, 32, 64]),
           bbox_coder=dict(
               type='DeltaXYWHBBoxCoder',
               target_means=[0.0, 0.0, 0.0, 0.0],
               target_stds=[1.0, 1.0, 1.0, 1.0]),
           loss_cls=dict(
               type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
           loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
       roi_head=dict(
           type='StandardRoIHead',
           bbox_roi_extractor=dict(
               type='SingleRoIExtractor',
               roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
               out_channels=256,
               featmap_strides=[4, 8, 16, 32]),
           bbox_head=dict(
               type='Shared2FCBBoxHead',
               in_channels=256,
               fc_out_channels=1024,
               roi_feat_size=7,
               num_classes=80,
               bbox_coder=dict(
                   type='DeltaXYWHBBoxCoder',
                   target_means=[0.0, 0.0, 0.0, 0.0],
                   target_stds=[0.1, 0.1, 0.2, 0.2]),
               reg_class_agnostic=False,
               loss_cls=dict(
                   type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
               loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
       train_cfg=dict(
           rpn=dict(
               assigner=dict(
                   type='MaxIoUAssigner',
                   pos_iou_thr=0.7,
                   neg_iou_thr=0.3,
                   min_pos_iou=0.3,
                   match_low_quality=True,
                   ignore_iof_thr=-1),
               sampler=dict(
                   type='RandomSampler',
                   num=256,
                   pos_fraction=0.5,
                   neg_pos_ub=-1,
                   add_gt_as_proposals=False),
               allowed_border=-1,
               pos_weight=-1,
               debug=False),
           rpn_proposal=dict(
               nms_pre=2000,
               max_per_img=1000,
               nms=dict(type='nms', iou_threshold=0.7),
               min_bbox_size=0),
           rcnn=dict(
               assigner=dict(
                   type='MaxIoUAssigner',
                   pos_iou_thr=0.5,
                   neg_iou_thr=0.5,
                   min_pos_iou=0.5,
                   match_low_quality=False,
                   ignore_iof_thr=-1),
               sampler=dict(
                   type='RandomSampler',
                   num=512,
                   pos_fraction=0.25,
                   neg_pos_ub=-1,
                   add_gt_as_proposals=True),
               pos_weight=-1,
               debug=False)),
       test_cfg=dict(
           rpn=dict(
               nms_pre=1000,
               max_per_img=1000,
               nms=dict(type='nms', iou_threshold=0.7),
               min_bbox_size=0),
           rcnn=dict(
               score_thr=0.05,
               nms=dict(type='nms', iou_threshold=0.5),
               max_per_img=100)))
   ```
   

   
2. 数据部分:data

   ```python
   dataset_type = 'CocoDataset' # 数据集类型 默认coco 在./mmdet/datasets下添加自定义
   data_root = 'data/coco/' # 数据根路径，存放原始数据和标注文件
   img_norm_cfg = dict(
       mean=[_, _, _], std=[_, _, _], to_rgb=True) ### 数据正则
   
   ### 训练数据处理流
   train_pipeline = [
       dict(type='LoadImageFromFile'),
       dict(type='LoadAnnotations', with_bbox=True),
       dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
       # 多尺度
       dict(
           type='Resize',
           img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                      (1333, 768), (1333, 800)],
           multiscale_mode='value',
           keep_ratio=True),
       dict(type='RandomFlip', flip_ratio=0.5),
       dict(
           type='Normalize',
           **img_norm_cfg),
       dict(type='Pad', size_divisor=32),
       dict(type='DefaultFormatBundle'),
       dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
   ]
   ### 测试验证数据处理流
   test_pipeline = [
       dict(type='LoadImageFromFile'),
       dict(
           type='MultiScaleFlipAug',
           img_scale=(1333, 800),
           # img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
           #          (1333, 768), (1333, 800)],
           flip=False,
           transforms=[
               dict(type='Resize', keep_ratio=True),
               dict(type='RandomFlip'),
               dict(type='Normalize',
                   **img_norm_cfg),
               dict(type='Pad', size_divisor=32),
               dict(type='ImageToTensor', keys=['img']),
               dict(type='Collect', keys=['img'])
           ])
   ]
   
   # train 数据类型与数据路径（附加数据信息）
   data = dict(
       samples_per_gpu=2, #batch size of each GPU.
       workers_per_gpu=2, #How many subprocesses to use for data loading for each GPU
       train=dict(
           type='CocoDataset',# 数据类型同dataset_type 
           ann_file='data/coco/annotations/instances_train2017.json', #标注文件路径
           img_prefix='data/coco/train2017/',#标注文件的图片前缀
           pipeline=train_pipeline) #数据处理流,    
       val=dict(
           type=_,
           ann_file=_,
           img_prefix=_,
       	pipeline=test_pipeline),
       test=dict(
           type=_,
           ann_file=_,
           img_prefix=_,
       	pipeline=test_pipeline)
       )
   
   # 评估周期与评估指标
   evaluation = dict(interval=1, metric='bbox')
   #evaluation = dict(metric=['bbox', 'segm'])
   ```

   

3. 训练策略: schedule

   ```python
   #优化器类型, 初始lr,momentum,weight_decay
   optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
   optimizer_config = dict(grad_clip=None)
   # 学习率schedule
   # step表示lr drop的epoch
   lr_config = dict(
       policy='step',
       warmup='linear',
       warmup_iters=500,
       warmup_ratio=0.001,
       step=[8,11])
   # 训练总周期
   runner = dict(type='EpochBasedRunner', max_epochs=12)
   # 保存checkpoints间隔
   checkpoint_config = dict(interval=1)
   # 打印log间隔
   log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
   custom_hooks = [dict(type='NumClassCheckHook')]
   dist_params = dict(backend='nccl')
   log_level = 'INFO'
   # 加载参数
   load_from = None
   # 重新加载(包含epoch等信息，会覆盖load_from)
   resume_from = None
   # 工作流 train val test
   workflow = [('train', 1),]
   # 保存目录
   work_dir = '_'
   ```

   

