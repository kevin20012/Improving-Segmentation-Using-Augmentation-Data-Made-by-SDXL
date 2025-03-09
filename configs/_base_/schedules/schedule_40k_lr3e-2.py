# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0005)
# optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
# param_scheduler = [
#     dict(
#         type='PolyLR',
#         eta_min=1e-4,
#         power=0.9,
#         begin=0,
#         end=40000,
#         by_epoch=False)
# ]
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]
# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# # optimizer
# optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.1)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# # learning policy
# param_scheduler = [
#     # Use a linear warm-up
#     dict(type='LinearLR',
#          start_factor=0.001,
#          by_epoch=False,
#          begin=0,
#          end=10000),
#     # Use a cosine learning
#     dict(type='CosineAnnealingLR',
#          T_max=110000,
#          by_epoch=False,
#          begin=10000,
#          end=120000)
# ]
# # training schedule for 40k
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=120000, val_interval=4000)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=20000),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     visualization=dict(type='SegVisualizationHook'))