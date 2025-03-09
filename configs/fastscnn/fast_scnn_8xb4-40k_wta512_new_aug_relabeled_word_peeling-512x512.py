_base_ = [
    '../_base_/models/fast_scnn.py', '../_base_/datasets/new_aug_relabeled_word_peeling.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_lr160k.py'
]
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor,
             decode_head=dict(num_classes=3, 
                              loss_decode=dict(
                type='FocalLoss', use_sigmoid=False, gamma=2.0,
        alpha=[0.1, 0.3, 0.6], loss_weight=0.4)
        ),
             auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=32,
            num_convs=1,
            num_classes=3,
            in_index=-2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='FocalLoss', use_sigmoid=False, gamma=2.0,
        alpha=[0.1, 0.3, 0.6], loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=32,
            num_convs=1,
            num_classes=3,
            in_index=-3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='FocalLoss', use_sigmoid=False, gamma=2.0,
        alpha=[0.1, 0.3, 0.6], loss_weight=0.4)),
    ],)
# Re-config the data sampler.
# train_dataloader = dict(batch_size=4, num_workers=4)
# val_dataloader = dict(batch_size=1, num_workers=4)
# test_dataloader = val_dataloader

# Re-config the optimizer.
# optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=4e-5)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
