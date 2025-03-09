_base_ = [
    '../_base_/models/lraspp_m-v3-d8.py',
    '../_base_/datasets/new_aug_relabeled_word_peeling_ratio_0dot8.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k_lr160k.py'
]
norm_cfg = dict(type='SyncBN', eps=0.001, requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://contrib/mobilenet_v3_large',
    decode_head=dict(
        type='LRASPPHead',
        in_channels=(16, 24, 960),
        in_index=(0, 1, 2),
        channels=128,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        loss_decode=dict(
                type='FocalLoss', use_sigmoid=False, gamma=2.0,
        alpha=[0.1, 0.3, 0.6], loss_weight=0.4)))

# # Re-config the data sampler.
# train_dataloader = dict(batch_size=4, num_workers=4)
# val_dataloader = dict(batch_size=1, num_workers=4)
# test_dataloader = val_dataloader

# runner = dict(type='IterBasedRunner', max_iters=320000)
