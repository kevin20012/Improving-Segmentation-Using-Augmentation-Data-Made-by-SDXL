_base_ = ['../_base_/models/deeplabv3plus_r50-d8.py',
          '../_base_/datasets/final_relabeled_ori_aug_re_val_re_test.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
        num_classes=3),
    auxiliary_head=dict(in_channels=256, num_classes=3))