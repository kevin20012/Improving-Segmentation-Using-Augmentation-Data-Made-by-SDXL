_base_ = ['../_base_/models/deeplabv3plus_r50-d8.py',
          '../_base_/datasets/wta_512_512x512_not_hard_relabeled_test.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3))
