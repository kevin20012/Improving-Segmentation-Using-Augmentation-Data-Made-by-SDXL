_base_ = ['../_base_/models/upernet_vit-b16_ln_mln.py',
          '../_base_/datasets/final_relabeled_ori_aug_only_junhyung_re_val_re_test.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k_lr160k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/deit_base_patch16_224-b5f2ef4d.pth',
    backbone=dict(drop_path_rate=0.1),
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3)
)
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
