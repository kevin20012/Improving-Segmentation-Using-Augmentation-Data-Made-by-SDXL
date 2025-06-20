_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/final_relabeled_ori_aug_only_junhyung_re_val_re_test_oversample4x.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_lr160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/beit_base_patch16_224_pt22k_ft22k.pth',
    backbone=dict(img_size=crop_size),
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3),
    test_cfg=dict(mode='whole')
    )

# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05),
#     constructor='LayerDecayOptimizerConstructor',
#     paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         power=1.0,
#         begin=1500,
#         end=160000,
#         eta_min=0.0,
#         by_epoch=False,
#     )
# ]

# By default, models are trained on 8 GPUs with 2 images per GPU
# train_dataloader = dict(batch_size=2)
# val_dataloader = dict(batch_size=1)
# test_dataloader = val_dataloader
