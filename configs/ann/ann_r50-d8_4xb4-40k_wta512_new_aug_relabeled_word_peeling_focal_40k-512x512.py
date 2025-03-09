_base_ = [
    '../_base_/models/ann_r50-d8.py',
    '../_base_/datasets/new_aug_relabeled_word_peeling.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k_lr160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=3,loss_decode=dict(
                type='FocalLoss', use_sigmoid=False, gamma=2.0,
        alpha=[0.1, 0.3, 0.6], loss_weight=0.4)),
    auxiliary_head=dict(num_classes=3,loss_decode=dict(
                type='FocalLoss', use_sigmoid=False, gamma=2.0,
        alpha=[0.1, 0.3, 0.6], loss_weight=0.4)))
