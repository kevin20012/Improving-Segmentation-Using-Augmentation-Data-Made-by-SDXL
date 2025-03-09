_base_ = [
    '../_base_/models/danet_r50-d8.py', '../_base_/datasets/new_aug_relabeled_word_scratch.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_lr160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3))
