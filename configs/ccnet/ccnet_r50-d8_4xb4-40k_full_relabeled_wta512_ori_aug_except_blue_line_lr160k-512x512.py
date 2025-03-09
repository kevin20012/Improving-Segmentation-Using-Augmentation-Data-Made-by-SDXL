_base_ = [
    '../_base_/models/ccnet_r50-d8.py',
    '../_base_/datasets/final_relabeled_ori_aug_except_blue_line_re_val_re_test.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k_lr160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3))
