_base_ = [
    '../_base_/models/fastfcn_r50-d32_jpu_psp.py',
    '../_base_/datasets/final_relabeled_ori_aug_only_junhyung_re_val_re_test_oversample3x.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k_lr160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3))
