_base_ = [
    '../_base_/models/fastfcn_r50-d32_jpu_psp.py',
    '../_base_/datasets/wta_512_aug_lora.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3))
