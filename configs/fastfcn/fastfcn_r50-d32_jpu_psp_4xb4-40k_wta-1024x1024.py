_base_ = [
    '../_base_/models/fastfcn_r50-d32_jpu_psp.py',
    '../_base_/datasets/wta_512_1024x1024.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3))
