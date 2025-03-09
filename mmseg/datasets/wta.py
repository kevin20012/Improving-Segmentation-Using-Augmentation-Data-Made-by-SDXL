# import os.path as osp
# # from mmseg.datasets.custom import CustomDataset
# # from mmseg.datasets import CustomDataset
# from .basesegdataset import BaseSegDataset
# from mmseg.registry import DATASETS

# @DATASETS.register_module()
# class WTADataset(BaseSegDataset):
#     CLASSES = ('normal', 'attached', 'broken')
#     PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]

#     def __init__(self, split, **kwargs):
#         super().__init__(img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
#         assert osp.exists(self.img_dir) and self.split is not None


from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class WTADataset(BaseSegDataset):
    METAINFO = dict(
        classes=('defect', 'attached', 'broken'),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)