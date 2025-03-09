import argparse
import os
import os.path as osp

import mmcv
import numpy as np
from mmseg.evaluation.metrics.iou_metric import IoUMetric
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.apis import inference_model
from tqdm import tqdm
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model with tile-based inference')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
             'If specified, it will be automatically saved '
             'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value '
             'to be overwritten is a list, it should be like key="[a,b]" '
             'or key=a,b. It also allows nested list/tuple values, e.g. '
             'key="[(a,b),(c,d)]". Note that the quotation marks are necessary '
             'and that no white space is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    parser.add_argument(
        '--tile-split',
        type=int,
        default=1,
        help='Number of splits both horizontally and vertically (n). '
             'Default=2 -> n×n=2×2=4개의 타일로 쪼갬.')
    # PyTorch 2.0에서는 '--local_rank'가 자동으로 넘어올 수 있으므로 아래 같이 처리
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    """--show 혹은 --show-dir 옵션을 활성화하면 VisualizationHook이 동작하도록 설정."""
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on drawing
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks. '
            'refer to usage "visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def tile_inference(model, image: np.ndarray, n: int = 2):
    """이미지를 n×n 조각으로 나눈 뒤, 각각 추론(inference)한 결과를 합쳐 최종 마스크를 생성.

    Args:
        model: MMSegmentation 모델 (mmengine.runner.Runner로 build된 모델).
        image (np.ndarray): (H, W, 3) 형태의 원본 이미지.
        n (int): 가로, 세로 분할 개수.

    Returns:
        np.ndarray: (H, W) 형태의 최종 예측 마스크.
    """
    h, w, _ = image.shape
    tile_h = h // n
    tile_w = w // n

    # 최종 결과를 담을 배열 (세그멘테이션 맵)
    final_mask = np.zeros((h, w), dtype=np.uint8)

    # 각 타일(tile)을 순회하며 추론
    for i in range(n):
        for j in range(n):
            h_start = i * tile_h
            w_start = j * tile_w

            # 나누어 떨어지지 않는 경우를 위해 마지막 타일은 이미지 끝까지 사용
            if i == n - 1:
                h_end = h
            else:
                h_end = h_start + tile_h

            if j == n - 1:
                w_end = w
            else:
                w_end = w_start + tile_w

            # 슬라이스된 타일 영역
            tile_img = image[h_start:h_end, w_start:w_end]

            # MMSeg 모델 추론 (결과는 리스트 형태이므로 [0] 사용)
            seg_result = inference_model(model, tile_img)
            tile_mask = seg_result.pred_sem_seg.data.cpu().numpy().astype(np.uint8)
            print('tile_mask: ',np.unique(tile_mask))

            # 최종 마스크에 해당 영역을 붙여넣음
            final_mask[h_start:h_end, w_start:w_end] = tile_mask

    return final_mask


def main():
    args = parse_args()

    # 1. Config 로드 및 수정
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir 설정 우선순위: CLI > config 내 설정 > 자동 지정
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # checkpoint 로드
    cfg.load_from = args.checkpoint

    # 시각화 옵션(--show, --show-dir) 설정
    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # TTA 옵션
    if args.tta:
        # tta_pipeline이 존재한다고 가정
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # 예측 결과를 저장할 디렉토리 설정
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # 2. Runner 생성
    runner = Runner.from_cfg(cfg)

    # 3. 모델 및 데이터로더 준비
    model = runner.model
    model.eval()
    model.cfg = cfg
    test_dataloader = runner.build_dataloader(cfg.test_dataloader)
    dataset = test_dataloader.dataset

    # 결과 저장 폴더 지정 (없다면 만들기)
    out_dir = args.out if args.out is not None else osp.join(cfg.work_dir, 'tile_inference_results')
    os.makedirs(out_dir, exist_ok=True)

    # ================================
    # 평가를 위한 변수 (컨퓨전 매트릭스)
    # ================================
    num_classes = len(dataset.METAINFO['classes'])  # 클래스 개수 (배경 포함)
    ignore_index = dataset.ignore_index if hasattr(dataset, 'ignore_index') else 255
    # 아래 4개 누적 변수 (class별로 교집합, 합집합, pred/label 카운트)
    total_area_intersect = np.zeros((num_classes,), dtype=np.float64)
    total_area_union = np.zeros((num_classes,), dtype=np.float64)
    total_area_pred = np.zeros((num_classes,), dtype=np.float64)
    total_area_label = np.zeros((num_classes,), dtype=np.float64)

    # 4. 실제 타일 단위 추론 + 결과 병합 + 저장
    print(f'\n[INFO] Tile-based inference (n={args.tile_split}×{args.tile_split})를 시작합니다.')

    prog_bar=tqdm(total=len(dataset))
    for idx in range(len(dataset)):
        data_info = dataset.get_data_info(idx)
        print(data_info)
        # MMSeg 1.x 기준으로 `img_path`나 `filename`으로 이미지 경로가 들어있음
        img_path = data_info.get('img_path', data_info.get('filename', None))
        if img_path is None:
            raise ValueError('Dataset 정보에서 img_path(혹은 filename)를 찾을 수 없습니다.')
        # 원본 이미지 읽기
        image = mmcv.imread(img_path)

        # 타일 단위 추론
        pred_mask = tile_inference(model, image, n=args.tile_split)
        print('pred_mask: ',np.unique(pred_mask))

        # 결과 시각화 혹은 저장
        # 간단히 png 마스크로 저장하는 예시
        save_path = osp.join(out_dir, osp.basename(img_path).rsplit('.', 1)[0] + '_pred.png')
        mmcv.imwrite(pred_mask, save_path)

        # ======================================
        # (추가) GT 마스크 로드 후 mIoU/acc 계산
        # ======================================
        # dataset에 따라 GT 마스크 경로가 다를 수 있음

        seg_map = data_info.get('seg_map_path', None)
        if seg_map is not None:
            gt_path = seg_map
            # GT 마스크(흑백) 읽기 (H, W) - uint8
            gt_mask = mmcv.imread(gt_path, flag='unchanged')
            print('gt_mask: ',np.unique(gt_mask))
            # ignore_index를 고려한 intersect_and_union 계산
            pred_mask_tensor = torch.as_tensor(pred_mask, dtype=torch.int64)
            gt_mask_tensor = torch.as_tensor(gt_mask, dtype=torch.int64)
            area_intersect, area_union, area_pred, area_label = IoUMetric.intersect_and_union(
                pred_mask_tensor, gt_mask_tensor, num_classes, ignore_index=ignore_index) #ignore_index 무시
            
            area_intersect = area_intersect.cpu().numpy()
            area_union = area_union.cpu().numpy()
            area_pred = area_pred.cpu().numpy()
            area_label = area_label.cpu().numpy()
            print(area_intersect, area_union, area_pred, area_label)

            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_pred += area_pred
            total_area_label += area_label

        prog_bar.update()
    prog_bar.close() 

    # ======================================
    # 전체 이미지에 대한 mIoU, mAcc, aAcc 계산
    # ======================================
    eps = 1e-6

    # Class별 IoU, Acc
    iou_per_class = (total_area_intersect + eps) / (total_area_union + eps)
    acc_per_class = (total_area_intersect + eps) / (total_area_label + eps)

    # mIoU, mAcc, aAcc
    mIoU = iou_per_class.mean()
    mAcc = acc_per_class.mean()
    # 전체 픽셀 정확도(aAcc)
    aAcc = (total_area_intersect.sum() + eps) / (total_area_label.sum() + eps)

    print('\n===== Evaluation Results =====')
    # print(f'mIoU: {mIoU * 100.0:.2f}%')
    # print(f'mAcc: {mAcc * 100.0:.2f}%')
    # print(f'aAcc: {aAcc * 100.0:.2f}%')

    # 필요하다면 class별 IoU도 출력 가능
    for i, class_name in enumerate(dataset.METAINFO['classes']):
        print(f'{class_name}: IoU={iou_per_class[i]*100.0:.2f}% Acc={acc_per_class[i]*100.0:.2f}%')

    print(f'\n[INFO] 모든 이미지에 대해 타일 단위 추론 + 평가 완료. 결과는 "{out_dir}"에 저장되었습니다.')


if __name__ == '__main__':
    main()