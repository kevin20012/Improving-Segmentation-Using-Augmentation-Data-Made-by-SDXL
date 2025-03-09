#! /bin/bash
read -p "Enter Project Name : " PROJECT_NAME #work_dirs 내의 저장되는 디렉토리의 이름입니다.
if [ "$PROJECT_NAME" == "" ]; then
    echo "Empty name is not acceptable."
    exit 1
fi
export CUDA_VISIBLE_DEVICES=0
read -p "Enter Gpu Count : " GPU_COUNT #GPU개수 입력
if (($GPU_COUNT <= 0)); then
    echo "0 or Negative number of Gpu is not acceptable."
    exit 1
fi

TEST_CANDI=( #시도해볼 모델 - config py파일의 상위 디렉토리의 이름도 함께 써주어야합니다.
    # Training for Final Result
    # gcnet/gcnet_r50-d8_4xb4-40k_wta512_ori_focal-512x512
    # gcnet/gcnet_r50-d8_4xb4-40k_wta512_new_aug_relabeled_word_rust_peeling_focal-512x512
    # apcnet/apcnet_r50-d8_4xb4-40k_full_relabeled_wta512_ori_focal-512x512
    # apcnet/apcnet_r50-d8_4xb4-40k_full_relabeled_wta512_new_aug_relabeled_word_peeling_focal-512x512
    # pspnet/pspnet_r50-d8_4xb4-40k_wta512_ori-512x512
    # pspnet/pspnet_r50-d8_4xb4-40k_wta512_new_aug_relabeled_word_peeling_focal-512x512
    # fastscnn/fast_scnn_8xb4-40k_wta512_ori-512x512_focal_loss_gamma_2
    # fastscnn/fast_scnn_8xb4-40k_wta512_new_aug_relabeled_word_peeling-512x512
    # mobilenet_v3/mobilenet-v3-d8_lraspp_4xb4-40k_wta512_ori_focal-512x512
    # mobilenet_v3/mobilenet-v3-d8_lraspp_4xb4-40k_wta512_new_aug_relabeled_word_peeling_focal-512x512
    # upernet/upernet_r50_4xb4-40k_wta512_ori_focal-512x512
    # upernet/upernet_r50_4xb4-40k_wta512_new_aug_relabeled_word_peeling_focal-512x512
    # ann/ann_r50-d8_4xb4-40k_wta512_40k_ori_focal-512x512
    # ann/ann_r50-d8_4xb4-40k_wta512_new_aug_relabeled_word_peeling_focal_40k-512x512
    # bisenetv2/bisenetv2_fcn_4xb4-40k_full_relabeled_wta512_ori_focal-512x512
    # bisenetv2/bisenetv2_fcn_4xb4-40k_full_relabeled_wta512_new_aug_relabeled_word_peeling_focal-512x512
    # beit/beit-base_upernet_8xb2-40k_wta512_ori_focal-512x512
    # beit/beit-base_upernet_8xb2-40k_wta512_new_aug_relabeled_word_peeling_focal-512x512
    # vit/vit_deit-b16_mln_upernet_8xb2-40k_wta512_ori_focal-512x512
    # vit/vit_deit-b16_mln_upernet_8xb2-40k_wta512_new_aug_relabeled_word_peeling_focal-512x512
    # ccnet/ccnet_r50-d8_4xb4-40k_full_relabeled_wta512_ori_focal-512x512
    # ccnet/ccnet_r50-d8_4xb4-40k_wta512_new_aug_relabeled_word_peeling_focal-512x512
    # danet/danet_r50-d8_4xb4-40k_wta512_ori_focal-512x512
    # danet/danet_r50-d8_4xb4-40k_wta512_new_aug_relabeled_word_peeling_focal-512x512
)
MODEL_CONFIG_PATH=./configs
WORK_DIR=./work_dirs
FIND_BEST_DIR=./bash
METRIC=mIoU

total_iter=40000
ckpt_save_step=4000

mkdir -p $WORK_DIR/$PROJECT_NAME/exp
touch $WORK_DIR/$PROJECT_NAME/exp/exp_log.txt
echo "$(date +%Y-%m-%d-%H:%M:%S) <<  ${PROJECT_NAME}  >>" > $WORK_DIR/$PROJECT_NAME/exp/exp_log.txt
for model in ${TEST_CANDI[@]}
do
    echo "$(date +%Y-%m-%d-%H:%M:%S) =================$model=================" >> $WORK_DIR/$PROJECT_NAME/exp/exp_log.txt
    echo "\n$(date +%Y-%m-%d-%H:%M:%S) 🚀🚀🚀🚀🚀$model Start!!!🚀🚀🚀🚀🚀"

    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Experiment Train Start=================" >> $WORK_DIR/$PROJECT_NAME/exp/exp_log.txt
    # 학습 시작
    python tools/train.py \
    $MODEL_CONFIG_PATH/$model.py --work-dir $WORK_DIR/$PROJECT_NAME/exp/$model/train
    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Experiment Train End=================" >> $WORK_DIR/$PROJECT_NAME/exp/exp_log.txt


    #가중치 파일 이동
    mkdir -p $WORK_DIR/$PROJECT_NAME/exp/$model/train/ckpt
    mv $WORK_DIR/$PROJECT_NAME/exp/$model/train/iter_* $WORK_DIR/$PROJECT_NAME/exp/$model/train/last_checkpoint $WORK_DIR/$PROJECT_NAME/exp/$model/train/ckpt

    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Experiment Test Start=================" >> $WORK_DIR/$PROJECT_NAME/exp/exp_log.txt
    # 추론 시작
    vis_data_path=$WORK_DIR/$PROJECT_NAME/exp/$model/train/$(ls $WORK_DIR/$PROJECT_NAME/exp/$model/train | grep [0-9][0-9][0-9]_[0-9][0-9][0-9])/vis_data
    json_log_path=$vis_data_path/$(ls $vis_data_path | grep [0-9]*_[0-9]*.json)
    echo iter_`python $FIND_BEST_DIR/find_best.py --json_log_path $json_log_path`.pth > $WORK_DIR/$PROJECT_NAME/exp/$model/train/ckpt/best_checkpoint
    mkdir -p $WORK_DIR/$PROJECT_NAME/exp/$model/test
    python tools/test.py \
    $MODEL_CONFIG_PATH/$model.py $WORK_DIR/$PROJECT_NAME/exp/$model/train/ckpt/$(< $WORK_DIR/$PROJECT_NAME/exp/$model/train/ckpt/best_checkpoint) > $WORK_DIR/$PROJECT_NAME/exp/$model/test/test_log.txt
done
