#!/bin/bash
# 渐进式训练 HFP + DC + DNGeo（VisDrone）
# 三阶段顺序执行：stage1 -> stage2 -> stage3

set -e

CONFIG_DIR="configs/SSG"
GPU_NUM=${1:-1}

echo "========== Stage 1: Train DC+DNGeo (5 epochs) =========="
python tools/train.py ${CONFIG_DIR}/gfl_res18_hfp_order_stage1_visdrone.py --gpu-ids 0

echo "========== Stage 2: Freeze DC+DNGeo, Train HFP only (5 epochs) =========="
python tools/train.py ${CONFIG_DIR}/gfl_res18_hfp_order_dc_choice_dngeo_reorder_choice_visdrone.py --gpu-ids 0

echo "========== Stage 3: Unfreeze all, Fine-tune with small lr (5 epochs) =========="
python tools/train.py ${CONFIG_DIR}/gfl_res18_hfp_order_stage3_visdrone.py --gpu-ids 0

echo "========== Progressive training completed =========="
echo "Final model: work_dirs/gfl_res18_hfp_order_stage3_visdrone/"
