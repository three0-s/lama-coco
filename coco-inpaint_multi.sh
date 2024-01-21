#! /bin/bash
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc_per_node 4 composite_coco_lama_multi.py \
    --coco_root /workspace/coco2017_tr_val/val2017 \
    --coco_ann_root /workspace/coco2017_tr_val/annotations \
    --batch 256 \
    --outdir /workspace/lama-coco-val \
    --model_path /workspace/lama-coco/big-lama/models/best.ckpt \
    --config_path /workspace/lama-coco/configs/prediction/default.yaml
