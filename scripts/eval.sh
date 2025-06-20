#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

entity="Lionel_Messi"
method="npo+rt+wd+ot"
model_type="llama-3.1-8b-instruct"
model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct"
ckpt_path=".checkpoints/$model_type/$entity/$method/BS32_LR1e-05_W0.0_S42"

python run.py \
    --model_type $model_type \
    --model_name_or_path $model_name_or_path \
    --target_entity $entity \
    --method $method \
    --ckpt_path $ckpt_path \
    --per_device_eval_batch_size 32 \
    --attn_implementation sdpa \
    --do_test \
    --bf16
