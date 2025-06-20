#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

entity="Lionel_Messi"
method="npo+rt+wd+ot"
model_type="llama-3.1-8b-instruct"
model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct"

python run.py \
    --model_type $model_type \
    --model_name_or_path $model_name_or_path \
    --target_entity $entity \
    --method $method \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.0 \
    --seed 42 \
    --wandb_mode online \
    --attn_implementation sdpa \
    --do_train \
    --bf16 \
    --torch_compile \
    --use_gradient_checkpointing \
    --alternate_updates
