import os
import os.path as osp
import glob
import argparse
import csv
import json
from rouge_score import rouge_scorer

import torch
from transformers import set_seed
from trl import SFTConfig

from trainer import CustomSFTTrainer
from dataset import load_data, custom_train_collate_fn
from model import load_tokenizer, load_model, create_reference_model
from evaluator import EvalCallback, evaluate

import wandb


def main(args):
    # Load tokenizer
    tokenizer = load_tokenizer(args)

    # Load data
    train_dataset = load_data(args, tokenizer, split="train")
    eval_dataloaders = load_data(args, tokenizer, split="validation")
    test_dataloaders = load_data(args, tokenizer, split="test")

    # Load model
    model = load_model(args, args.ckpt_path)
    if args.do_train and ("dpo" in args.method or "npo" in args.method or "ot" in args.method):
        ref_model = create_reference_model(model)
    else:
        ref_model = None

    # Load evaluators
    rouge_evaluator = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    trainer = CustomSFTTrainer(
        args=SFTConfig(
            output_dir=args.output_dir,
            dataloader_num_workers=args.num_workers,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch",
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_steps,
            save_strategy="epoch",
            save_only_model=True,
            torch_compile=args.torch_compile,
            report_to="wandb" if args.wandb_mode == "online" else "none",
            # * SFT arguments
            max_seq_length=args.max_seq_len,
            dataset_text_field="text",
            packing=args.packing,
            dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
        ),
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=custom_train_collate_fn if not args.packing else None,
        # * Custom arguments
        ref_model=ref_model,
        method=args.method,
        dpo_beta=args.dpo_beta,
        reg_lambda=args.reg_lambda,
        alternate_updates=args.alternate_updates,
    )
    trainer.add_callback(EvalCallback(trainer, eval_dataloaders, rouge_evaluator))
    
    if args.do_train:
        trainer.train()

    if args.do_eval:
        metrics, preds = evaluate(model, tokenizer, eval_dataloaders, rouge_evaluator, do_eval=True)
        metrics = dict(sorted(metrics.items()))
        with open(f"{args.output_dir}/eval_results.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
        with open(f"{args.output_dir}/eval_preds.json", "w") as f:
            json.dump(preds, f, indent=4)

    if args.do_test:
        metrics, preds = evaluate(model, tokenizer, test_dataloaders, rouge_evaluator, do_eval=True)
        metrics = dict(sorted(metrics.items()))
        with open(f"{args.output_dir}/test_results.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
        with open(f"{args.output_dir}/test_preds.json", "w") as f:
            json.dump(preds, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Opt-Out Unlearning")
    # Model arguments
    parser.add_argument("--model_type", type=str, default="llama3.1-8b-instruct")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--method", type=str, default="original")
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--target_entity", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--packing", action="store_true")
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reg_lambda", type=float, default=0.1)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--alternate_updates", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    args = parser.parse_args()

    # Raise any exceptions before training
    if args.alternate_updates and ("+rt" not in args.method and "+wd" not in args.method):
        raise ValueError(f"Alternate updates are not supported without proper retain and world data.")
    if args.ckpt_path and args.do_train:
        raise ValueError("Cannot train with a checkpoint path.")

    # Set seed
    set_seed(args.seed, deterministic=args.deterministic)

    # Set number of threads for CPU computation in OPTOUT
    torch.set_num_threads(1)

    # Set distributed training if necessary
    world_size = torch.cuda.device_count()
    args.distributed = world_size != 1
    args.train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    args.eval_batch_size = args.per_device_eval_batch_size * world_size

    # Set data type
    if args.bf16:
        args.dtype = torch.bfloat16
    elif args.fp16:
        args.dtype = torch.float16
    else:
        args.dtype = torch.float32

    # Set output directory
    if args.ckpt_path:
        args.ckpt_path = sorted(glob.glob(osp.join(args.ckpt_path, "checkpoint-*")), key=lambda x: int(x.split("-")[-1]))[-1]
        args.output_dir = args.ckpt_path
    else:
        if args.method in ["original", "icu"]:
            if args.do_train:
                raise ValueError(f"{args.method} method is not supported for training.")
            args.group_name = osp.join(args.model_type, args.target_entity, args.method)
            args.run_name = ""
        else:
            args.group_name = osp.join(args.model_type, args.target_entity)
            args.run_name = f"{args.method}/BS{args.train_batch_size}_LR{args.learning_rate}_W{args.warmup_ratio}_S{args.seed}"

        args.output_dir = osp.join(".checkpoints", args.group_name, args.run_name)

    # Do not overwrite checkpoint files
    if args.do_train and glob.glob(osp.join(args.output_dir, "checkpoint-*")):
        raise FileExistsError(f"Output directory {args.output_dir} already exists.")

    # Set up wandb
    if args.wandb_mode == "online":
        wandb.init(
            project="Opt-Out",
            group=args.group_name,
            name=args.run_name,
            mode=args.wandb_mode,
        )

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
