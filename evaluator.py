import csv

import torch
from transformers import TrainerCallback

from tqdm import tqdm


class EvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataloaders, rouge_evaluator):
        super().__init__()
        self.trainer = trainer
        self.eval_dataloaders = eval_dataloaders
        self.rouge_evaluator = rouge_evaluator

    def on_epoch_end(self, args, state, control, **kwargs):
        metrics, _ = evaluate(kwargs['model'], kwargs['tokenizer'], self.eval_dataloaders, self.rouge_evaluator)
        self.trainer.log(metrics)
        metrics = dict(sorted(metrics.items()))
        metrics["epoch"] = state.epoch
        with open(f"{args.output_dir}/traineval_results.csv", "a") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(metrics)

        # early stop after one epoch if method is vanilla GA, DPO, NPO, or IDK
        # or after two epochs otherwise
        if self.trainer.method == "ga" or self.trainer.method == "dpo" or self.trainer.method == "npo" or self.trainer.method == "idk":
            control.should_training_stop = True
        else:
            if state.epoch >= 1.9:  # doesn't fall perfectly at 2.0
                control.should_training_stop = True


def evaluate(model, tokenizer, dataloaders, rouge_evaluator, do_eval=False):
    metrics, _metrics, predictions = {}, {}, {}
    eval_dataloaders, gen_dataloaders = dataloaders
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    model.eval()
    # Compute loss
    for split, dataloader in eval_dataloaders.items():
        losses = []
        for batch in tqdm(dataloader, desc=f"Computing losses for {split}"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            if "perturbed" in split:
                bsz, num_pert, _ = batch["input_ids"].shape
                batch = {k: v.view(bsz*num_pert, -1) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            # Compute loss per token
            shifted_logits = outputs.logits[:, :-1, :].contiguous()
            shifted_labels = batch["labels"][:, 1:].contiguous()
            loss = loss_fct(shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
            num_gt_tokens = batch["labels"].ne(-100).sum(-1)
            if "perturbed" in split:
                loss = loss.view(bsz, num_pert)
                num_gt_tokens = num_gt_tokens.view(bsz, num_pert)
            loss_per_token = loss / num_gt_tokens
            losses.append(loss_per_token)

        _metrics[f"{split}/losses"] = torch.cat(losses)

    # Compute probability
    metrics["forget/prob"] = torch.exp(-1 * _metrics["forget_original/losses"]).mean().item()
    metrics["retain/prob"] = torch.exp(-1 * _metrics["retain_original/losses"]).mean().item()
    world_true_prob = torch.exp(-1 * _metrics["world_original/losses"])
    world_false_prob = torch.exp(-1 * _metrics["world_perturbed/losses"])
    world_all_prob = torch.cat([world_true_prob.unsqueeze(-1), world_false_prob], dim=-1).sum(-1)
    metrics["world/prob"] = (world_true_prob / world_all_prob).mean().item()

    # Compute truth ratio
    forget_truth_ratio = torch.exp(_metrics["forget_perturbed/losses"].mean(-1) - _metrics["forget_paraphrased/losses"])
    retain_truth_ratio = torch.exp(_metrics["retain_perturbed/losses"].mean(-1) - _metrics["retain_paraphrased/losses"])
    world_truth_ratio = torch.exp(_metrics["world_perturbed/losses"].mean(-1) - _metrics["world_paraphrased/losses"])

    metrics["forget/truth_ratio"] = torch.mean(torch.minimum(forget_truth_ratio, 1/forget_truth_ratio)).item()
    metrics["retain/truth_ratio"] = torch.mean(torch.maximum(torch.tensor(0.0), 1 - 1/retain_truth_ratio)).item()
    metrics["world/truth_ratio"] = torch.mean(torch.maximum(torch.tensor(0.0), 1 - 1/world_truth_ratio)).item()

    # Compute ROUGE-L recall
    if do_eval: # do not compute ROUGE-L recall during training for efficiency
        for split, dataloader in gen_dataloaders.items():
            preds, labels = [], []
            for batch in tqdm(dataloader, desc=f"Generating responses for {split}"):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                with torch.no_grad():
                    gen_outputs = model.generate(input_ids=batch["input_ids"],
                                                 attention_mask=batch["attention_mask"],
                                                 max_new_tokens=512 if split == "world" else 128,
                                                 use_cache=True,
                                                 do_sample=False,
                                                 num_beams=1,
                                                 temperature=0.0,
                                                 top_p=1.0,
                                                 pad_token_id=tokenizer.eos_token_id)

                decoded_outputs = tokenizer.batch_decode(gen_outputs[:, batch["input_ids"].shape[1]:],
                                                         skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
                decoded_labels = tokenizer.batch_decode(batch["labels"],
                                                        skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)
                preds += decoded_outputs
                labels += decoded_labels

            data = []
            for gen, gt in zip(preds, labels):
                data.append({'pred': gen, 'gt': gt})
            predictions[split] = data

            # Compute ROUGE-L recall
            rougeL_recall = 0
            for gen, gt in zip(preds, labels):
                rougeL_recall += rouge_evaluator.score(gt, gen)['rougeL'].recall
            rougeL_recall /= len(preds)
            metrics[f"{split}/rougeL_recall"] = rougeL_recall

    return metrics, predictions
