import os
from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
import safetensors.torch
from transformers import PreTrainedModel
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME, is_peft_available
from transformers.trainer import TRAINING_ARGS_NAME
from peft import PeftModel
from trl import SFTTrainer

from model import sliced_wasserstein_distance


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, ref_model, method, dpo_beta, reg_lambda, alternate_updates, *args, **kwargs):
        super(CustomSFTTrainer, self).__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.method = method
        self.dpo_beta = dpo_beta
        self.reg_lambda = reg_lambda
        self.alternate_updates = alternate_updates

    def compute_loss(self, model, inputs, return_outputs=False):
        # * forget_samples, (idk_samples,) retain_samples, world_samples = inputs
        outputs = model(**inputs[0])
        loss = outputs[0]

        ######################
        # Unlearning methods #
        ######################
        if "ga" in self.method:
            loss = -loss
        elif "npo" in self.method:
            with torch.no_grad():
                outputs_oracle = self.ref_model(**inputs[0])
                loss_oracle = outputs_oracle[0]
            neg_log_ratios = loss - loss_oracle
            loss = -F.logsigmoid(self.dpo_beta * neg_log_ratios).mean() * 2 / self.dpo_beta
        elif "dpo" in self.method:
            idk_outputs = model(**inputs[1])
            idk_loss = idk_outputs[0]
            with torch.no_grad():
                idk_outputs_oracle = self.ref_model(**inputs[1])
                forget_outputs_oracle = self.ref_model(**inputs[0])
                idk_loss_oracle = idk_outputs_oracle[0]
                forget_loss_oracle = forget_outputs_oracle[0]
                idk_loss_oracle = -1 * idk_loss_oracle
                forget_loss_oracle = -1 * forget_loss_oracle
            # -1 * NLL loss = log probabilities
            idk_loss = -1 * idk_loss
            forget_loss = -1 * loss
            pi_logratios = idk_loss - forget_loss
            ref_logratios = idk_loss_oracle - forget_loss_oracle
            loss = -F.logsigmoid(self.dpo_beta * (pi_logratios - ref_logratios)).mean() * 2 / self.dpo_beta

        if self.alternate_updates:
            self.accelerator.backward(loss)
            self.optimizer.step()
            model.zero_grad()
            loss = 0.0

        if "idk" in self.method and ("+ga" in self.method or "+npo" in self.method):
            idk_outputs = model(**inputs[1])
            idk_loss = idk_outputs[0]
            loss = loss + idk_loss

        #######################
        # Retain if necessary #
        #######################
        if "+rt+wd" in self.method:
            retain_outputs = model(**inputs[-2])
            retain_loss = retain_outputs[0]
            retain_outputs2 = model(**inputs[-1])
            retain_loss2 = retain_outputs2[0]
            loss = loss + retain_loss + retain_loss2
        elif "+rt" in self.method or "+wd" in self.method:
            retain_outputs = model(**inputs[-1])
            retain_loss = retain_outputs[0]
            loss = loss + retain_loss

        ##################
        # Regularization #
        ##################
        if "+ot" in self.method:
            reg = 0.0
            for ref_param, param in zip(self.ref_model.parameters(), model.parameters()):
                if len(param.shape) == 1:
                    reg += sliced_wasserstein_distance(ref_param.unsqueeze(0), param.unsqueeze(0)).to(loss.device)
                else:
                    reg += sliced_wasserstein_distance(ref_param, param).to(loss.device)
            loss = loss + self.reg_lambda * reg

        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # * Cast to bfloat16 for fast saving
        model = deepcopy(self.model).bfloat16()

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model, supported_classes):
            if state_dict is None:
                state_dict = model.state_dict()

            if isinstance(self.accelerator.unwrap_model(model), supported_classes):
                self.accelerator.unwrap_model(model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                # logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Remove bf16 model to save memory
        del model
        torch.cuda.empty_cache()
