import os.path as osp
import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def load_data(args, tokenizer, split="train"):
    entity = ' '.join(args.target_entity.split("_"))
    icu = args.method == "icu"
    
    if split == "train":
        formatting_func = CustomFormatter(tokenizer, entity=entity, icu=icu).formatting_func
        train_dataset = CustomDataset(args, tokenizer, entity, formatting_func)
        return train_dataset

    if split == "validation" or split == "test":
        forget_data = load_dataset("6rightjade/ELUDe", "forget_qa", split="train")
        retain_data = load_dataset("6rightjade/ELUDe", "retain_qa", split=split)
        forget_data = forget_data.filter(lambda x: x["entity"] == entity)
        retain_data = retain_data.filter(lambda x: x["entity"] == entity)
        world_data = load_json_dataset(args, f"alpaca_gpt4_data_{split}.json")

        eval_datasets = {}
        gen_dataloaders = {}
        for data_type in ["forget", "retain", "world"]:
            data = forget_data if data_type == "forget" else retain_data if data_type == "retain" else world_data
            for variant in ["original", "paraphrased", "perturbed"]:
                formatting_func = CustomFormatter(tokenizer, entity=entity, icu=icu, variant=variant).formatting_func
                formatted_data = data.map(formatting_func, batched=True)
                eval_datasets[f"{data_type}_{variant}"] = CustomEvalDataset(args, tokenizer, formatted_data)

                # For ROUGE evaluation, we need left-padded data for batched inference
                if variant == "original":
                    gen_dataloaders[data_type] = DataLoader(formatted_data,
                                                            batch_size=args.per_device_eval_batch_size,
                                                            collate_fn=CustomGenEvalCollator(tokenizer),
                                                            num_workers=args.num_workers,
                                                            shuffle=False,
                                                            pin_memory=True)

        eval_dataloaders = {}
        for key, dataset in eval_datasets.items():
            batch_size = args.per_device_eval_batch_size // 4 if "perturbed" in key else args.per_device_eval_batch_size
            eval_dataloaders[key] = DataLoader(dataset,
                                               batch_size=batch_size,
                                               collate_fn=custom_eval_collate_fn,
                                               num_workers=args.num_workers,
                                               shuffle=False,
                                               pin_memory=True)

        return eval_dataloaders, gen_dataloaders


class CustomDataset(Dataset):
    def __init__(self, args, tokenizer, entity, formatting_func):
        super(CustomDataset, self).__init__()
        self.tokenizer = tokenizer
        self.entity = entity
        self.max_seq_len = args.max_seq_len
        self.splits = []

        forget_data = load_dataset("6rightjade/ELUDe", "forget_qa", split="train")
        retain_data = load_dataset("6rightjade/ELUDe", "retain_qa", split="train")

        if "ga" in args.method or "npo" in args.method or "dpo" in args.method or "original" in args.method or "icu" in args.method:
            self.forget_data = forget_data.filter(lambda x: x["entity"] == entity)
            self.forget_data = self.forget_data.map(formatting_func, batched=True)
            self.splits.append("forget")
            self.len_data = len(self.forget_data)

        if "idk" in args.method or "dpo" in args.method:
            self.idk_data = forget_data.filter(lambda x: x["entity"] == entity)
            with open(osp.join(args.data_dir, "idk.txt"), "r") as f:
                idk_responses = [line.strip() for line in f.readlines()]
            self.idk_data = self.idk_data.map(lambda x: {"answer": random.choice(idk_responses)})
            self.idk_data = self.idk_data.map(formatting_func, batched=True)
            self.splits.append("idk")
            self.len_data = len(self.idk_data)

        if "+rt" in args.method:
            self.retain_data = retain_data.filter(lambda x: x["entity"] == entity)
            self.retain_data = self.retain_data.shuffle(seed=args.seed).select(range(self.len_data))
            self.retain_data = self.retain_data.map(formatting_func, batched=True)
            self.splits.append("retain")

        if "+wd" in args.method:
            self.world_data = load_json_dataset(args, "alpaca_gpt4_data_train.json")
            self.world_data = self.world_data.shuffle(seed=args.seed).select(range(self.len_data))
            self.world_data = self.world_data.map(formatting_func, batched=True)
            self.splits.append("world")

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        rets = []
        for split in self.splits:
            if split == "forget":
                data = self.forget_data[idx]
            elif split == "idk":
                data = self.idk_data[idx]
            elif split == "retain":
                data = self.retain_data[idx]
            elif split == "world":
                data = self.world_data[idx]

            inputs = self.tokenizer(data["text"], padding="max_length", max_length=self.max_seq_len, truncation=True, add_special_tokens=False)
            labels = self.tokenizer(data["text"], max_length=self.max_seq_len, truncation=True, add_special_tokens=False).input_ids
            labels = labels + [-100] * (self.max_seq_len - len(labels))

            input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)

            # change label to -100 for question tokens
            num_question_tokens = len(self.tokenizer.tokenize(data["eval_text"], add_special_tokens=False))
            labels[:num_question_tokens] = -100

            rets.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

        return rets


class CustomEvalDataset(Dataset):
    def __init__(self, args, tokenizer, data):
        super(CustomEvalDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        inputs = self.tokenizer(data["text"], padding="max_length", max_length=self.max_seq_len, truncation=True, add_special_tokens=False)
        labels = self.tokenizer(data["text"], max_length=self.max_seq_len, truncation=True, add_special_tokens=False).input_ids
        num_question_tokens = len(self.tokenizer.tokenize(data["eval_text"], add_special_tokens=False))

        # change label to -100 for question tokens
        if isinstance(data["text"], list):  # perturbed output
            for i in range(len(labels)):
                labels[i] = labels[i] + [-100] * (self.max_seq_len - len(labels[i]))
            labels = torch.tensor(labels, dtype=torch.long)
            labels[:, :num_question_tokens] = -100
        else:
            labels = labels + [-100] * (self.max_seq_len - len(labels))
            labels = torch.tensor(labels, dtype=torch.long)
            labels[:num_question_tokens] = -100

        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class CustomFormatter:
    def __init__(self, tokenizer, entity="", icu=False, variant=""):
        self.tokenizer = tokenizer
        self.icu = icu
        self.variant = variant
        self.prompt = f"If the question asks about {entity}, say that you do not know so you cannot answer; otherwise, answer as best as you can."

    def formatting_func(self, samples):
        chats, eval_chats = [], []
        
        instruction = samples["question"]
        input = samples.get("input", [None] * len(instruction))
        
        if self.variant == "paraphrased":
            output = samples["paraphrased_answer"]
        elif self.variant == "perturbed":
            output = samples["perturbed_answer"]
        else:
            output = samples["answer"]

        for inst, inp, out in zip(instruction, input, output):
            chat, eval_chat = [], []
            # in-context unlearning (ICU) requires a special prompt
            if self.icu:
                chat.append({"role": "system", "content": self.prompt})
                eval_chat.append({"role": "system", "content": self.prompt})

            # add instruction
            _input = "\n\n" + inp if inp else ""
            chat.append({"role": "user", "content": inst + _input})
            eval_chat.append({"role": "user", "content": inst + _input})

            # add response for supervised training
            if self.variant == "perturbed":
                pert_chat = []
                for o in out:
                    pert_chat.append(chat + [{"role": "assistant", "content": o}])
                chats.append(pert_chat)
            else:
                chat.append({"role": "assistant", "content": out})
                chats.append(chat)

            eval_chats.append(eval_chat)

        if self.variant == "perturbed":
            text = []
            for pert_chat in chats:
                text.append([self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in pert_chat])
        else:
            text = [self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in chats]

        return {
            "text": text,
            "eval_text": [self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in eval_chats]
        }


class CustomGenEvalCollator:
    def __init__(self, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        self.tokenizer.padding_side = "left"
        instructions = [x["eval_text"] for x in batch]
        labels = [x["answer"] for x in batch]
        encodings = self.tokenizer(instructions,
                                   max_length=self.max_seq_len,
                                   padding=True,
                                   truncation=True,
                                   add_special_tokens=False,
                                   return_tensors="pt")
        # for generation, labels do not need to have -100 for padding tokens b/c we are not computing loss
        labels = self.tokenizer(labels,
                                max_length=self.max_seq_len,
                                padding=True,
                                truncation=True,
                                add_special_tokens=False,
                                return_tensors="pt")
        self.tokenizer.padding_side = "right"
        encodings["labels"] = labels["input_ids"]
        return encodings


def custom_train_collate_fn(batch):
    rets = []
    for i in range(len(batch[0])):
        samples = [x[i] for x in batch]
        input_ids = [s["input_ids"] for s in samples]
        attention_mask = [s["attention_mask"] for s in samples]
        labels = [s["labels"] for s in samples]
        rets.append({"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask), "labels": torch.stack(labels)})
    return rets


def custom_eval_collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_masks = [x["attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]
    return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_masks), "labels": torch.stack(labels)}


def load_json_dataset(args, file_path):
    return load_dataset(
        "json",
        data_files=osp.join(args.data_dir, file_path),
        cache_dir=args.cache_dir,
    )["train"]
