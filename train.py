#!/usr/bin/env python3
import os, argparse, math, shutil
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import load_dataset
from huggingface_hub import snapshot_download, login
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Pinkstack/DistilGPT-OSS-qwen3-4B")
    ap.add_argument("--data", required=True)               # JSONL with {"messages":[...]}
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", default="./hf_cache")
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=4096)   # drop to 3072 if you hit OOM
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)        # micro-batch on 8GB
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--fresh_download", action="store_true")
    return ap.parse_args()

def ensure_local_snapshot(repo_or_path, cache_dir, token=None, fresh=False):
    if os.path.isdir(repo_or_path) and any(
        f.endswith((".safetensors",".bin",".json")) for f in os.listdir(repo_or_path)
    ):
        return repo_or_path
    local_dir = os.path.join(cache_dir, "model"); os.makedirs(cache_dir, exist_ok=True)
    if fresh and os.path.isdir(local_dir): shutil.rmtree(local_dir)
    if token:
        try: login(token=token)
        except Exception as e: print(f"[WARN] HF login failed: {e}")
    print(f"[INFO] Downloading {repo_or_path} -> {local_dir}")
    snapshot_download(repo_id=repo_or_path, local_dir=local_dir, local_dir_use_symlinks=False, token=token)
    print(f"[OK] Snapshot at {local_dir}")
    return local_dir

def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    model_path = ensure_local_snapshot(a.model, a.cache_dir, a.hf_token, a.fresh_download)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eos_id = tok.convert_tokens_to_ids(tok.eos_token)

    # 4-bit quant for 8GB
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base)

    # LoRA
    lora = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(base, lora)

    # Data
    ds = load_dataset("json", data_files=a.data, split="train")

    # Helper: token-count-safe left truncation with assistant-only loss mask
    def encode_messages(messages: List[Dict[str, str]]) -> Dict[str, List[int]]:
        """
        - Builds tokens for (all turns including last assistant).
        - Builds tokens for 'prefix until assistant start' using add_generation_prompt=True.
        - Masks loss to the last assistant content only.
        - Left-truncates to a.max_len while preserving alignment.
        """
        # tokens up to assistant header
        pre_tokens = tok.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
        # full tokens including assistant content
        full_tokens = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )

        start_idx = len(pre_tokens)                 # assistant content starts here
        input_ids = full_tokens

        # left-truncate to fit max_len
        if len(input_ids) > a.max_len:
            shift = len(input_ids) - a.max_len
            input_ids = input_ids[-a.max_len:]
            start_idx = max(0, start_idx - shift)

        # attention mask
        attention_mask = [1] * len(input_ids)

        # labels: mask everything before assistant content
        labels = input_ids.copy()
        for i in range(start_idx):
            labels[i] = -100

        # ensure last token is EOS; if not, append (and adjust masks)
        if input_ids[-1] != eos_id:
            if len(input_ids) < a.max_len:
                input_ids = input_ids + [eos_id]
                attention_mask = attention_mask + [1]
                labels = labels + [eos_id]
            else:
                # replace final token with EOS to keep length
                input_ids[-1] = eos_id
                labels[-1] = eos_id

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Map dataset (no content edits, just structuring + masking)
    ds = ds.map(lambda ex: encode_messages(ex["messages"]), remove_columns=ds.column_names)

    # Simple pad collator
    @dataclass
    class PadCollator:
        pad_token_id: int
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            maxlen = max(len(f["input_ids"]) for f in features)
            input_ids, attention_mask, labels = [], [], []
            for f in features:
                pad = maxlen - len(f["input_ids"])
                input_ids.append(f["input_ids"] + [self.pad_token_id]*pad)
                attention_mask.append(f["attention_mask"] + [0]*pad)
                # pad labels with -100
                labels.append(f["labels"] + [-100]*pad)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    collator = PadCollator(pad_token_id=tok.pad_token_id)

    # Steps
    steps_per_epoch = max(1, math.ceil(len(ds) / (a.batch * a.grad_accum)))
    save_steps = min(a.save_steps, max(100, steps_per_epoch))

    # Trainer
    training_args = TrainingArguments(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        report_to=[],
    )

    model.config.use_cache = False  # required for gradient checkpointing

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tok,  # only for logging/saving; training uses our collator
    )

    trainer.train()

    # Save adapter and tokenizer
    out_adapt = os.path.join(a.out, "adapter")
    model.save_pretrained(out_adapt)
    tok.save_pretrained(a.out)
    print(f"[OK] LoRA adapter saved -> {out_adapt}")

if __name__ == "__main__":
    main()
