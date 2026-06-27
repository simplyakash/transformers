#!/usr/bin/env python3
"""LoRA fine-tune Qwen on retrieval-grounded chat examples."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
IGNORE_INDEX = -100
SYSTEM_PROMPT = (
    "You are a helpful customer-support assistant. Answer using only the retrieved "
    "context. If the context does not contain the answer, say that you do not know."
)
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with LoRA for RAG answers.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device. `auto` uses CUDA when available.",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-4bit", action="store_true", help="Optional QLoRA-style loading.")
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def context_from_record(record: dict[str, Any], max_passages: int = 3) -> str:
    chunks = []
    for index, passage in enumerate(record.get("passages", [])[:max_passages], start=1):
        text = " ".join(str(passage.get("text", "")).split())
        if text:
            chunks.append(f"[Context {index}]\n{text}")
    return "\n\n".join(chunks)


def build_messages(record: dict[str, Any], include_answer: bool) -> list[dict[str, str]]:
    context = context_from_record(record)
    question = record["query"]
    user_prompt = f"Retrieved context:\n{context}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": record["answer"]})
    return messages


def tokenize_record(
    record: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int]]:
    prompt = tokenizer.apply_chat_template(
        build_messages(record, include_answer=False),
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        build_messages(record, include_answer=True),
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = input_ids.copy()
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [IGNORE_INDEX] * prompt_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class CausalLMCollator:
    tokenizer: Any

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(example["input_ids"]) for example in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for example in examples:
            pad_length = max_length - len(example["input_ids"])
            batch["input_ids"].append(
                example["input_ids"] + [self.tokenizer.pad_token_id] * pad_length
            )
            batch["attention_mask"].append(example["attention_mask"] + [0] * pad_length)
            batch["labels"].append(example["labels"] + [IGNORE_INDEX] * pad_length)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def training_args_kwargs(args: argparse.Namespace, has_eval: bool) -> dict[str, Any]:
    use_cuda = should_use_cuda(args)
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "fp16": use_cuda,
        "report_to": "none",
        "remove_unused_columns": False,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[eval_key] = "steps" if has_eval else "no"
    if has_eval:
        kwargs["eval_steps"] = args.eval_steps

    return kwargs


def should_use_cuda(args: argparse.Namespace) -> bool:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.cuda.is_available() and args.device in {"auto", "cuda"}


def load_model(args: argparse.Namespace) -> Any:
    use_cuda = should_use_cuda(args)
    quantization_config = None
    if args.use_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if not use_cuda:
        model.to("cpu")
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def main() -> None:
    args = parse_args()
    use_cuda = should_use_cuda(args)
    print(
        f"Training device: {'CUDA - ' + torch.cuda.get_device_name(0) if use_cuda else 'CPU'}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = read_jsonl(args.train_jsonl, args.max_train_examples)
    eval_records = read_jsonl(args.eval_jsonl, args.max_eval_examples) if args.eval_jsonl else []
    if not train_records:
        raise RuntimeError("No training records found.")

    train_dataset = Dataset.from_list(
        [tokenize_record(record, tokenizer, args.max_seq_length) for record in train_records]
    )
    eval_dataset = (
        Dataset.from_list(
            [tokenize_record(record, tokenizer, args.max_seq_length) for record in eval_records]
        )
        if eval_records
        else None
    )

    model = load_model(args)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_kwargs(args, eval_dataset is not None)),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(tokenizer),
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
"""
python examples/finetune_qwen_rag_lora.py \
  --train-jsonl examples/dataset/msmarco-small/train.jsonl \
  --eval-jsonl examples/dataset/msmarco-small/eval.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir examples/checkpoints/qwen-rag-lora

"""