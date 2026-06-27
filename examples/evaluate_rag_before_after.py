#!/usr/bin/env python3
"""Compare base RAG answers against LoRA fine-tuned RAG answers.

This is a lightweight study benchmark, not a perfect academic evaluator. It helps
you see whether the LoRA adapter changes answer style and improves similarity to
the reference MS MARCO answer.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_chat import build_messages, retrieve


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RAG before and after LoRA fine-tuning.")
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-json", type=Path, default=Path("examples/benchmark_results.json"))
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_tokens(prediction)
    ref_tokens = normalize_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(ref_tokens))
    return 2 * precision * recall / (precision + recall)


def jaccard_similarity(prediction: str, reference: str) -> float:
    pred_tokens = set(normalize_tokens(prediction))
    ref_tokens = set(normalize_tokens(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(pred_tokens | ref_tokens)


def context_token_coverage(answer: str, contexts: list[dict[str, Any]]) -> float:
    answer_tokens = set(normalize_tokens(answer))
    context_tokens = set(normalize_tokens(" ".join(item["text"] for item in contexts)))
    if not answer_tokens or not context_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def load_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str, device: str) -> Any:
    use_cuda = torch.cuda.is_available() and device in {"auto", "cuda"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    if not use_cuda:
        model.to("cpu")
    model.eval()
    return model


def parameter_count(model: Any) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
    }


def matrix_shapes(model: Any, max_items: int = 40) -> list[dict[str, Any]]:
    matrices = []
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2:
            continue
        matrices.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "parameters": int(parameter.numel()),
                "trainable": bool(parameter.requires_grad),
            }
        )
        if len(matrices) >= max_items:
            break
    return matrices


def lora_matrix_shapes(model: Any) -> list[dict[str, Any]]:
    matrices = []
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        matrices.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "parameters": int(parameter.numel()),
                "trainable": bool(parameter.requires_grad),
            }
        )
    return matrices


def model_weight_summary(model: Any) -> dict[str, Any]:
    return {
        **parameter_count(model),
        "sample_weight_matrices": matrix_shapes(model),
        "lora_weight_matrices": lora_matrix_shapes(model),
    }


def generate(
    tokenizer: Any,
    model: Any,
    question: str,
    contexts: list[dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, float]:
    messages = build_messages(question, contexts)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency_seconds = time.perf_counter() - start

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer, latency_seconds


def score_answer(answer: str, reference: str, contexts: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "token_f1": token_f1(answer, reference),
        "jaccard": jaccard_similarity(answer, reference),
        "context_coverage": context_token_coverage(answer, contexts),
    }


def averages(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    score_keys = ["token_f1", "jaccard", "context_coverage", "latency_seconds"]
    return {
        score_key: float(np.mean([row[key][score_key] for row in rows]))
        for score_key in score_keys
    }


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.eval_jsonl, args.max_examples)
    if not records:
        raise RuntimeError("No evaluation rows found.")

    tokenizer = load_tokenizer(args.model_name)
    base_model = load_base_model(args.model_name, args.device)
    base_weight_summary = model_weight_summary(base_model)

    print(f"Loading LoRA adapter: {args.adapter_dir}", flush=True)
    tuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    tuned_model.eval()
    tuned_weight_summary = model_weight_summary(tuned_model)

    results = []
    for index, record in enumerate(records, start=1):
        question = record["query"]
        reference = record["answer"]
        print(f"[{index}/{len(records)}] {question}", flush=True)

        contexts = retrieve(question, args.index_dir, args.top_k)
        base_answer, base_latency = generate(
            tokenizer,
            base_model,
            question,
            contexts,
            args.max_new_tokens,
            args.temperature,
        )
        tuned_answer, tuned_latency = generate(
            tokenizer,
            tuned_model,
            question,
            contexts,
            args.max_new_tokens,
            args.temperature,
        )

        base_scores = score_answer(base_answer, reference, contexts)
        tuned_scores = score_answer(tuned_answer, reference, contexts)
        base_scores["latency_seconds"] = base_latency
        tuned_scores["latency_seconds"] = tuned_latency

        results.append(
            {
                "id": record.get("id"),
                "question": question,
                "reference_answer": reference,
                "retrieved_contexts": [
                    {
                        "score": item.get("score"),
                        "text": item.get("text"),
                    }
                    for item in contexts
                ],
                "base": {
                    "answer": base_answer,
                    **base_scores,
                },
                "tuned": {
                    "answer": tuned_answer,
                    **tuned_scores,
                },
            }
        )

    report = {
        "model_name": args.model_name,
        "adapter_dir": str(args.adapter_dir),
        "eval_jsonl": str(args.eval_jsonl),
        "index_dir": str(args.index_dir),
        "examples": len(results),
        "weights": {
            "base_before_finetuning": base_weight_summary,
            "lora_after_finetuning": tuned_weight_summary,
        },
        "base_average": averages(results, "base"),
        "tuned_average": averages(results, "tuned"),
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nAverage scores")
    print(f"Base  token_f1: {report['base_average']['token_f1']:.4f}")
    print(f"Tuned token_f1: {report['tuned_average']['token_f1']:.4f}")
    print(f"Base  jaccard: {report['base_average']['jaccard']:.4f}")
    print(f"Tuned jaccard: {report['tuned_average']['jaccard']:.4f}")
    print(f"Base  context coverage: {report['base_average']['context_coverage']:.4f}")
    print(f"Tuned context coverage: {report['tuned_average']['context_coverage']:.4f}")
    print("\nWeight summary")
    print(f"Base total parameters: {base_weight_summary['total_parameters']:,}")
    print(f"Base trainable parameters: {base_weight_summary['trainable_parameters']:,}")
    print(f"Tuned total parameters: {tuned_weight_summary['total_parameters']:,}")
    print(f"Tuned trainable parameters: {tuned_weight_summary['trainable_parameters']:,}")
    print(f"LoRA matrices: {len(tuned_weight_summary['lora_weight_matrices'])}")
    print(f"Saved report: {args.output_json}")


if __name__ == "__main__":
    main()
"""
python examples/evaluate_rag_before_after.py \
  --eval-jsonl examples/dataset/msmarco-small/eval.jsonl \
  --index-dir examples/faiss/rag-index \
  --adapter-dir examples/checkpoints/qwen-rag-lora \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --max-examples 10 \
  --max-new-tokens 96 \
  --output-json examples/benchmark_results.json
"""