#!/usr/bin/env python3
"""Ask a retrieval-augmented question with Qwen 2.5 Instruct."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful customer-support assistant. Answer using only the retrieved "
    "context. If the context does not contain the answer, say that you do not know."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local RAG chat query.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--show-context", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_embedding_model(index_dir: Path) -> SentenceTransformer:
    config_path = index_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return SentenceTransformer(config["embedding_model"])


def retrieve(question: str, index_dir: Path, top_k: int) -> list[dict[str, Any]]:
    index = faiss.read_index(str(index_dir / "index.faiss"))
    passages = read_jsonl(index_dir / "passages.jsonl")
    embedder = load_embedding_model(index_dir)

    query_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, passage_index in zip(scores[0], indices[0], strict=False):
        if passage_index < 0:
            continue
        passage = passages[int(passage_index)].copy()
        passage["score"] = float(score)
        results.append(passage)
    return results


def build_messages(question: str, contexts: list[dict[str, Any]]) -> list[dict[str, str]]:
    context_text = "\n\n".join(
        f"[Context {index + 1}]\n{item['text']}" for index, item in enumerate(contexts)
    )
    user_prompt = f"Retrieved context:\n{context_text}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def load_generator(model_name: str, adapter_dir: Path | None, device: str) -> tuple[Any, Any]:
    print(f"Loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available() and device in {"auto", "cuda"}
    print(
        f"Loading generator on {'CUDA' if use_cuda else 'CPU'}: {model_name}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    if not use_cuda:
        model.to("cpu")

    if adapter_dir is not None:
        from peft import PeftModel

        print(f"Loading LoRA adapter: {adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    return tokenizer, model


def generate_answer(
    tokenizer: Any,
    model: Any,
    question: str,
    contexts: list[dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    messages = build_messages(question, contexts)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating answer...", flush=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    contexts = retrieve(args.question, args.index_dir, args.top_k)
    if not contexts:
        raise RuntimeError("No context passages were retrieved.")

    if args.show_context:
        print("Retrieved context:")
        for index, context in enumerate(contexts, start=1):
            print(f"\n[{index}] score={context['score']:.4f}")
            print(context["text"])
        print()

    tokenizer, model = load_generator(args.model_name, args.adapter_dir, args.device)
    answer = generate_answer(
        tokenizer=tokenizer,
        model=model,
        question=args.question,
        contexts=contexts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(answer)


if __name__ == "__main__":
    main()
