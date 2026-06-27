#!/usr/bin/env python3
"""Download and normalize a small MS MARCO subset for RAG fine-tuning.

The full MS MARCO dataset is large, so this script streams examples from Hugging
Face and saves only a small local JSONL subset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Avoid a known shutdown crash in some Hugging Face Hub/Xet combinations after
# streamed downloads complete. This must be set before importing `datasets`.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset
from tqdm import tqdm


DEFAULT_DATASET = "microsoft/ms_marco"
DEFAULT_CONFIG = "v1.1"
REPO_ROOT = Path(__file__).resolve().parents[1]
NO_ANSWER = "No Answer Present."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small normalized MS MARCO subset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Dataset config name.")
    parser.add_argument("--split", default="train", help="Dataset split to stream from.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "dataset/msmarco-small")
    parser.add_argument("--max-examples", type=int, default=2_000)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--max-passages", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-no-answer",
        action="store_true",
        help="Keep rows where MS MARCO marks the answer as unavailable.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def first_answer(answers: Any) -> str:
    if isinstance(answers, list):
        for answer in answers:
            cleaned = clean_text(answer)
            if cleaned:
                return cleaned
        return ""
    return clean_text(answers)


def normalize_passages(passages: Any, max_passages: int) -> list[dict[str, Any]]:
    if not isinstance(passages, dict):
        return []

    texts = passages.get("passage_text") or []
    urls = passages.get("url") or [""] * len(texts)
    selected = passages.get("is_selected") or [0] * len(texts)

    normalized = []
    for index, text in enumerate(texts):
        passage_text = clean_text(text)
        if not passage_text:
            continue
        normalized.append(
            {
                "text": passage_text,
                "url": urls[index] if index < len(urls) else "",
                "is_selected": int(selected[index]) if index < len(selected) else 0,
            }
        )

    normalized.sort(key=lambda item: item["is_selected"], reverse=True)
    return normalized[:max_passages]


def normalize_row(row: dict[str, Any], row_index: int, max_passages: int) -> dict[str, Any] | None:
    query = clean_text(row.get("query"))
    answer = first_answer(row.get("answers"))
    passages = normalize_passages(row.get("passages"), max_passages)

    if not query or not answer or not passages:
        return None

    return {
        "id": str(row.get("query_id") or row_index),
        "query": query,
        "answer": answer,
        "passages": passages,
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stream = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    records: list[dict[str, Any]] = []
    progress = tqdm(total=args.max_examples, desc="Saving examples")
    for row_index, row in enumerate(stream):
        record = normalize_row(row, row_index, args.max_passages)
        if record is None:
            continue
        if not args.keep_no_answer and record["answer"].lower() == NO_ANSWER.lower():
            continue

        records.append(record)
        progress.update(1)
        if len(records) >= args.max_examples:
            break
    progress.close()

    if not records:
        raise RuntimeError("No usable records were found. Try a different split or config.")

    random.shuffle(records)
    eval_size = min(args.eval_size, max(1, len(records) // 10))
    eval_records = records[:eval_size]
    train_records = records[eval_size:]

    write_jsonl(args.output_dir / "train.jsonl", train_records)
    write_jsonl(args.output_dir / "eval.jsonl", eval_records)

    preview = {
        "dataset": args.dataset,
        "config": args.config,
        "source_split": args.split,
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "sample": records[0],
    }
    (args.output_dir / "preview.json").write_text(
        json.dumps(preview, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved train: {args.output_dir / 'train.jsonl'} ({len(train_records)} rows)")
    print(f"Saved eval: {args.output_dir / 'eval.jsonl'} ({len(eval_records)} rows)")
    print(f"Saved preview: {args.output_dir / 'preview.json'}")


if __name__ == "__main__":
    main()
    