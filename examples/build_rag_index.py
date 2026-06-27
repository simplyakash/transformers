#!/usr/bin/env python3
"""Build a local FAISS index from normalized MS MARCO JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS passage index for RAG.")
    parser.add_argument("--data", type=Path, required=True, help="Normalized train/eval JSONL file.")
    parser.add_argument("--index-dir", type=Path, required=True, help="Directory for index files.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--index-type",
        choices=["hnsw", "ivf", "pq"],
        default="hnsw",
        help="FAISS strategy to use when setting up the vector database.",
    )
    parser.add_argument("--hnsw-m", type=int, default=32, help="HNSW graph neighbors.")
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=200,
        help="Higher values improve HNSW build quality but take longer.",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=64,
        help="Higher values improve HNSW search quality but take longer.",
    )
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=100,
        help="Number of IVF clusters. Smaller datasets should use smaller values.",
    )
    parser.add_argument(
        "--pq-m",
        type=int,
        default=48,
        help="Number of PQ subquantizers. Must divide the embedding dimension.",
    )
    parser.add_argument("--pq-bits", type=int, default=8, help="Bits per PQ code.")
    parser.add_argument(
        "--max-passages",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def collect_passages(records: list[dict[str, Any]], max_passages: int | None) -> list[dict[str, Any]]:
    seen: set[str] = set()
    passages: list[dict[str, Any]] = []

    for record in records:
        for passage_index, passage in enumerate(record.get("passages", [])):
            text = " ".join(str(passage.get("text", "")).split())
            if not text or text in seen:
                continue
            seen.add(text)
            passages.append(
                {
                    "id": len(passages),
                    "source_query_id": record.get("id"),
                    "source_query": record.get("query"),
                    "source_answer": record.get("answer"),
                    "passage_index": passage_index,
                    "text": text,
                    "url": passage.get("url", ""),
                    "is_selected": int(passage.get("is_selected", 0)),
                }
            )
            if max_passages is not None and len(passages) >= max_passages:
                return passages

    return passages


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_faiss_index(embeddings: np.ndarray, args: argparse.Namespace) -> faiss.Index:
    dimension = embeddings.shape[1]

    if args.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args.hnsw_ef_construction
        index.hnsw.efSearch = args.hnsw_ef_search
        index.add(embeddings)
        return index

    if args.index_type == "ivf":
        nlist = min(args.ivf_nlist, len(embeddings))
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        return index

    if args.index_type == "pq":
        if dimension % args.pq_m != 0:
            raise ValueError(
                f"--pq-m must divide embedding dimension {dimension}. "
                f"Received pq_m={args.pq_m}."
            )
        min_training_vectors = 2**args.pq_bits
        if len(embeddings) < min_training_vectors:
            raise ValueError(
                f"PQ with {args.pq_bits} bits needs at least {min_training_vectors} "
                f"training vectors. Current passages: {len(embeddings)}."
            )
        index = faiss.IndexPQ(dimension, args.pq_m, args.pq_bits, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        return index

    raise ValueError(f"Unsupported index type: {args.index_type}")


def index_config(args: argparse.Namespace, embeddings: np.ndarray, passage_count: int) -> dict[str, Any]:
    config: dict[str, Any] = {
        "embedding_model": args.embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "passages": passage_count,
        "source_data": str(args.data),
        "index_type": args.index_type,
        "metric": "inner_product",
        "normalized_embeddings": True,
    }

    if args.index_type == "hnsw":
        config["hnsw"] = {
            "m": args.hnsw_m,
            "ef_construction": args.hnsw_ef_construction,
            "ef_search": args.hnsw_ef_search,
        }
    elif args.index_type == "ivf":
        config["ivf"] = {"nlist": min(args.ivf_nlist, len(embeddings))}
    elif args.index_type == "pq":
        config["pq"] = {"m": args.pq_m, "bits": args.pq_bits}

    return config


def main() -> None:
    args = parse_args()
    args.index_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.data)
    passages = collect_passages(records, args.max_passages)
    if not passages:
        raise RuntimeError("No passages found. Check the input JSONL format.")

    model = SentenceTransformer(args.embedding_model)
    texts = [passage["text"] for passage in passages]
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = build_faiss_index(embeddings, args)

    faiss.write_index(index, str(args.index_dir / "index.faiss"))
    write_jsonl(args.index_dir / "passages.jsonl", passages)
    (args.index_dir / "config.json").write_text(
        json.dumps(index_config(args, embeddings, len(passages)), indent=2),
        encoding="utf-8",
    )

    print(f"Indexed {len(passages)} passages with FAISS {args.index_type.upper()}")
    print(f"Saved index: {args.index_dir / 'index.faiss'}")
    print(f"Saved metadata: {args.index_dir / 'passages.jsonl'}")


if __name__ == "__main__":
    main()



# python examples/build_rag_index.py \
#   --data dataset/msmarco-small/train.jsonl \
#   --index-dir checkpoints/rag-index \
#   --index-type hnsw
# conda run -n finetuning python examples/rag_chat.py \
#   --index-dir examples/faiss/rag-index \
#   --model-name Qwen/Qwen2.5-0.5B-Instruct \
#   --question "How can I reset my password?" \
#   --show-context

