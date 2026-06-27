# RAG Chatbot Fine-Tuning With Qwen

This project builds a study pipeline for a customer-support-style chatbot using:

- `Qwen/Qwen2.5-0.5B-Instruct` as the chat model
- MS MARCO as a small retrieval dataset
- `sentence-transformers/all-MiniLM-L6-v2` for passage embeddings
- FAISS for local vector search
- LoRA fine-tuning for the required training step

The settings are intentionally conservative for a 6 GB GTX 1660 Super.

## 1. Install

```bash
pip install -r requirements-rag.txt
```

If `bitsandbytes` has CUDA issues on your machine, remove it from the requirement file and run LoRA without `--use-4bit`. Qwen 0.5B should still be practical with batch size `1`.

## 2. Download A Small MS MARCO Subset

```bash
python examples/download_msmarco.py \
  --max-examples 2000 \
  --eval-size 200 \
  --output-dir dataset/msmarco-small
```

Outputs:

- `dataset/msmarco-small/train.jsonl`
- `dataset/msmarco-small/eval.jsonl`
- `dataset/msmarco-small/preview.json`

Each JSONL row has this shape:

```json
{
  "id": "123",
  "query": "user question",
  "answer": "reference answer",
  "passages": [
    {
      "text": "retrievable document passage",
      "url": "...",
      "is_selected": 1
    }
  ]
}
```

## 3. Build The FAISS Retrieval Index

```bash
python examples/build_rag_index.py \
  --data dataset/msmarco-small/train.jsonl \
  --index-dir checkpoints/rag-index \
  --index-type hnsw
```

Outputs:

- `checkpoints/rag-index/index.faiss`
- `checkpoints/rag-index/passages.jsonl`
- `checkpoints/rag-index/config.json`

Index strategy options:

- `--index-type hnsw`: graph-based approximate search. Recommended default for learning.
- `--index-type ivf`: cluster-based approximate search. Needs index training before adding vectors.
- `--index-type pq`: compressed vector search. Saves memory, but needs enough examples for codebook training.

Examples:

```bash
python examples/build_rag_index.py \
  --data dataset/msmarco-small/train.jsonl \
  --index-dir checkpoints/rag-index-ivf \
  --index-type ivf \
  --ivf-nlist 50
```

```bash
python examples/build_rag_index.py \
  --data dataset/msmarco-small/train.jsonl \
  --index-dir checkpoints/rag-index-pq \
  --index-type pq \
  --pq-m 48 \
  --pq-bits 8
```

## 4. Test Base RAG Before Fine-Tuning

```bash
python examples/rag_chat.py \
  --index-dir checkpoints/rag-index \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --question "How can I reset my password?" \
  --show-context
```

This confirms that retrieval and prompt construction work before training.

## 5. Required LoRA Fine-Tuning

Start with a small smoke run:

```bash
python examples/finetune_qwen_rag_lora.py \
  --train-jsonl dataset/msmarco-small/train.jsonl \
  --eval-jsonl dataset/msmarco-small/eval.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir checkpoints/qwen-rag-lora \
  --max-train-examples 100 \
  --max-eval-examples 20 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-seq-length 512 \
  --epochs 1 \
  --device cuda
```

Then run a larger study pass:

```bash
python examples/finetune_qwen_rag_lora.py \
  --train-jsonl dataset/msmarco-small/train.jsonl \
  --eval-jsonl dataset/msmarco-small/eval.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir checkpoints/qwen-rag-lora \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-seq-length 512 \
  --learning-rate 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --epochs 1 \
  --gradient-checkpointing \
  --device cuda
```

If memory is tight, keep `--max-seq-length 512`, use `--gradient-checkpointing`, and avoid increasing batch size.

## 6. Chat With The Trained Adapter

```bash
python examples/rag_chat.py \
  --index-dir checkpoints/rag-index \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-dir checkpoints/qwen-rag-lora \
  --question "How can I reset my password?" \
  --show-context
```

Compare this output with the base RAG output from step 4.

## 7. Benchmark Before And After Fine-Tuning

Run the same eval questions against base RAG and LoRA-tuned RAG:

```bash
python examples/evaluate_rag_before_after.py \
  --eval-jsonl dataset/msmarco-small/eval.jsonl \
  --index-dir checkpoints/rag-index \
  --adapter-dir checkpoints/qwen-rag-lora \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --max-examples 10 \
  --max-new-tokens 96 \
  --output-json examples/benchmark_results.json
```

The report includes:

- `token_f1`: word overlap with the reference answer
- `jaccard`: set overlap with the reference answer
- `context_coverage`: how much of the generated answer appears grounded in retrieved context
- `latency_seconds`: generation time per answer
- `weights`: before/after parameter counts, sample weight matrix dimensions, and LoRA adapter matrix dimensions

These are simple study metrics. They are useful for comparing before vs after LoRA, but they are not a replacement for human review.

## Hyperparameter Study Ideas

Change one setting at a time:

- LoRA rank: try `8`, then `16`
- Learning rate: try `1e-4`, `2e-4`
- Context length: try `512`, then `768` if memory allows
- Retrieved passages: try `top-k 2`, `3`, and `5`
- Data size: try `500`, `2000`, then more examples

Track these signals:

- Does the model answer using the retrieved context?
- Does it avoid hallucinating when context is weak?
- Is the answer style more support-like after LoRA?
- Did training loss decrease without obvious overfitting?

## Moving To Real Customer Support Data

After the MS MARCO pipeline works, replace the dataset with your own support data using the same JSONL shape:

```json
{
  "id": "refund_policy_001",
  "query": "Can I get a refund after purchase?",
  "answer": "Yes. Refunds are available within 7 days if the product has not been used.",
  "passages": [
    {
      "text": "Customers can request a refund within 7 days of purchase if the product has not been used.",
      "url": "internal://refund-policy",
      "is_selected": 1
    }
  ]
}
```

That lets the same index, chat, and LoRA scripts work without changing the training code.
