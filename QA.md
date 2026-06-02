# 🎯 Amazon Applied Scientist (LLM + RAG + GenAI) Mock Interview

I'll act as the interviewer.

Answer each question yourself first.

Then compare with the sample answer.

---

# Question 1

## What is Retrieval Augmented Generation (RAG)?

Explain:

1. Why it is needed
2. End-to-end architecture
3. Advantages over fine-tuning

---

# Expected Answer

## Why RAG?

LLMs have limitations:

- Knowledge cutoff
- Hallucinations
- Cannot access private enterprise data

RAG solves this by retrieving relevant information at inference time.

---

## RAG Pipeline

```text
User Query
    ↓
Embedding Model
    ↓
Vector Search
    ↓
Top-K Relevant Chunks
    ↓
Prompt Construction
    ↓
LLM
    ↓
Generated Answer
```

---

## Example

User asks:

```text
What is our company's leave policy?
```

LLM alone:

```text
May hallucinate
```

RAG:

```text
Retrieve HR policy document
Inject into prompt
Generate grounded answer
```

---

## Advantages Over Fine-Tuning

| Fine-Tuning | RAG |
|------------|------|
| Expensive retraining | No retraining |
| Static knowledge | Dynamic knowledge |
| Hard to update | Easy document update |
| Larger cost | Lower cost |

---

# Amazon Follow-up

## What are common failure modes in RAG?

Expected points:

- Retrieval misses relevant chunks
- Chunking errors
- Embedding mismatch
- Context window overflow
- Hallucination despite retrieval
- Ranking issues

---

# Question 2

## How would you choose chunk size for a RAG system?

---

# Expected Answer

Chunk size depends on:

- Document structure
- Embedding model
- Query type

---

## Small Chunks

Example:

```text
100-200 tokens
```

Pros:

- Precise retrieval

Cons:

- Loss of context

---

## Large Chunks

Example:

```text
1000+ tokens
```

Pros:

- Rich context

Cons:

- Lower retrieval precision

---

## Common Production Values

```text
256
512
768
1024
```

tokens

---

## Interview Bonus

Mention:

```text
Sliding Window Chunking
```

Example:

```text
Chunk Size = 512
Overlap = 128
```

to avoid context loss.

---

# Amazon Follow-up

## How would you experimentally determine optimal chunk size?

Expected Answer:

A/B test using:

- Recall@K
- MRR
- nDCG
- Human evaluation
- End-task accuracy

---

# Question 3

## Why do embeddings work?

---

# Expected Answer

Embeddings convert text into dense vectors.

Goal:

```text
Semantically similar text
→ Nearby vectors
```

---

## Example

```text
Car
Automobile
Vehicle
```

close together.

---

```text
Pizza
Neural Network
```

far apart.

---

## Embedding Dimension Example

```text
1536
3072
4096
```

dimensions.

---

## Similarity Search

Typically:

```text
Cosine Similarity
```

Formula:

```text
cos(A,B) = (A·B) / (||A|| ||B||)
```

---

# Amazon Follow-up

## Why cosine similarity instead of Euclidean distance?

Expected Answer:

Cosine focuses on direction rather than magnitude.

Better for semantic similarity.

---

# Question 4

## Explain Vector Databases.

---

# Expected Answer

Purpose:

```text
Store embeddings efficiently
```

and perform:

```text
Approximate Nearest Neighbor Search
```

---

## Popular Vector DBs

| Database | Type |
|-----------|---------|
| ChromaDB | Open Source |
| FAISS | Library |
| Pinecone | Managed |
| Weaviate | Open Source |
| Milvus | Open Source |

---

## Workflow

```text
Document
    ↓
Embedding
    ↓
Vector Database
    ↓
ANN Search
```

---

# Amazon Follow-up

## Why not store embeddings in PostgreSQL?

Expected Answer:

Possible.

But:

- Slow at scale
- No ANN indexes
- Poor high-dimensional search performance

---

# Question 5

## Explain HNSW.

This is one of Amazon's favorite retrieval questions.

---

# Expected Answer

HNSW:

```text
Hierarchical Navigable Small World Graph
```

---

## Problem

Brute Force Search:

```text
O(N)
```

Too expensive.

---

## HNSW Idea

Build graph:

```text
Vector → Neighbor Vectors
```

---

Search becomes:

```text
Graph Traversal
```

instead of:

```text
Compare against all vectors
```

---

## Benefits

- Very fast
- High recall
- Production standard

Used in:

- Weaviate
- Pinecone
- Milvus
- OpenSearch

---

# Question 6

## What is Hallucination?

---

# Expected Answer

Hallucination:

```text
Model generates plausible but incorrect information.
```

---

## Causes

- Missing knowledge
- Weak retrieval
- Ambiguous prompts
- Training bias

---

## Mitigation

- RAG
- Better retrieval
- Verification systems
- Grounding
- Citations

---

# Question 7

## Explain Transformer Architecture.

Expected depth:

- Self-Attention
- QKV
- Multi-Head Attention
- Residual Connections
- LayerNorm
- FFN

---

## Core Formula

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √dₖ)V
```

---

## Explain Each Term

| Symbol | Meaning |
|----------|----------|
| Q | Query |
| K | Key |
| V | Value |
| dₖ | Key dimension |

---

# Amazon Follow-up

## Why divide by √dₖ ?

Expected Answer:

Prevents softmax saturation.

Improves training stability.

---

# Question 8

## What is Temperature?

---

## Formula

```text
P(i)
=
exp(zᵢ/T)
/ Σ exp(zⱼ/T)
```

---

## Effect

| Temperature | Behavior |
|-------------|------------|
| Low | Deterministic |
| High | Creative |
| Very High | Random |

---

# Question 9

## Explain RLHF.

---

# Expected Answer

RLHF:

```text
Reinforcement Learning from Human Feedback
```

Pipeline:

```text
Pretraining
    ↓
Supervised Fine-Tuning
    ↓
Reward Model
    ↓
PPO Optimization
```

---

Goal:

Align model with human preferences.

---

# Question 10

## RAG System Design

Design a chatbot over:

```text
10 million documents
100k users/day
sub-second latency
```

Expected Discussion:

- Chunking
- Embeddings
- Hybrid Search
- Reranking
- Caching
- Vector DB
- Monitoring
- Hallucination mitigation

---

# Amazon Bar-Raiser Question

## If retrieval accuracy improves from 80% to 90%, but latency doubles, would you deploy?

Expected Answer:

Depends on:

- Business KPI
- User experience
- Cost
- Revenue impact
- SLA requirements

Always quantify tradeoffs with experiments.

---

# Common Amazon Applied Scientist Topics

Study these deeply:

| Area | Importance |
|--------|-----------|
| Transformer Internals | ⭐⭐⭐⭐⭐ |
| Attention | ⭐⭐⭐⭐⭐ |
| RAG | ⭐⭐⭐⭐⭐ |
| Vector Databases | ⭐⭐⭐⭐⭐ |
| HNSW | ⭐⭐⭐⭐⭐ |
| Embeddings | ⭐⭐⭐⭐⭐ |
| RLHF | ⭐⭐⭐⭐ |
| Fine-Tuning | ⭐⭐⭐⭐ |
| LoRA | ⭐⭐⭐⭐ |
| Quantization | ⭐⭐⭐⭐ |
| Evaluation Metrics | ⭐⭐⭐⭐⭐ |
| Hallucination | ⭐⭐⭐⭐⭐ |
| Agentic AI | ⭐⭐⭐⭐ |
| VLMs | ⭐⭐⭐⭐ |
| Multimodal RAG | ⭐⭐⭐⭐ |



# 📊 RAG Evaluation Metrics

These metrics measure how good your retrieval system is before the LLM generates an answer.

---

# 1. Recall@K

Measures:

```text
Did we retrieve the correct document in the top K results?
```

---

## Formula

```text
Recall@K =
(Number of relevant documents retrieved in Top-K)
/
(Total relevant documents)
```

---

## Example

Ground Truth Relevant Documents:

```text
[D3]
```

Retrieved Top-5:

```text
[D1, D7, D3, D8, D9]
```

Since:

```text
D3 is present
```

Recall@5:

```text
1/1 = 100%
```

---

## Interpretation

| Recall@K | Meaning |
|-----------|----------|
| High | Retriever finds relevant docs |
| Low | Retriever misses relevant docs |

---

# 2. MRR (Mean Reciprocal Rank)

Measures:

```text
How early does the first correct document appear?
```

Amazon loves this metric.

---

## Formula

```text
MRR = Average(1 / Rank)
```

---

## Example 1

Retrieved:

```text
[D3, D7, D8, D9]
```

Correct document:

```text
D3 at Rank 1
```

Score:

```text
1/1 = 1.0
```

---

## Example 2

Retrieved:

```text
[D1, D7, D3, D9]
```

Correct document:

```text
D3 at Rank 3
```

Score:

```text
1/3 = 0.333
```

---

## Interpretation

| MRR | Meaning |
|------|----------|
| 1.0 | Correct doc always first |
| High | Correct docs appear early |
| Low | Users must scroll/search |

---

# 3. nDCG (Normalized Discounted Cumulative Gain)

Measures:

```text
Quality of ranking
```

taking into account:

- relevance
- position

Higher-ranked documents get more credit.

---

## Idea

Relevant documents near the top:

```text
Good
```

Relevant documents near the bottom:

```text
Less useful
```

---

## Example

Retrieved:

```text
Rank 1 → Highly Relevant
Rank 2 → Relevant
Rank 3 → Not Relevant
```

nDCG rewards:

```text
relevant results appearing earlier
```

---

## Range

| Value | Meaning |
|---------|----------|
| 1.0 | Perfect ranking |
| 0 | Poor ranking |

---

# 4. Human Evaluation

Measures:

```text
Would a human consider the answer useful?
```

---

## Evaluators Judge

- Correctness
- Relevance
- Completeness
- Faithfulness
- Hallucination

---

## Example Scale

| Score | Meaning |
|---------|----------|
| 1 | Bad |
| 3 | Acceptable |
| 5 | Excellent |

---

## Why Needed?

Automated metrics often miss:

- reasoning quality
- factual consistency
- user satisfaction

---

# 5. End-Task Accuracy

Measures:

```text
Did the entire system solve the business problem?
```

---

## Example

Customer Support Bot

Question:

```text
How many annual leaves do I get?
```

Expected:

```text
24 leaves
```

System Output:

```text
24 leaves
```

Result:

```text
Correct
```

Accuracy:

```text
Correct Answers / Total Questions
```

---

# Interview Summary

| Metric | Measures |
|----------|-----------|
| Recall@K | Retrieval coverage |
| MRR | Position of first relevant document |
| nDCG | Overall ranking quality |
| Human Evaluation | User-perceived quality |
| End-Task Accuracy | Business success metric |

---

# 🎤 Amazon Interview Answer

> Recall@K measures whether relevant documents are retrieved. MRR measures how early the first relevant result appears. nDCG evaluates the overall ranking quality by rewarding relevant documents at higher positions. Human evaluation measures answer quality judged by people, while end-task accuracy measures whether the complete RAG system successfully solves the user's task.
>
> # 🧠 What are ANN Indexes?

ANN stands for:

```text
Approximate Nearest Neighbor
```

ANN indexes are specialized data structures used to:

```text
find similar vectors very quickly
```

without comparing against every vector.

---

# Why Do We Need ANN?

Suppose you have:

```text
10 million embeddings
```

and a query vector arrives.

---

## Brute Force Search

Compare query with:

```text
all 10 million vectors
```

Complexity:

```text
O(N)
```

Very slow.

---

# ANN Idea

Instead of checking every vector:

```text
Search only promising regions
```

This gives:

```text
Much faster retrieval
```

with:

```text
~95-99% accuracy
```

instead of exact 100%.

---

# Example

Suppose query is:

```text
"How do I apply for leave?"
```

Embedding:

```text
[0.12, 0.45, 0.89, ...]
```

ANN index quickly finds:

```text
HR Policy
Leave Policy
Vacation Rules
```

without scanning millions of vectors.

---

# Popular ANN Index Types

| Index | Idea |
|---------|---------|
| HNSW | Graph-based search |
| IVF | Cluster-based search |
| PQ | Compressed vectors |
| IVF-PQ | Clustering + compression |
| ScaNN | Google's ANN search |
| DiskANN | Microsoft large-scale search |

---

# 1. HNSW (Most Popular)

```text
Hierarchical Navigable Small World
```

Builds:

```text
Vector → Neighbor Graph
```

Search:

```text
Graph Traversal
```

instead of:

```text
Checking all vectors
```

---

# 2. IVF

```text
Inverted File Index
```

Idea:

```text
Cluster vectors first
```

Example:

```text
Cluster 1 → Finance
Cluster 2 → HR
Cluster 3 → Legal
```

Query:

```text
Search only nearest cluster
```

---

# 3. Product Quantization (PQ)

Idea:

```text
Compress vectors
```

Example:

```text
1536 dimensions
```

stored in:

```text
much smaller memory footprint
```

Useful for:

```text
billions of vectors
```

---

# ANN Tradeoff

| Method | Speed | Accuracy |
|----------|---------|----------|
| Exact Search | Slow | 100% |
| ANN Search | Fast | ~95-99% |

---

# In RAG Systems

Pipeline:

```text
User Query
    ↓
Embedding Model
    ↓
ANN Index
    ↓
Top-K Documents
    ↓
LLM
```

Without ANN:

```text
Retrieval latency becomes too high
```

---

# Amazon Interview Answer

> ANN (Approximate Nearest Neighbor) indexes are data structures used to efficiently retrieve similar embeddings from large vector databases. Instead of comparing a query against every vector, ANN methods such as HNSW and IVF search only a subset of the vector space, providing much faster retrieval with a small loss in accuracy. They are essential for scalable RAG systems handling millions or billions of embeddings.

# 🧠 Deep Dive into ANN Search Algorithms

When Amazon asks about ANN, they usually want:

```text
How does HNSW work?
How does IVF work?
How does PQ work?
What are the tradeoffs?
```

---

# 1. Brute Force Search (Baseline)

Suppose:

```text
1 Million Vectors
Dimension = 1536
```

Query:

```text
"How many leaves do employees get?"
```

Convert to embedding:

```text
Q = [0.12, 0.43, ...]
```

Now compare Q against:

```text
Vector1
Vector2
Vector3
...
Vector1000000
```

using cosine similarity.

---

## Complexity

:contentReference[oaicite:0]{index=0}

For 1 million vectors:

```text
1 million comparisons
```

Too slow.

---

# 2. HNSW (Hierarchical Navigable Small World)

Most commonly used ANN index.

Used by:
- Pinecone
- Weaviate
- OpenSearch
- Milvus

---

## Core Idea

Build a graph.

Instead of:

```text
Vector → Database
```

Store:

```text
Vector → Neighbor Vectors
```

---

# Example

Suppose vectors represent:

```text
Car
Vehicle
Automobile
Bike
Pizza
Football
```

Graph:

```text
Car ───── Vehicle
 │          │
 │          │
Automobile  Bike

Pizza ─── Football
```

Similar vectors become neighbors.

---

## Search Process

Query:

```text
"sedan car"
```

Start from some node:

```text
Pizza
```

Move to:

```text
Vehicle
```

Move to:

```text
Car
```

Move to:

```text
Automobile
```

Keep moving toward higher similarity.

---

## Why Fast?

Instead of:

```text
1 million comparisons
```

Maybe:

```text
100 graph hops
```

---

## Multi-Layer Structure

HNSW actually builds:

```text
Level 3
Level 2
Level 1
Level 0
```

---

### Top Layer

Very few nodes.

```text
A ------ B

      C
```

Fast navigation.

---

### Lower Layers

More nodes.

```text
A--B--C--D--E--F
```

---

### Bottom Layer

Contains all vectors.

```text
Millions of vectors
```

---

## Search

```text
Top Layer
    ↓
Middle Layer
    ↓
Bottom Layer
```

Like GPS:

```text
Country
  ↓
City
  ↓
Street
```

---

## Complexity

Approximately:

```text
O(log N)
```

instead of:

```text
O(N)
```

---

## Advantages

✅ Very fast

✅ High recall

✅ Industry standard

---

## Disadvantages

❌ More memory

❌ Slow index construction

---

# 3. IVF (Inverted File Index)

IVF uses clustering.

---

## Idea

Instead of storing:

```text
1 million vectors
```

Create clusters.

---

# Example

Suppose documents belong to:

```text
Finance
HR
Legal
Sports
```

Clusters:

```text
Cluster 1 → Finance

Cluster 2 → HR

Cluster 3 → Legal

Cluster 4 → Sports
```

---

## Query

User asks:

```text
How many leaves do employees get?
```

Embedding belongs near:

```text
HR Cluster
```

---

## Search

Instead of:

```text
Search all clusters
```

Search only:

```text
HR Cluster
```

---

## Complexity

Much smaller search space.

---

# Visual

Instead of:

```text
1M vectors
```

Search:

```text
Cluster #24

contains

5000 vectors
```

Only search those.

---

## Advantages

✅ Faster than brute force

✅ Easy implementation

---

## Disadvantages

❌ Wrong cluster → miss answer

❌ Lower recall than HNSW

---

# 4. PQ (Product Quantization)

Problem:

```text
1 Billion vectors
```

Huge memory.

---

# Example

Embedding:

```text
1536 dimensions
```

Store normally:

```text
1536 floating numbers
```

Huge storage.

---

## PQ Idea

Split vector.

Example:

```text
1536 dimensions

→ 16 chunks

→ 96 dimensions each
```

---

Instead of storing:

```text
actual values
```

Store:

```text
cluster IDs
```

---

Example:

```text
Chunk 1 → Code 17
Chunk 2 → Code 82
Chunk 3 → Code 4
```

---

This compresses:

```text
1536 dimensions
```

into:

```text
a few bytes
```

---

## Advantages

✅ Massive memory savings

✅ Billion-scale retrieval

---

## Disadvantages

❌ Slight accuracy loss

---

# 5. IVF + PQ

Most common production setup.

---

## Pipeline

```text
Vectors
   ↓
Cluster (IVF)
   ↓
Compress (PQ)
   ↓
Store
```

---

## Search

```text
Query
   ↓
Find Cluster
   ↓
Search Compressed Vectors
   ↓
Return Top-K
```

---

# Example

Without IVF-PQ

```text
1 Billion vectors
```

Need:

```text
~6 TB RAM
```

---

With IVF-PQ

```text
~200 GB
```

Possible.

---

# 6. ScaNN (Google)

Used internally by Google.

Idea:

```text
Smart clustering
+
Quantization
+
Re-ranking
```

Optimized for TPUs.

---

# 7. DiskANN (Microsoft)

Problem:

```text
Dataset too large for RAM
```

Store index on SSD.

---

Allows:

```text
Billions of vectors
```

with:

```text
low memory usage
```

---

# Interview Comparison Table

| Method | Idea | Speed | Accuracy | Memory |
|----------|----------|----------|----------|----------|
| Brute Force | Compare all vectors | Slow | ⭐⭐⭐⭐⭐ | High |
| IVF | Search nearest cluster | Fast | ⭐⭐⭐ | Medium |
| PQ | Compress vectors | Fast | ⭐⭐⭐ | Very Low |
| IVF-PQ | Cluster + Compress | Very Fast | ⭐⭐⭐⭐ | Low |
| HNSW | Graph traversal | Very Fast | ⭐⭐⭐⭐⭐ | High |
| ScaNN | Google optimized ANN | Very Fast | ⭐⭐⭐⭐ | Medium |
| DiskANN | SSD-based ANN | Very Fast | ⭐⭐⭐⭐ | Very Low |

---

# What Amazon Usually Likes to Hear

For:

```text
Enterprise RAG
10M–100M documents
```

Recommended:

```text
HNSW
```

because:

- high recall
- excellent latency
- production proven

For:

```text
Billions of vectors
```

Recommended:

```text
IVF-PQ
or
DiskANN
```

because memory becomes the bottleneck.

---

# 🎤 Interview Answer

> "HNSW uses a hierarchical graph where vectors are connected to their nearest neighbors, allowing logarithmic-time graph traversal instead of linear scanning. IVF partitions vectors into clusters and searches only the most relevant clusters. PQ compresses vectors into compact codes to reduce memory usage. HNSW typically provides the best recall and latency for enterprise RAG, while IVF-PQ is preferred when datasets grow to billions of embeddings."
