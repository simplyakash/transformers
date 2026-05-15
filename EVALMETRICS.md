# 📊 Evaluation Metrics for LLMs & VLMs

---

# 🧠 1. Language Modeling Metrics (LLMs)

---

# 🔹 Perplexity (PPL)

Measures how well a model predicts text.

Lower = better

Range:

```text
1 → ∞  (lower is better)
```

👉 Modern strong LLMs often get:

```text
PPL < 20
```

on standard benchmarks.

---

# 📊 Perplexity Interpretation

| PPL | Interpretation |
|---|---|
| ~1–10 | Excellent (near-perfect prediction) |
| 10–30 | Good |
| 30–100 | Weak |
| >100 | Poor |

---

# 🧠 Core Idea

```text
“How surprised is the model by the data?”
```

---

# 📌 Formula

```text
PPL = exp(-(1/T) ∑ₜ₌₁ᵀ log p(xₜ))
``` 
---

# 📌 Steps

1. For each token xₜ, get probability p(xₜ)
2. Take log
3. Average over all tokens
4. Apply exponential

👉 Lower = better

---

# 🔹 Cross-Entropy Loss

Range:

```text
0 → ∞  (lower is better)
```

Used for:

- training
- evaluation
- token probability accuracy

---

# 📊 Cross-Entropy Interpretation

| Loss | Interpretation |
|---|---|
| <1.0 | Excellent |
| 1–2 | Good |
| 2–4 | Weak |
| >4 | Poor |

---

# 📌 Formula

```text
L = -∑ₜ₌₁ᵀ yₜ log(pₜ)
```


---

# 📌 Steps

1. Compute softmax probabilities
2. Multiply with one-hot labels
3. Take log
4. Sum over tokens

Equivalent to:

```text
L = -∑ log(p_correct)
```

---

# 🔹 Accuracy / Top-k Accuracy

Used in classification tasks.

Measures:

```text
Exact match between prediction and ground truth
```

---

# 🔹 F1 / Precision / Recall

Used for tasks like:

- Named Entity Recognition
- QA span prediction

---

# 🔹 F1 Score (Token Overlap)

Measures partial correctness.

---

# 📊 F1 Interpretation

| Score | Interpretation |
|---|---|
| >85% | Strong |
| 70–85% | Good |
| 50–70% | Weak |
| <50% | Poor |

---

# 📌 Formula

```text
F1 = 2PR / (P + R)
```

---

# 📌 Steps

1. Tokenize prediction & ground truth
2. Compute:
   - Precision = overlap / predicted
   - Recall = overlap / ground truth
3. Combine

---

# ✍️ 2. Text Generation Metrics

---

# 🔹 BLEU (Bilingual Evaluation Understudy)

Measures n-gram overlap with reference.

Common in machine translation.

---

# 📊 BLEU Interpretation

| Score | Interpretation |
|---|---|
| >0.5 (50) | Strong |
| 0.3–0.5 | Decent |
| 0.1–0.3 | Weak |
| <0.1 | Poor |

---

# ⚠️ Weakness

```text
Doesn’t capture meaning well
```

---

# 📌 Steps

1. Extract n-grams (1–4 grams)
2. Count matches with reference
3. Compute precision:

:contentReference[oaicite:3]{index=3}

4. Take geometric mean
5. Apply brevity penalty

---

# 📌 Final BLEU Formula

```text
BLEU = BP · exp(∑ wₙ log pₙ)
```

---

# 🔹 ROUGE

Recall-based overlap metric.

Used in summarization.

---

# 📌 Variants

- ROUGE-1 → unigrams
- ROUGE-2 → bigrams
- ROUGE-L → longest sequence

---

# 📊 ROUGE Interpretation

| Score | Interpretation |
|---|---|
| >0.5 | Strong |
| 0.3–0.5 | Good |
| 0.2–0.3 | Weak |
| <0.2 | Poor |

---

# 📌 ROUGE-N Formula

```text
ROUGE = overlapping n-grams / total n-grams in reference
```


---

# 📌 ROUGE-L

Based on:

```text
Longest Common Subsequence (LCS)
```

---

# 🧪 METEOR

Considers:

- synonyms
- stemming

Better semantic understanding than BLEU.

---

# 📊 METEOR Interpretation

| Score | Interpretation |
|---|---|
| >0.4 | Strong |
| 0.25–0.4 | Good |
| 0.1–0.25 | Weak |
| <0.1 | Poor |

---

# 📌 Steps

1. Align words:
   - exact
   - stem
   - synonym
2. Compute Precision (P) and Recall (R)
3. Combine:

```text
F = 10PR / (R + 9P)
```

4. Apply fragmentation penalty

---

# 🔹 CIDEr

Designed for image captioning.

Weights important words more.

---

# 📌 Steps

1. Extract n-grams
2. Compute TF-IDF weights
3. Compare using cosine similarity
4. Average across references

---

# 🔹 SPICE

Compares semantic scene graphs.

Focuses on:

```text
meaning, not wording
```

---

# 🤖 3. Semantic / Embedding-Based Metrics

---

# 🔹 BERTScore

Uses contextual embeddings (like BERT).

Measures semantic similarity.

---

# 🧠 Key Idea

```text
Much better for meaning vs surface overlap
```

---

# 📊 BERTScore Range

```text
~0.7 → 1.0  (higher is better)
```

---

# 📊 BERTScore Interpretation

| Score | Interpretation |
|---|---|
| >0.95 | Excellent |
| 0.90–0.95 | Strong |
| 0.85–0.90 | Decent |
| <0.85 | Weak |

---

# 📌 Steps

1. Convert tokens to embeddings
2. Match each token to most similar token
3. Compute cosine similarity
4. Aggregate into Precision / Recall / F1

---

# 🔹 MoverScore / Sentence Similarity

Distance between embeddings.

Captures:

```text
paraphrases
```

---

# 🧪 4. QA & Reasoning Metrics (LLMs)

---

# 🔹 Exact Match (EM)

Strict string match.

---

# 📊 EM Interpretation

| Score | Interpretation |
|---|---|
| >80% | Strong |
| 60–80% | Good |
| 40–60% | Weak |
| <40% | Poor |

---

# 📌 Formula

```text
EM =
1   if exact match
0   otherwise
```
---

# 📌 Steps

1. Normalize text
2. Compare strings

---

# 🔹 Pass@k

Used in code generation.

Meaning:

```text
“Did any of k samples get it right?”
```

---

# 📊 Pass@k Interpretation

| Score | Interpretation |
|---|---|
| >80% | Strong |
| 50–80% | Good |
| 20–50% | Weak |
| <20% | Poor |

---

# 🔹 Chain-of-Thought Evaluation

Checks reasoning steps.

Often:

```text
human evaluated
```

---

# 🖼️ 5. Vision-Language Model (VLM) Metrics

Used in models like:

- CLIP
- Flamingo
- LLaVA

---

# 🔹 Image-Text Retrieval Metrics

Metrics:

- Recall@K (R@1, R@5, R@10)
- Mean Rank
- Median Rank

Measures:

```text
Can the model retrieve correct image/text?
```

---

# 📊 Retrieval Interpretation

| Metric | Strong | Weak |
|---|---|---|
| R@1 | >40–60% | <20% |
| R@5 | >70–90% | <50% |
| R@10 | >85–95% | <60% |

---

# 🔹 Contrastive Accuracy

Measures:

```text
matching vs non-matching pairs
```

---

# 🔹 Captioning Metrics (VLMs)

Same as text generation metrics:

- BLEU
- ROUGE
- CIDEr
- SPICE

---

# 🔹 Visual Question Answering (VQA) Accuracy

Open-ended answer matching.

Often uses:

```text
soft scoring
```

---

# 📏 6. Alignment & Human Preference Metrics

---

# 🔹 Human Evaluation

Humans rate:

- Fluency
- Coherence
- Helpfulness
- Safety

---

# 🔹 Win Rate / Pairwise Comparison

Compare outputs from:

```text
two models
```

---

# 🔹 Reward Model Score

Used in:

```text
RLHF pipelines
```

---

# 🔹 Truthfulness / Hallucination Metrics

Measures:

- factual accuracy
- hallucination rate
- faithfulness

---

# ⚖️ 7. Robustness & Safety Metrics

Measures:

- toxicity
- bias
- adversarial robustness

---

# 🧩 Big Picture

| Category | What it measures |
|---|---|
| Perplexity | Prediction quality |
| BLEU / ROUGE | Surface overlap |
| BERTScore | Semantic similarity |
| Recall@K | Retrieval ability |
| EM / F1 | Exact correctness |
| Human eval | Real-world usefulness |

---

# ⚡ Key Insight

No single metric is enough.

Modern evaluation combines:

- automatic metrics
- human judgment

because both are necessary for reliable evaluation.
