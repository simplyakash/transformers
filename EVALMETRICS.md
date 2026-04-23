📊 Evaluation Metrics for LLMs & VLMs

## 🧠 1. Language Modeling Metrics (LLMs)

## 🔹 Perplexity (PPL)
Measures how well a model predicts text
Lower = better

Range: 1→∞ (lower is better)  

👉 Modern strong LLMs often get <20 on standard benchmarks

| PPL    | Interpretation                      |
| ------ | ----------------------------------- |
| ~1–10  | Excellent (near-perfect prediction) |
| 10–30  | Good                                |
| 30–100 | Weak                                |
| >100   | Poor                                |


👉 Core idea:

“How surprised is the model by the data?”

### Formula
$$
\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log p(x_t)\right)
$$

### Steps
1. For each token \(x_t\), get probability \(p(x_t)\)  
2. Take log  
3. Average over all tokens  
4. Apply exponential  

👉 Lower = better


## 🔹 Cross-Entropy Loss

Range: 0→∞ (lower is better)

Training + evaluation metric
Token-level probability accuracy

| Loss | Interpretation |
| ---- | -------------- |
| <1.0 | Excellent      |
| 1–2  | Good           |
| 2–4  | Weak           |
| >4   | Poor           |

### Formula
$$
L = -\sum_{t=1}^{T} y_t \log(p_t)
$$

### Steps
1. Compute softmax probabilities  
2. Multiply with one-hot labels  
3. Take log  
4. Sum over tokens  

👉 Equivalent to:

L = -∑ log(p_correct)


## 🔹 Accuracy / Top-k Accuracy

Used in classification tasks
Exact match between prediction and ground truth



## 🔹 F1  / Precision / Recall
Used for tasks like:
Named Entity Recognition
QA span prediction

## 🔹 F1 Score (token overlap)
Partial correctness

| Score  | Interpretation |
| ------ | -------------- |
| >85%   | Strong         |
| 70–85% | Good           |
| 50–70% | Weak           |
| <50%   | Poor           |

### Formula
$$
F1 = \frac{2PR}{P + R}
$$

### Steps
1. Tokenize prediction & ground truth  
2. Compute:
   - Precision = overlap / predicted  
   - Recall = overlap / ground truth  
3. Combine  


✍️ 2. Text Generation Metrics

## 🔹 BLEU (Bilingual Evaluation Understudy)
Measures n-gram overlap with reference
Common in machine translation

🔹 BLEU (0–1 or 0–100)

| Score     | Interpretation |
| --------- | -------------- |
| >0.5 (50) | Strong         |
| 0.3–0.5   | Decent         |
| 0.1–0.3   | Weak           |
| <0.1      | Poor           |


⚠️ Weakness:

Doesn’t capture meaning well

### Steps
1. Extract n-grams (1–4 grams)  
2. Count matches with reference  
3. Compute precision:
   $$
   p_n = \frac{\text{matched n-grams}}{\text{total predicted n-grams}}
   $$
4. Take geometric mean  
5. Apply brevity penalty  

### Final
$$
\text{BLEU} = BP \cdot \exp\left(\sum w_n \log p_n\right)
$$


## 🔹 ROUGE
Recall-based overlap metric
Used in summarization

Variants:

ROUGE-1 (unigrams)
ROUGE-2 (bigrams)
ROUGE-L (longest sequence)

| Score   | Interpretation |
| ------- | -------------- |
| >0.5    | Strong         |
| 0.3–0.5 | Good           |
| 0.2–0.3 | Weak           |
| <0.2    | Poor           |

### ROUGE-N (Recall)
$$
\text{ROUGE} = \frac{\text{overlapping n-grams}}{\text{total n-grams in reference}}
$$

### ROUGE-L
- Based on Longest Common Subsequence (LCS)

---



---

## 🧪 METEOR

Considers synonyms + stemming
Better than BLEU for semantics
| Score    | Interpretation |
| -------- | -------------- |
| >0.4     | Strong         |
| 0.25–0.4 | Good           |
| 0.1–0.25 | Weak           |
| <0.1     | Poor           |

### Steps
1. Align words (exact, stem, synonym)  
2. Compute Precision (P) and Recall (R)  
3. Combine:
$$
F = \frac{10PR}{R + 9P}
$$
4. Apply fragmentation penalty  

🔹 CIDEr
Designed for image captioning
Weights important words more
### Steps
1. Extract n-grams  
2. Compute TF-IDF weights  
3. Compare using cosine similarity  
4. Average across references  

🔹 SPICE
Compares semantic scene graphs
Focuses on meaning, not wording

🤖 3. Semantic / Embedding-Based Metrics

🔹 BERTScore
Uses contextual embeddings (like BERT)
Measures semantic similarity

👉 Much better for meaning vs surface overlap

Range: ~0.7 → 1.0 (higher is better)

| Score     | Interpretation |
| --------- | -------------- |
| >0.95     | Excellent      |
| 0.90–0.95 | Strong         |
| 0.85–0.90 | Decent         |
| <0.85     | Weak           |

### Steps
1. Convert tokens to embeddings  
2. Match each token to most similar token  
3. Compute cosine similarity  
4. Aggregate into Precision / Recall / F1  

🔹 MoverScore / Sentence Similarity
Distance between embeddings
Captures paraphrases

🧪 4. QA & Reasoning Metrics (LLMs)

🔹 Exact Match (EM)
Strict string match

| Score  | Interpretation |
| ------ | -------------- |
| >80%   | Strong         |
| 60–80% | Good           |
| 40–60% | Weak           |
| <40%   | Poor           |

### Formula
$$
EM =
\begin{cases}
1 & \text{if exact match} \\
0 & \text{otherwise}
\end{cases}
$$

### Steps
1. Normalize text  
2. Compare strings 



## 🔹 Pass@k
Used in code generation
“Did any of k samples get it right?”

| Score  | Interpretation |
| ------ | -------------- |
| >80%   | Strong         |
| 50–80% | Good           |
| 20–50% | Weak           |
| <20%   | Poor           |


🔹 Chain-of-Thought Evaluation
Checks reasoning steps (often human-evaluated)

🖼️ 5. Vision-Language Model (VLM) Metrics

Used in models like CLIP

## 🔹 Image-Text Retrieval Metrics
Recall@K (R@1, R@5, R@10)
Mean Rank / Median Rank

👉 Measures:

Can the model retrieve correct image/text?

| Metric | Strong  | Weak |
| ------ | ------- | ---- |
| R@1    | >40–60% | <20% |
| R@5    | >70–90% | <50% |
| R@10   | >85–95% | <60% |


## 🔹 Contrastive Accuracy
Matching vs non-matching pairs

## 🔹 Captioning Metrics (VLMs)

Same as text generation:

BLEU
ROUGE
CIDEr
SPICE

## 🔹 Visual Question Answering (VQA) Accuracy
Open-ended answer matching
Often uses soft scoring
📏 6. Alignment & Human Preference Metrics

## 🔹 Human Evaluation
Fluency
Coherence
Helpfulness
Safety

## 🔹 Win Rate / Pairwise Comparison
Compare outputs from two models

## 🔹 Reward Model Score
Used in RLHF pipelines

## 🔹 Truthfulness / Hallucination Metrics
Fact-checking benchmarks
Faithfulness scores
⚖️ 7. Robustness & Safety Metrics
Toxicity scores
Bias evaluation
Adversarial robustness
🧩 Big Picture
Category	What it measures
Perplexity	Prediction quality
BLEU/ROUGE	Surface overlap
BERTScore	Semantic similarity
Recall@K	Retrieval ability
EM/F1	Exact correctness
Human eval	Real-world usefulness
⚡ Key Insight

No single metric is enough.

👉 Modern evaluation combines:

Automatic metrics (fast, scalable)
Human judgment (accurate, nuanced)
