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



# Batch Normalization vs Layer Normalization

The easiest way to understand the difference is to see **which values are used to calculate the mean and variance**.

---

# Example Input

Suppose a neural network layer produces the following output:

| Sample | Feature 1 | Feature 2 | Feature 3 |
|----------|----------|----------|----------|
| A | 2 | 4 | 6 |
| B | 8 | 10 | 12 |

Shape:

```text
(batch_size = 2, features = 3)
```

```text
[
 [ 2,  4,  6],
 [ 8, 10, 12]
]
```

---

# 1. Batch Normalization (BatchNorm)

BatchNorm normalizes **each feature across all samples in the batch**.

## Feature 1

Values:

```text
[2, 8]
```

Mean:

μ = (2 + 8) / 2 = 5

Variance:

σ² = ((2 - 5)² + (8 - 5)²) / 2
   = (9 + 9) / 2
   = 9

Standard Deviation:

σ = 3

Normalized Values:

```text
(2 - 5) / 3 = -1
(8 - 5) / 3 =  1
```

Result:

```text
[-1, 1]
```

---

## Feature 2

Values:

```text
[4, 10]
```

Mean = 7

Std = 3

Normalized:

```text
[-1, 1]
```

---

## Feature 3

Values:

```text
[6, 12]
```

Mean = 9

Std = 3

Normalized:

```text
[-1, 1]
```

---

## Final BatchNorm Output

| Sample | F1 | F2 | F3 |
|----------|----------|----------|----------|
| A | -1 | -1 | -1 |
| B | 1 | 1 | 1 |

---

## Visualization

BatchNorm looks **vertically** across the batch:

```text
Feature1  [2, 8]
Feature2  [4,10]
Feature3  [6,12]
           ↑
     normalize here
```

BatchNorm asks:

> "For Feature 1, how does each sample compare to the other samples in the batch?"

---

# 2. Layer Normalization (LayerNorm)

LayerNorm normalizes **all features inside a single sample**.

---

## Sample A

Values:

```text
[2, 4, 6]
```

Mean:

μ = (2 + 4 + 6) / 3
  = 4

Variance:

σ² = ((2 - 4)² + (4 - 4)² + (6 - 4)²) / 3
   = (4 + 0 + 4) / 3
   = 8/3

Standard Deviation:

σ ≈ 1.633

Normalized:

```text
(2 - 4) / 1.633 ≈ -1.225
(4 - 4) / 1.633 = 0
(6 - 4) / 1.633 ≈ 1.225
```

Result:

```text
[-1.225, 0, 1.225]
```

---

## Sample B

Values:

```text
[8, 10, 12]
```

Mean = 10

Std ≈ 1.633

Normalized:

```text
[-1.225, 0, 1.225]
```

---

## Final LayerNorm Output

| Sample | F1 | F2 | F3 |
|----------|----------|----------|----------|
| A | -1.225 | 0 | 1.225 |
| B | -1.225 | 0 | 1.225 |

---

## Visualization

LayerNorm looks **horizontally** across features:

```text
Sample A [2, 4, 6]
          ↑
 normalize here

Sample B [8,10,12]
          ↑
 normalize here
```

LayerNorm asks:

> "Within this sample, how does each feature compare to the other features?"

---

# Mathematical Difference

## BatchNorm

For feature j:

μⱼ = (1/m) Σ xᵢⱼ

σ²ⱼ = (1/m) Σ (xᵢⱼ − μⱼ)²

Where:

- m = batch size
- j = feature index

Normalization:

x̂ᵢⱼ = (xᵢⱼ − μⱼ) / √(σ²ⱼ + ε)

Statistics are computed using **other samples in the batch**.

---

## LayerNorm

For sample i:

μᵢ = (1/d) Σ xᵢⱼ

σ²ᵢ = (1/d) Σ (xᵢⱼ − μᵢ)²

Where:

- d = number of features
- i = sample index

Normalization:

x̂ᵢⱼ = (xᵢⱼ − μᵢ) / √(σ²ᵢ + ε)

Statistics are computed using **features from the same sample**.

---

# Why CNNs Use BatchNorm

CNNs usually train with:

```text
Batch Size = 32
Batch Size = 64
Batch Size = 128
```

Large batches provide reliable statistics.

Benefits:

✓ Faster convergence

✓ More stable gradients

✓ Acts as a regularizer

✓ Works very well in computer vision

---

# Why Transformers Use LayerNorm

Consider a sentence:

```text
"I love AI"
```

During inference:

```text
Batch Size = 1
```

During training:

```text
Batch Size = 8
Batch Size = 32
Batch Size = 128
```

If BatchNorm were used:

- Statistics change when batch size changes
- Output depends on other examples in the batch
- Difficult for variable-length sequences

LayerNorm solves this because:

- Every token is normalized independently
- Works even when batch size = 1
- Stable for NLP and transformers

---

# Transformer Example

Suppose a token embedding is:

```text
[1.2, 0.5, -0.8, 2.1]
```

LayerNorm computes:

```text
mean(token)
variance(token)
```

using only these 4 numbers.

It does not care what other tokens or other sentences are doing.

This makes it ideal for transformers.

---

# Learnable Parameters

Both BatchNorm and LayerNorm contain learnable parameters:

γ (gamma) → scale

β (beta) → shift

Final output:

x̂ = γ × normalized(x) + β

This allows the network to learn the best scaling after normalization.

---

# Interview Summary

| Property | BatchNorm | LayerNorm |
|-----------|-----------|-----------|
| Statistics Computed Across | Batch | Features |
| Depends On Other Samples | Yes | No |
| Works With Batch Size = 1 | Poorly | Yes |
| Commonly Used In | CNNs | Transformers |
| Training/Inference Behavior | Different | Same |
| NLP Friendly | No | Yes |
| Vision Friendly | Excellent | Good |

---

# Memory Trick

```text
BatchNorm
    ↓
Normalize DOWN the batch
(Vertical)

LayerNorm
    →
Normalize ACROSS features
(Horizontal)
```

---

# One-Line Interview Answer

BatchNorm normalizes each feature using statistics from all samples in a batch, while LayerNorm normalizes each sample using statistics from all features within that sample.


# BatchNorm vs LayerNorm Using a Real Neural Network Example

Suppose we have a neural network:

```text
Input Layer
    ↓
Hidden Layer 1 (7 neurons)
    ↓
Hidden Layer 2 (5 neurons)
    ↓
Output
```

Let's say we process a batch of 3 samples.

---

# Hidden Layer 1 Output (7 Neurons)

After the first linear layer, suppose we get:

| Sample | N1 | N2 | N3 | N4 | N5 | N6 | N7 |
|----------|----|----|----|----|----|----|----|
| S1 | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
| S2 | 15 | 25 | 35 | 45 | 55 | 65 | 75 |
| S3 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |

Shape:

```text
(batch_size = 3, hidden_dim = 7)
```

```text
[
 [10,20,30,40,50,60,70],
 [15,25,35,45,55,65,75],
 [20,30,40,50,60,70,80]
]
```

---

# How BatchNorm Works

BatchNorm looks at one neuron at a time.

For Neuron N1:

```text
[10,15,20]
```

These values came from:

```text
S1 → N1 = 10
S2 → N1 = 15
S3 → N1 = 20
```

Mean:

```text
μ = (10 + 15 + 20)/3
  = 15
```

Std:

```text
σ ≈ 4.08
```

Normalized:

```text
(10-15)/4.08 ≈ -1.22
(15-15)/4.08 = 0
(20-15)/4.08 ≈ 1.22
```

Result:

```text
[-1.22, 0, 1.22]
```

---

Now Neuron N2:

```text
[20,25,30]
```

Mean:

```text
25
```

Std:

```text
4.08
```

Normalized:

```text
[-1.22,0,1.22]
```

---

BatchNorm repeats this for:

```text
N1
N2
N3
N4
N5
N6
N7
```

independently.

---

# Visualization of BatchNorm

```text
        N1   N2   N3   N4   N5   N6   N7
S1      10   20   30   40   50   60   70
S2      15   25   35   45   55   65   75
S3      20   30   40   50   60   70   80

        ↑
        │
Normalize column-wise
```

BatchNorm asks:

> "For neuron N1, what is the mean and variance across all samples?"

---

# How LayerNorm Works

LayerNorm looks at one sample at a time.

Take Sample S1:

```text
[10,20,30,40,50,60,70]
```

Mean:

```text
μ = (10+20+30+40+50+60+70)/7
  = 40
```

Variance:

```text
σ² = 400
```

Std:

```text
σ = 20
```

Normalized:

```text
(10-40)/20 = -1.5
(20-40)/20 = -1.0
(30-40)/20 = -0.5
(40-40)/20 = 0
(50-40)/20 = 0.5
(60-40)/20 = 1.0
(70-40)/20 = 1.5
```

Result:

```text
[-1.5,-1,-0.5,0,0.5,1,1.5]
```

---

For Sample S2:

```text
[15,25,35,45,55,65,75]
```

Mean:

```text
45
```

Std:

```text
20
```

Normalized:

```text
[-1.5,-1,-0.5,0,0.5,1,1.5]
```

---

For Sample S3:

```text
[20,30,40,50,60,70,80]
```

Mean:

```text
50
```

Std:

```text
20
```

Normalized:

```text
[-1.5,-1,-0.5,0,0.5,1,1.5]
```

---

# Visualization of LayerNorm

```text
        N1   N2   N3   N4   N5   N6   N7
S1      10   20   30   40   50   60   70
         ←──── Normalize ────→

S2      15   25   35   45   55   65   75
         ←──── Normalize ────→

S3      20   30   40   50   60   70   80
         ←──── Normalize ────→
```

LayerNorm asks:

> "Inside this sample, how do the 7 neurons compare to each other?"

---

# Now Hidden Layer 2 (5 Neurons)

Suppose after the next layer we get:

```text
[
 [2, 4, 6, 8,10],
 [3, 5, 7, 9,11],
 [4, 6, 8,10,12]
]
```

Shape:

```text
(batch_size = 3, hidden_dim = 5)
```

Exactly the same rules apply.

---

## BatchNorm

Normalize each column:

```text
Neuron1 → [2,3,4]
Neuron2 → [4,5,6]
Neuron3 → [6,7,8]
Neuron4 → [8,9,10]
Neuron5 → [10,11,12]
```

Column-wise normalization.

---

## LayerNorm

Normalize each row:

```text
Sample1 → [2,4,6,8,10]
Sample2 → [3,5,7,9,11]
Sample3 → [4,6,8,10,12]
```

Row-wise normalization.

---

# Why Transformers Prefer LayerNorm

Consider a transformer token embedding:

```text
Token "cat"

[0.12, 1.45, -0.88, 2.11, 0.43, ...]
```

For GPT-like models:

```text
hidden_size = 4096
```

So one token may look like:

```text
[4096 numbers]
```

LayerNorm computes:

```text
mean(4096 numbers)
variance(4096 numbers)
```

for that token only.

It does not need:

- Other sentences
- Other tokens
- Large batches

Therefore it works perfectly during inference when:

```text
batch_size = 1
```

which is exactly how ChatGPT-like models generate text.

---

# Intuition

Imagine a classroom.

### BatchNorm

Compare students by subject.

```text
Math Scores

A = 60
B = 80
C = 90
```

Mean is computed across students.

Question:

"How good is student A compared to other students in Math?"

---

### LayerNorm

Compare subjects within one student.

```text
Student A

Math    = 60
Science = 80
English = 90
History = 70
```

Mean is computed across subjects.

Question:

"Which subjects are strong or weak for this student?"

---

# Ultimate Memory Trick

```text
BatchNorm
─────────
Same Neuron
Across Many Samples

Column-wise
Vertical

        ↓
        ↓
        ↓


LayerNorm
─────────
Same Sample
Across Many Neurons

Row-wise
Horizontal

←────────────→
```

**BatchNorm = Normalize columns**

**LayerNorm = Normalize rows**


# 🎯 Precision@K vs Recall@K

Both **Precision@K** and **Recall@K** are ranking metrics used in:

- Recommendation Systems
- Search Engines
- Information Retrieval
- LLM Retrieval (RAG)
- Content Ranking
- Ads Ranking

Although they sound similar, they answer **different questions**.

---

# 📌 Intuition

Imagine Netflix recommends **10 movies** to a user.

Some of them are actually relevant to the user, and some are not.

Now we ask two different questions.

### Precision@K asks

> **Out of the Top K items I recommended, how many were actually relevant?**

### Recall@K asks

> **Out of ALL the relevant items that exist, how many did I successfully recommend in the Top K?**

---

# 📊 Example

Suppose there are **20 movies** in the database.

The user actually likes **8** of them.

Those relevant movies are

```text
A B C D E F G H
```

The recommender returns the **Top 5**

```text
A C X Y Z
```

where

- A ✅ Relevant
- C ✅ Relevant
- X ❌ Not Relevant
- Y ❌ Not Relevant
- Z ❌ Not Relevant

---

# Step 1: Count Relevant Recommendations

Relevant recommendations returned

```text
A
C
```

Total relevant returned = **2**

---

# 📌 Precision@5

Formula

$\text{Precision@K}=\frac{\text{Relevant items in Top K}}{K}$

Here

Relevant returned = **2**

K = **5**

Therefore

$\text{Precision@5}=\frac{2}{5}=0.40$

or

```text
40%
```

Meaning

> **40% of the recommendations shown to the user were relevant.**

---

# 📌 Recall@5

Formula

$\text{Recall@K}=\frac{\text{Relevant items in Top K}}{\text{Total Relevant Items}}$

Here

Relevant returned = **2**

Total relevant items = **8**

Therefore

$\text{Recall@5}=\frac{2}{8}=0.25$

or

```text
25%
```

Meaning

> **The system found only 25% of all the items the user would have liked.**

---

# 📊 Visual Explanation

Actual relevant movies

```text
A  B  C  D  E  F  G  H
```

Recommended Top 5

```text
A  C  X  Y  Z
```

Overlap

```text
A  C
```

Precision

```text
Relevant Returned
-----------------
Total Returned

= 2 / 5
```

Recall

```text
Relevant Returned
------------------
Total Relevant

= 2 / 8
```

---

# 🎯 Another Example

Suppose

The user likes

```text
A B C D
```

Your model recommends

```text
A B C D E
```

### Precision@5

Relevant returned = **4**

Recommended = **5**

$\text{Precision@5}=\frac{4}{5}=0.80$

---

### Recall@5

Relevant returned = **4**

Actual relevant = **4**

$\text{Recall@5}=1.0$

Perfect recall.

---

# 🎯 Another Example

User likes

```text
A B C D E
```

Recommendations

```text
A B C
```

Precision@3

Relevant returned = **3**

Recommended = **3**

$\text{Precision@3}=1.0$

Perfect precision.

Recall@3

Relevant returned = **3**

Actual relevant = **5**

$\text{Recall@3}=\frac{3}{5}=0.60$

The recommendations shown are all relevant, but two relevant items were missed.

---

# 🎯 Extreme Cases

## High Precision, Low Recall

Recommendations

```text
A
```

Only one movie recommended.

It is relevant.

Precision

```text
1 / 1 = 100%
```

Recall

Suppose there are 20 relevant movies.

```text
1 / 20 = 5%
```

Interpretation

- Almost everything recommended is correct.
- But many relevant items were never recommended.

---

## Low Precision, High Recall

Recommend

```text
100 movies
```

Suppose

80 are relevant.

Precision

```text
80 / 100 = 80%
```

If there are exactly 80 relevant movies in the catalog

Recall

```text
80 / 80 = 100%
```

Interpretation

- You found every relevant item.
- But you also recommended many unnecessary items.

> **Note:** If there are more than 80 relevant items in total, the recall would be less than 100%.

---

# 📈 Precision vs Recall

| Precision | Recall |
|------------|---------|
| Measures recommendation quality | Measures recommendation coverage |
| Focuses on returned items | Focuses on all relevant items |
| Higher means fewer false positives | Higher means fewer false negatives |
| Important when wrong recommendations are costly | Important when missing relevant items is costly |

---

# 📌 Mathematical Formulas

Precision@K

$\text{Precision@K}=\frac{\text{Relevant items in Top K}}{K}$

Recall@K

$\text{Recall@K}=\frac{\text{Relevant items in Top K}}{\text{Total Relevant Items}}$

---

# 🎬 Recommendation System Example

Suppose a user has watched and liked

```text
Harry Potter
Interstellar
Inception
Titanic
Avatar
```

The recommendation system predicts

```text
Interstellar
Avatar
Frozen
Cars
Toy Story
```

Relevant recommendations

```text
Interstellar
Avatar
```

Precision@5

$\frac{2}{5}=40\%$

Recall@5

$\frac{2}{5}=40\%$

In this example, the user likes exactly five movies, so the denominator for Recall is also five.

---

# 🔍 Search Engine Example

Search query

```text
Machine Learning Books
```

Suppose there are **100 relevant books** in the index.

Google returns

Top 10

```text
9 Relevant
1 Irrelevant
```

Precision@10

$\frac{9}{10}=90\%$

Recall@10

$\frac{9}{100}=9\%$

Interpretation

Google returned very high-quality results, but only a small fraction of all relevant books.

---

# 🤖 Why Recommendation Systems Use Both

If you optimize only Precision:

- You may recommend only a few "safe" items.
- Users miss many other relevant items.

If you optimize only Recall:

- You may recommend too many items.
- The list becomes noisy and less useful.

A good ranking system aims to balance both metrics.

---

# 🧠 Interview Questions

## Q1. What does Precision@K measure?

> Precision@K measures the fraction of the **Top K recommended items** that are actually relevant.

Formula

$\text{Precision@K}=\frac{\text{Relevant items in Top K}}{K}$

---

## Q2. What does Recall@K measure?

> Recall@K measures the fraction of **all relevant items** that appear within the Top K recommendations.

Formula

$\text{Recall@K}=\frac{\text{Relevant items in Top K}}{\text{Total Relevant Items}}$

---

## Q3. Can a model have high Precision but low Recall?

Yes.

For example, recommending only one highly relevant item can produce **100% Precision**, but if many other relevant items exist, the Recall will be low.

---

## Q4. Which metric is more important?

It depends on the application:

- **Precision@K** is more important when users only look at the first few results (e.g., homepage recommendations or top search results).
- **Recall@K** is more important when finding as many relevant items as possible matters (e.g., document retrieval, medical search, or the retrieval stage of a RAG pipeline).

---

# 📝 Key Takeaways

- **Precision@K** answers: **"Of the Top K items shown, how many are relevant?"**
- **Recall@K** answers: **"Of all the relevant items available, how many did we retrieve in the Top K?"**
- Precision focuses on the **quality** of the returned results.
- Recall focuses on the **coverage** of all relevant results.
- Improving one metric can sometimes reduce the other, so ranking systems often optimize a balance of both.



# 📚 Are Precision@K and Recall@K Calculated Differently in Retrieval and Recommendation?

**Short Answer:**

> **No.** The mathematical formulas for **Precision@K** and **Recall@K** are **exactly the same** in both **Retrieval Systems** and **Recommendation Systems**.

The **difference is not in the formulas**, but in **how we define a "relevant" item** and **how we obtain the ground truth**.

---

# 📖 Mathematical Formulas

## Precision@K

Measures:

> **Out of the Top K items returned, how many are actually relevant?**

Formula

$\text{Precision@K}=\frac{\text{Number of Relevant Items in Top K}}{K}$

---

## Recall@K

Measures:

> **Out of all relevant items that exist, how many did we retrieve in the Top K?**

Formula

$\text{Recall@K}=\frac{\text{Number of Relevant Items in Top K}}{\text{Total Number of Relevant Items}}$

---

# 🎯 The Key Difference

The **calculation remains identical**.

The **meaning of "relevant" changes** depending on the application.

| System | What is a "Relevant Item"? |
|---------|----------------------------|
| Information Retrieval | A document that correctly answers the query |
| Search Engine | A webpage that satisfies the search intent |
| RAG Retriever | A chunk/document containing information needed to answer the question |
| Recommendation System | An item the user is likely to click, watch, purchase, or enjoy |

---

# 🔍 Case 1: Information Retrieval

Suppose we have a document database.

User searches

```text
"What is Self-Supervised Learning?"
```

The database contains

```text
1000 documents
```

Among them,

```text
50 documents
```

actually answer the query.

These **50 documents** are the **ground truth relevant documents**.

---

## Retriever Output

Top 10 documents returned

```text
D1 ✅
D2 ✅
D3 ❌
D4 ✅
D5 ❌
D6 ❌
D7 ✅
D8 ❌
D9 ❌
D10 ✅
```

Relevant retrieved

```text
5
```

---

## Precision@10

$\text{Precision@10}=\frac{5}{10}=0.5$

Meaning

> **50% of the returned documents were relevant.**

---

## Recall@10

$\text{Recall@10}=\frac{5}{50}=0.10$

Meaning

> **Only 10% of all relevant documents were retrieved.**

---

# 🎬 Case 2: Recommendation Systems

Suppose Netflix has

```text
1,000,000 movies
```

The user eventually watches and enjoys

```text
20 movies
```

These become our **relevant items** (ground truth).

The recommender suggests

```text
Top 10 Movies
```

The user actually watches

```text
4
```

of them.

---

## Precision@10

$\text{Precision@10}=\frac{4}{10}=0.40$

Meaning

> **40% of the recommended movies were relevant to the user.**

---

## Recall@10

$\text{Recall@10}=\frac{4}{20}=0.20$

Meaning

> **The recommender found 20% of all the movies the user would eventually watch.**

---

# 📊 Side-by-Side Comparison

| Metric | Retrieval | Recommendation |
|----------|-----------|----------------|
| Precision@10 | $\frac{\text{Relevant Documents Retrieved}}{10}$ | $\frac{\text{Relevant Items Recommended}}{10}$ |
| Recall@10 | $\frac{\text{Relevant Documents Retrieved}}{\text{Total Relevant Documents}}$ | $\frac{\text{Relevant Items Recommended}}{\text{Total Relevant Items}}$ |

Notice that **the equations are identical**.

---

# 🤔 Where Does the Difference Actually Come From?

The biggest difference is **how we determine the ground truth (relevant items).**

## 📚 Retrieval

Ground truth is usually available.

It comes from

- Human annotations
- Benchmark datasets
- Search relevance judgments
- QA datasets

Example

```text
Query

↓

Human annotators label
which documents are relevant.
```

So we know exactly

```text
Total Relevant Documents = 50
```

---

## 🎬 Recommendation

Ground truth is usually **not known beforehand**.

Instead, it is inferred from user behavior.

Examples

- Movies watched
- Songs played
- Products purchased
- Ads clicked
- Videos liked
- Watch time

Pipeline

```text
Recommend Items

↓

User Interacts

↓

Interaction Logs Become Ground Truth
```

So the "relevant items" are discovered **after** observing user interactions.

---

# 🎯 Why Is Recall More Challenging in Recommendation?

Imagine YouTube.

```text
100 Million Videos
```

How many videos would a user actually enjoy?

There is no perfect answer.

The user may never see many videos they would have liked.

Therefore, recommendation systems often define relevance using historical interactions over a fixed evaluation period.

Example

```text
Recommended Today

↓

User Watches During Next Week

↓

Watched Videos = Relevant Items
```

Unlike retrieval, the complete set of relevant items is often **unknown**.

---

# 🤖 Precision and Recall in RAG

A Retrieval-Augmented Generation (RAG) system has **two stages**.

## Stage 1 — Retriever

```text
Question

↓

Retriever

↓

Relevant Documents
```

Metrics

- Precision@K
- Recall@K
- MRR
- nDCG
- Hit Rate

---

## Stage 2 — Generator

```text
Retrieved Documents

↓

LLM

↓

Final Answer
```

Metrics

- Exact Match (EM)
- F1 Score
- ROUGE
- BLEU
- Human Evaluation
- Faithfulness
- Answer Relevancy

A retriever may have excellent Recall@K, but if the LLM misinterprets the retrieved context, the final answer can still be incorrect.

---

# 📊 Practical Comparison

| Aspect | Retrieval | Recommendation |
|--------|-----------|----------------|
| User provides | Explicit query | User profile/history |
| Goal | Find relevant documents | Predict user preferences |
| Relevant item | Matches the query | User will interact with it |
| Ground truth | Usually human-labeled | Usually inferred from user behavior |
| Precision formula | Same | Same |
| Recall formula | Same | Same |
| Typical datasets | MS MARCO, BEIR, TREC | MovieLens, Netflix, Amazon Reviews |
| Main challenge | Matching query intent | Predicting future interests |

---

# 🧠 Interview Perspective

## Q1. Are Precision@K and Recall@K calculated differently in retrieval and recommendation systems?

> **No.** The mathematical definitions are exactly the same in both domains. The difference lies in how "relevant" items are defined and how the ground truth is obtained.

---

## Q2. What is considered a relevant item?

### Retrieval

A document that correctly answers the user's query.

### Recommendation

An item that the user is likely to interact with, such as clicking, watching, purchasing, or liking.

---

## Q3. Why is Recall harder to estimate in recommendation systems?

Because we usually **do not know every item the user would have liked**. We only observe the items the user interacted with, making the true set of relevant items incomplete.

---

# 📝 Key Takeaways

- ✅ **Precision@K and Recall@K use the same mathematical formulas** in retrieval and recommendation systems.
- ✅ The main difference is the **definition of relevance**.
- ✅ Retrieval uses an **explicit query**, while recommendation uses **implicit user preferences**.
- ✅ Retrieval often has **human-labeled relevance judgments**, whereas recommendation typically relies on **historical user interactions**.
- ✅ In recommendation systems, the complete set of relevant items is often unknown, making Recall more difficult to estimate accurately.
- ✅ In RAG systems, Precision@K and Recall@K evaluate the **retriever**, while the **generator** is evaluated using answer-quality metrics such as Exact Match, F1, ROUGE, BLEU, and human evaluation.
