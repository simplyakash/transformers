# 🎤 Interview Question: Regularization

## ❓Q1. Training loss keeps decreasing, but validation loss starts increasing. What is happening?

### ✅ Answer

This is a classic case of **Overfitting**.

The model has started **memorizing the training data** (including noise and outliers) instead of learning patterns that generalize to unseen data.

```
                 Loss
                  ▲
                  │
Validation Loss   │          ╭───────╮
                  │         ╱         ╲
                  │        ╱           ╲
                  │       ╱             ╲
Training Loss     │      ╱
                  │     ╱
                  │    ╱
                  └──────────────────────────► Epochs
```

---

## ❓Q2. Why does overfitting happen?

Overfitting occurs when the model becomes too specialized to the training data.

### Common reasons

- 📉 Small dataset
- 🧠 Model is too complex
- 🔁 Too many training epochs
- ❌ No regularization
- 🔊 Noisy labels
- 📊 High-dimensional feature space

---

# ❓Q3. How can we prevent overfitting?

## 1️⃣ L2 Regularization (Weight Decay)

Adds a penalty on large weights.

### Formula

```text
Loss = Original Loss + λ Σ(w²)
```

### Intuition

Instead of allowing weights to become very large,

```
Without L2

Feature A ─────────────► Weight = 25

With L2

Feature A ─────────────► Weight = 3.2
```

Large weights are discouraged.

Result:

- ✅ Better generalization
- ✅ Smoother decision boundary
- ✅ Less variance

---

## 2️⃣ L1 Regularization

Adds the absolute value of weights.

### Formula

```text
Loss = Original Loss + λ Σ|w|
```

Instead of shrinking every weight,

it completely removes some.

```
Before

[2.4, 1.3, 0.8, 0.02, 0.01]

↓

After L1

[2.1, 1.0, 0.4, 0, 0]
```

Result

- Removes unnecessary features
- Produces sparse models
- Automatic feature selection

---

## 3️⃣ Dropout

Randomly switches off neurons during training.

```
Layer

● ● ● ● ● ●

↓

Random Dropout

● ✖ ● ✖ ● ✖
```

Benefits

- Prevents neurons from depending on each other
- Works like averaging many small neural networks
- Improves generalization

Typical values

- 20%–50% dropout

---

## 4️⃣ Early Stopping

Monitor validation loss.

```
Epoch

1   2   3   4   5   6   7

Validation Loss

0.90
0.62
0.41
0.35  ✅ Best
0.38
0.44
0.51
```

Stop training at Epoch 4.

---

## 5️⃣ Data Augmentation

Increase dataset diversity.

### Images

```
Original

😀

↓

Rotate
Flip
Crop
Brightness
Noise
```

### Text

- Synonym replacement
- Back translation
- Paraphrasing

---

## 6️⃣ More Training Data

More data generally reduces overfitting because the model learns broader patterns instead of memorizing specific examples.

---

## 7️⃣ Simpler Model

Reduce complexity by

- Fewer layers
- Smaller hidden dimensions
- Pruning
- Feature selection

---

# ❓Q4. Explain L1 vs L2

| Feature | L1 | L2 |
|---------|----|----|
| Formula | λΣ\|w\| | λΣw² |
| Also Called | Lasso | Ridge |
| Feature Selection | ✅ Yes | ❌ No |
| Sparse Model | ✅ Yes | ❌ No |
| Drives Weights to Zero | ✅ Yes | ❌ No |
| Stable with Correlated Features | ❌ | ✅ |

---

# 📌 L1 Visualization

```
Weights

Before

10
8
5
1
0.5
0.1

↓

After

8
6
3
0
0
0
```

Many weights become exactly zero.

---

# 📌 L2 Visualization

```
Weights

Before

10
8
5
1
0.5
0.1

↓

After

7
5.8
3.9
0.7
0.3
0.05
```

Every weight becomes smaller, but almost none become exactly zero.

---

# ❓Why does L1 create sparse models?

Because the penalty uses the **absolute value**.

The optimization landscape has sharp corners at zero.

```
          ▲
          │
        ╱ │ ╲
      ╱   │   ╲
─────┼────┼──────► Weight
      ╲   │   ╱
        ╲ │ ╱
          ▼
```

Gradient descent naturally pushes many weights exactly to zero.

---

# ❓Why doesn't L2 create sparse models?

Because the squared function is smooth.

```
          ▲
         ╱╲
       ╱    ╲
     ╱        ╲
───╱────────────╲────► Weight
```

Weights are continuously reduced but rarely become exactly zero.

---

# ❓What is Lambda (λ)?

Lambda controls the strength of regularization.

```
Small λ

Model
   │
   ├── Learns aggressively
   └── Risk of Overfitting

Large λ

Model
   │
   ├── Strong penalty
   └── Risk of Underfitting
```

---

# ❓Which regularization would you use for **1 million features**?

### Preferred Answer

✅ **L1 Regularization**

Why?

- Automatically removes irrelevant features.
- Produces sparse models.
- Reduces memory usage.
- Faster inference.

### Better Practical Answer

If many features are correlated,

✅ **Elastic Net**

```
Loss

=

Original Loss

+

λ₁ Σ|w|

+

λ₂ Σ(w²)
```

Elastic Net combines

- Feature selection from L1
- Stability from L2

---

# 💼 Practical Example

Imagine building a recommendation model for Pocket FM.

Features

- User Age
- Device
- Session Duration
- Previous Stories
- Listening Time
- Genre Preference
- Thousands of Embedding Features

```
1,000,000 Features

↓

L1

↓

Only 120,000 useful features remain

↓

Smaller Model
Lower Latency
Better Generalization
```

---

# 🎯 Interview Summary (30 Seconds)

> "When training loss decreases while validation loss increases, the model is overfitting. It has started memorizing the training data rather than learning generalizable patterns. To address this, I would use techniques such as L1/L2 regularization, dropout, early stopping, data augmentation, or simplifying the model. L1 promotes sparsity by driving some weights to zero, making it useful for feature selection in high-dimensional datasets. L2 shrinks all weights smoothly, improving generalization while retaining all features. For datasets with millions of features, I would typically choose L1 or Elastic Net depending on feature correlations."
>
> # 🎤 Interview Question: Tokenization

> **Interviewer:** "Let's move to NLP. Explain tokenization in detail. Why do we need it? Also explain different tokenization algorithms like WordPiece, BPE, and SentencePiece."

---

# ❓Q1. What is Tokenization?

## ✅ Definition

**Tokenization** is the process of converting raw text into smaller units called **tokens** that a machine learning model can process.

> 💡 **Key Point:** LLMs cannot understand raw text directly. They operate on **token IDs**, not characters or words.

### Example

```text
Input Sentence

I love reading books.
```

↓

### Word Tokens

```text
["I", "love", "reading", "books", "."]
```

↓

### Token IDs

```text
[40, 1520, 8912, 5621, 13]
```

↓

### Embeddings

```text
40    → [0.21, -0.45, ...]
1520  → [1.34, 0.87, ...]
8912  → [-0.52, 1.09, ...]
```

---

# Why can't LLMs read text directly?

Computers understand only numbers.

```
Text

↓

Tokenizer

↓

Token IDs

↓

Embedding Layer

↓

Vectors

↓

Transformer
```

Without tokenization, there is no way to convert language into numerical representations.

---

# Complete Pipeline

```text
Raw Text

↓

Tokenizer

↓

Token IDs

↓

Embedding Layer

↓

Positional Encoding

↓

Transformer Layers

↓

Output Tokens

↓

Detokenizer

↓

Final Text
```

---

# Types of Tokenization

---

# 1️⃣ Character-Level Tokenization

Sentence

```text
Hello
```

Tokens

```text
["H", "e", "l", "l", "o"]
```

Advantages

- Small vocabulary
- Can represent any word
- No unknown words

Disadvantages

- Long sequences
- Slow training
- Difficult to learn long-term dependencies

---

# 2️⃣ Word-Level Tokenization

Sentence

```text
I love Machine Learning
```

Tokens

```text
["I", "love", "Machine", "Learning"]
```

Advantages

- Easy to understand
- Short sequences

Disadvantages

- Huge vocabulary
- Cannot handle unseen words
- Poor multilingual support

Example

```
Vocabulary

Cat
Dog
Apple

Input

ChatGPT

↓

Unknown Word ❌
```

---

# 3️⃣ Subword Tokenization ⭐ (Used by Modern LLMs)

Instead of splitting into words,

split into meaningful pieces.

Example

```
Unbelievable

↓

Un

believe

able
```

Advantages

- Handles unseen words
- Small vocabulary
- Better efficiency
- Good multilingual support

This is the approach used by modern LLMs.

---

# Byte Pair Encoding (BPE)

Used by:

- GPT-2
- GPT-3
- RoBERTa (variant)

### Idea

Start with characters and repeatedly merge the most frequent adjacent pairs.

Example corpus

```text
low
lowest
lower
```

Initially

```text
l o w
l o w e s t
l o w e r
```

Most frequent pair

```text
l + o

↓

lo
```

Next

```text
lo + w

↓

low
```

Continue merging until the desired vocabulary size is reached.

Example

```
l o w

↓

lo w

↓

low
```

Advantages

- Learns common words efficiently
- Handles unknown words
- Compact vocabulary

---

# WordPiece

Used by

- BERT
- DistilBERT
- ALBERT

Instead of choosing the **most frequent** merge,

WordPiece chooses the merge that maximizes **language model likelihood**.

Example

```
Playing

↓

Play

##ing
```

The `##` prefix indicates the token continues a previous word.

Example

```
Unhappiness

↓

Un

##happy

##ness
```

Advantages

- Better language modeling
- More semantically meaningful subwords

---

# SentencePiece

Used by

- T5
- LLaMA
- mT5
- ALBERT (some variants)

Unlike BPE and WordPiece,

SentencePiece **does not require whitespace**.

Example

```
I love AI
```

Instead of

```
I
love
AI
```

SentencePiece treats the entire sentence as one stream.

```
▁I
▁love
▁AI
```

The `▁` symbol represents a word boundary.

Advantages

- Works for languages without spaces (Chinese, Japanese, Thai)
- Language-independent
- Excellent multilingual support

---

# Comparison

| Feature | Character | Word | BPE | WordPiece | SentencePiece |
|----------|-----------|------|-----|-----------|---------------|
| Unknown Words | ✅ | ❌ | ✅ | ✅ | ✅ |
| Vocabulary Size | Very Small | Very Large | Medium | Medium | Medium |
| Sequence Length | Long | Short | Medium | Medium | Medium |
| Used in Modern LLMs | ❌ | ❌ | ✅ | ✅ | ✅ |

---

# Why does GPT use BPE?

Reasons:

- Efficient vocabulary size (~50k–100k tokens)
- Handles unseen words by breaking them into subwords
- Compresses common words into single tokens
- Reduces sequence length compared to character-level tokenization

Example

```
extraordinary

↓

extra

ordinary
```

Instead of

```
e
x
t
r
a
...
```

---

# Why not tokenize on whitespace?

Whitespace tokenization fails for:

### New words

```
ChatGPTPlusUltra

↓

Unknown ❌
```

### Misspellings

```
recieve
```

### Different languages

```
你好世界
```

Many languages do not use spaces consistently.

Subword tokenization solves these issues.

---

# What happens when the tokenizer sees an unseen word?

Example

```
Electroencephalograph
```

↓

Instead of returning UNKNOWN,

BPE splits it into known pieces.

```
Electro

encephalo

graph
```

This allows the model to process words it has never seen before.

---

# Why is vocabulary size usually 32K–100K?

Trade-off:

Small vocabulary

```
Vocabulary ↓

Sequence Length ↑

Training Cost ↑
```

Large vocabulary

```
Vocabulary ↑

Embedding Matrix ↑

Memory ↑
```

Most LLMs choose a vocabulary size between **32K and 100K** to balance memory usage and sequence length.

---

# Special Tokens

Examples

| Token | Purpose |
|--------|---------|
| `<PAD>` | Padding |
| `<EOS>` | End of sequence |
| `<BOS>` | Beginning of sequence |
| `<UNK>` | Unknown token (rare in modern subword tokenizers) |
| `<CLS>` | Classification token (BERT) |
| `<SEP>` | Separator between sentences |

---

# Interview Follow-up

### Why is `<UNK>` rare in GPT-style models?

Because subword tokenization (e.g., BPE) can decompose almost any word into known subword units, making a dedicated unknown token largely unnecessary.

---

# 🎯 30-Second Interview Summary


> "Tokenization converts raw text into token IDs that a model can process. Modern LLMs use **subword tokenization** because it balances vocabulary size, sequence length, and the ability to handle unseen words. GPT uses **Byte Pair Encoding (BPE)**, which merges the most frequent symbol pairs. BERT uses **WordPiece**, which selects merges based on language model likelihood. Models like LLaMA and T5 use **SentencePiece**, which is language-independent and doesn't rely on whitespace. After tokenization, token IDs are mapped to embeddings and fed into the Transformer."
>
> # 🎤 Interview Question: Pre-training, Mid-training & Post-training

> **Interviewer:** "Can you explain the complete lifecycle of training a modern Large Language Model? Specifically, what are Pre-training, Mid-training, and Post-training? Why do we need each stage, and what happens in each?"

---

# 🏗️ Complete LLM Training Pipeline

```text
                    Internet Data
                         │
                         ▼
               ┌──────────────────┐
               │   Data Collection │
               └──────────────────┘
                         │
                         ▼
               ┌──────────────────┐
               │   Tokenization    │
               └──────────────────┘
                         │
                         ▼
══════════════════════════════════════════════
            ① PRE-TRAINING
══════════════════════════════════════════════
                         │
                         ▼
         Learns Language & World Knowledge
                         │
                         ▼
══════════════════════════════════════════════
            ② MID-TRAINING
══════════════════════════════════════════════
                         │
                         ▼
      Domain Adaptation / Continued Pretraining
                         │
                         ▼
══════════════════════════════════════════════
            ③ POST-TRAINING
══════════════════════════════════════════════
                         │
                         ▼
    Instruction Following + Alignment + Safety
                         │
                         ▼
                 Final Chat Model
```

---

# 📌 Stage 1 — Pre-training

## What is it?

Pre-training is the **first and most computationally expensive** phase where the model learns:

- Grammar
- Facts
- Reasoning patterns
- Coding syntax
- Mathematics
- General world knowledge

The model has **no concept of answering questions** yet.

It simply learns to **predict the next token**.

---

## Training Objective

Given:

```text
The capital of France is
```

The model predicts:

```text
Paris
```

Another example:

```text
I love playing
```

↓

```text
football
```

The objective is called **Causal Language Modeling (CLM)**.

---

# Mathematical Objective

The model maximizes

```text
P(next token | previous tokens)
```

Example

```text
Input

The sky is

↓

Predict

blue
```

Loss Function

```text
Cross Entropy Loss
```

---

# Data Used

Huge datasets such as

- Books
- Wikipedia
- GitHub
- Research papers
- News articles
- Public web pages
- Stack Overflow
- Educational content

Typically

```text
Trillions of tokens
```

Examples

```
GPT-3

≈300B tokens

Llama 3

≈15T tokens
```

*(Exact numbers vary by model version.)*

---

# What does the model learn?

It learns

✅ Grammar

```
I am

↓

happy
```

---

✅ Facts

```
Capital of India

↓

New Delhi
```

---

✅ Coding

```
for i in range()

↓

:
```

---

✅ Reasoning Patterns

```
2 + 3

↓

5
```

---

# Output after Pre-training

The model becomes

```
Very knowledgeable

BUT

Not conversational.
```

Example

Prompt

```
Tell me a joke.
```

Response

```
Tell me a joke is a phrase...
```

Not ideal.

---

# 📌 Stage 2 — Mid-training (Continued Pre-training)

## What is Mid-training?

Mid-training adapts a pretrained model to a **specific domain** before instruction tuning.

Also called

- Continued Pre-training
- Domain Adaptive Pre-training (DAPT)

---

## Why do we need it?

The base model has broad knowledge but may lack depth in specialized domains.

Examples

```
Medical AI

↓

Medical textbooks
Clinical papers

↓

Better medical understanding
```

---

```
Legal AI

↓

Court judgments
Legal documents

↓

Better legal reasoning
```

---

```
Code Assistant

↓

Millions of GitHub repositories

↓

Better code generation
```

---

## Objective

The objective **does not change**.

Still predicts

```text
Next Token
```

Only the training data changes.

---

# Example

General GPT

```
Explain Transformer
```

↓

Good answer.

---

Medical GPT

```
Explain diabetes treatment
```

↓

Much more accurate because it saw domain-specific medical literature during mid-training.

---

# 📌 Stage 3 — Post-training

This stage converts a pretrained model into a helpful assistant.

Without post-training

```
Prompt

Write an email

↓

Model

Email...
email...
email...
```

Not useful.

---

After post-training

```
Prompt

Write a professional resignation email.

↓

Well-structured email.
```

---

# Components of Post-training

---

# 1️⃣ Supervised Fine-Tuning (SFT)

Human-written examples are used.

Dataset

```
Question

↓

Ideal Answer
```

Example

```
User

Explain Gradient Descent

↓

Assistant

Detailed explanation...
```

The model learns

```
Question

↓

Good Answer
```

---

# 2️⃣ RLHF (Reinforcement Learning from Human Feedback)

Pipeline

```text
Model Outputs

↓

Humans Rank Responses

↓

Reward Model

↓

Reinforcement Learning

↓

Improved Assistant
```

Example

Question

```
Explain AI
```

Response A

```
AI is...
```

Response B

```
AI is a field of computer science...
```

Humans prefer B.

The reward model learns these preferences.

---

# 3️⃣ DPO (Direct Preference Optimization)

Modern alternative to RLHF.

Instead of

```
Policy

↓

Reward Model

↓

PPO

↓

Updated Policy
```

DPO directly learns from preference pairs.

Example

```
Preferred Answer ✅

Rejected Answer ❌
```

↓

Optimize the model to assign higher probability to the preferred answer.

Advantages

- Simpler
- More stable
- No separate reward model
- Easier to train

Many modern LLMs use DPO or similar preference optimization techniques.

---

# Why is Post-training Needed?

Pre-training teaches

```
Language
```

Post-training teaches

```
Behavior
```

Examples

- Helpfulness
- Instruction following
- Safety
- Refusal behavior
- Conversational tone
- Formatting
- Multi-turn dialogue

---

# Comparison

| Stage | Goal | Data | Objective |
|--------|------|------|-----------|
| Pre-training | Learn language & knowledge | Internet-scale corpus | Next-token prediction |
| Mid-training | Domain adaptation | Domain-specific corpus | Next-token prediction |
| Post-training | Learn to behave as an assistant | Instruction & preference datasets | SFT + RLHF/DPO |

---

# Interview Follow-up Questions

## Why can't we skip pre-training?

Because the model would have **no language understanding**.

Instruction tuning alone cannot teach grammar, reasoning, or factual knowledge from scratch.

---

## Why not directly fine-tune from scratch?

Fine-tuning assumes the model already understands language.

Training from scratch would require enormous datasets and compute, making it prohibitively expensive.

---

## Why is Mid-training cheaper than Pre-training?

Because the model already has strong general language representations.

It only needs additional exposure to domain-specific data rather than learning language from the beginning.

---

## Why is Post-training relatively small?

It focuses on changing the model's **behavior**, not rebuilding its knowledge.

Typically, post-training datasets are much smaller than pre-training corpora.

---

# Real-World Example (Pocket FM)

Suppose Pocket FM wants an AI narrator and story assistant.

### 1. Pre-training

Learn general English, storytelling, grammar, and world knowledge from large internet datasets.

### 2. Mid-training

Continue training on:

- Audiobook transcripts
- Story scripts
- Fiction novels
- Dialogue-heavy content

The model learns narrative style and storytelling conventions.

### 3. Post-training

Train with instruction-following examples such as:

```
User:
Summarize this chapter.

↓

Assistant:
Concise chapter summary.
```

Preference optimization then makes responses more engaging, safe, and aligned with user expectations.

---

# 🎯 60-Second Interview Answer

> "A modern LLM is typically trained in three stages. **Pre-training** teaches the model language, reasoning, and world knowledge by predicting the next token on trillions of tokens from books, web pages, code, and other public text. **Mid-training**, also called continued pre-training, further adapts the model to a specific domain—such as medicine, law, finance, or storytelling—while keeping the same next-token prediction objective. **Post-training** aligns the model with human expectations using supervised fine-tuning on instruction-response pairs and preference optimization techniques like RLHF or DPO. Pre-training builds knowledge, mid-training specializes that knowledge, and post-training teaches the model how to behave as a helpful, safe, and conversational assistant."
