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
# 🎯 What are Weights in Machine Learning?

A **weight** is a learnable parameter that determines how much importance the model gives to an input feature.

Think of a simple linear model:

```text
y = w₁x₁ + w₂x₂ + w₃x₃ + b
```

Where:

- `x₁, x₂, x₃` = input features
- `w₁, w₂, w₃` = weights
- `b` = bias

Example:

Suppose we're predicting the price of a house.

```text
Price

=

(Size × 5000)

+

(Bedrooms × 100000)

+

(Age × -2000)

+

Bias
```

Here,

```text
Weight for Size = 5000

Weight for Bedrooms = 100000

Weight for Age = -2000
```

The weights tell the model **how influential each feature is**.

---

# In Neural Networks

Weights connect neurons.

```
Input Layer          Hidden Layer

x₁  ● ───(w₁)────► ○

x₂  ● ───(w₂)────► ○

x₃  ● ───(w₃)────► ○
```

Every connection has a weight.

A modern LLM contains **billions of weights**.

For example,

```
7 Billion Parameter Model

↓

≈ 7 Billion Weights (plus some biases)
```

These weights store everything the model has learned:

- Grammar
- Facts
- Coding knowledge
- Reasoning
- Language patterns

---

# Why Large Weights Can Be a Problem

Suppose we have:

```text
y = 1000x₁ + 0.1x₂
```

The model depends almost entirely on `x₁`.

If `x₁` contains noise,

the prediction changes drastically.

Large weights make the model:

- Sensitive to noise
- Less robust
- More likely to overfit

---

# What Does L2 Regularization Do?

The loss becomes

```text
Loss

=

Original Loss

+

λ Σ(w²)
```

Notice:

The optimizer wants to **minimize** the total loss.

If a weight becomes very large,

its square becomes much larger.

Example

```
Weight = 2

Penalty = 2² = 4

--------------

Weight = 10

Penalty = 10² = 100

--------------

Weight = 20

Penalty = 20² = 400
```

Large weights are penalized much more than small weights.

---

# But HOW Does the Weight Actually Become Smaller?

This happens through **Gradient Descent**.

The update rule is

```text
New Weight

=

Old Weight

−

Learning Rate × Gradient
```

Without L2:

```text
Loss = Original Loss
```

Gradient:

```text
∂Loss/∂w
```

---

With L2:

```text
Loss

=

Original Loss

+

λw²
```

Take the derivative.

Derivative of

```text
λw²
```

is

```text
2λw
```

Therefore,

the total gradient becomes

```text
Gradient

=

Original Gradient

+

2λw
```

Notice something important:

The extra term

```text
2λw
```

always points toward **zero**.

---

# Numerical Example

Suppose

```text
Weight = 8

Learning Rate = 0.1

Original Gradient = 1

λ = 0.2
```

Without L2:

```text
Gradient

=

1
```

Update

```text
w

=

8

−

0.1 × 1

=

7.9
```

---

With L2

Extra Gradient

```text
2 × 0.2 × 8

=

3.2
```

Total Gradient

```text
1 + 3.2

=

4.2
```

Update

```text
w

=

8

−

0.1 × 4.2

=

7.58
```

Notice

```
Without L2

8

↓

7.9

--------------

With L2

8

↓

7.58
```

The weight shrinks **much faster**.

---

# Why Doesn't It Become Exactly Zero?

The derivative of `w²` is proportional to `w`:

```text
d(w²)/dw = 2w
```

As the weight gets closer to zero,

the gradient also becomes smaller.

Example

```
Weight = 8

Penalty Gradient = 16

↓

Weight = 2

Penalty Gradient = 4

↓

Weight = 0.5

Penalty Gradient = 1

↓

Weight = 0.1

Penalty Gradient = 0.2
```

Eventually, the shrinking force becomes very small, so weights approach zero but rarely become exactly zero.

---

# Why Does This Reduce Overfitting?

Imagine fitting a curve.

Without L2:

```
Training Data

●   ●      ●
   ●

Curve

~~~~~~~~~~~~~~~
```

The curve twists to fit every point, including noise.

With L2:

```
Training Data

●   ●      ●
   ●

Curve

──────────────
```

The model is encouraged to use **smaller weights**, producing smoother decision boundaries that generalize better to unseen data.

---

# Interview One-Liner

> "Weights are the learnable parameters that determine the influence of each feature or neuron connection. L2 regularization adds a penalty proportional to the square of the weights to the loss function. During gradient descent, this introduces an additional gradient term `2λw`, which continuously pulls weights toward zero. Because large weights incur a much larger penalty than small ones, the optimizer naturally prefers smaller weights, reducing model complexity and improving generalization."

# 🎯 Why Does L1 Regularization Make Some Weights Exactly Zero?

Recall the L1 loss function:

```text
Loss

=

Original Loss

+

λ Σ|w|
```

Unlike L2, L1 penalizes the **absolute value** of the weights.

---

# Step 1: What is the Gradient of L1?

For L2:

```text
Penalty = λw²

Derivative = 2λw
```

Notice that the gradient depends on the value of `w`.

As `w` gets smaller, the gradient also gets smaller.

---

For L1:

```text
Penalty = λ|w|
```

The derivative is:

```text
        +λ    if w > 0

d|w| =
        -λ    if w < 0

Undefined at w = 0
```

Or equivalently:

```text
d|w| = λ × sign(w)
```

where

```text
sign(w)

=

+1   if w > 0

-1   if w < 0

0    if w = 0   (subgradient)
```

The important observation is:

> **The gradient has a constant magnitude (λ), regardless of how small the weight is.**

---

# Step 2: Compare L1 and L2

Suppose

```text
Weight = 10
```

L2 gradient:

```text
2 × 10 = 20
```

L1 gradient:

```text
1
```

---

Now suppose

```text
Weight = 0.1
```

L2 gradient:

```text
2 × 0.1 = 0.2
```

L1 gradient:

```text
1
```

Notice the difference:

```
Weight      L2 Gradient      L1 Gradient

10              20               1

1                2               1

0.1            0.2               1

0.01          0.02               1
```

L2's shrinking force becomes weaker as the weight approaches zero.

L1 keeps applying the same-sized push toward zero.

---

# Step 3: Numerical Example

Suppose

```text
Weight = 0.08

Learning Rate = 0.1

λ = 0.5
```

Ignoring the data-loss gradient for simplicity:

The L1 update is

```text
w_new

=

w_old

−

η × λ

=

0.08

−

0.1 × 0.5

=

0.03
```

Next step:

```text
0.03

−

0.05

=

-0.02
```

The optimizer doesn't usually let the weight oscillate through zero. In practice (using subgradients or proximal methods), it is **clipped to exactly zero**.

So:

```
0.08

↓

0.03

↓

0.00 ✅
```

---

# L2 Example

Start with the same weight:

```text
Weight = 0.08
```

Gradient:

```text
2w

=

0.16
```

Update:

```text
0.08

−

0.1 × 0.16

=

0.064
```

Next:

```text
0.064

↓

0.0512

↓

0.041

↓

0.033

↓

...
```

It keeps getting smaller but rarely reaches exactly zero.

---

# Visual Intuition

## L2 Penalty

```
Penalty
  ▲
  │
  │      ╭───╮
  │    ╱       ╲
  │  ╱           ╲
──┼──────────────────► Weight
```

This curve is smooth.

Near zero, the slope is also close to zero.

So the optimizer loses the "force" needed to eliminate weights.

---

## L1 Penalty

```
Penalty
  ▲
  │
  │     ╱
  │    ╱
──┼───┼────────► Weight
  │    ╲
  │     ╲
```

The sharp corner at zero means there is always pressure toward the origin from either side.

This encourages exact zeros.

---

# Geometric Intuition (Very Common Interview Question)

Suppose the optimizer is minimizing the loss while staying within a regularization constraint.

### L2 Constraint

```
      ○
   ○     ○
 ○         ○
○     ●     ○
 ○         ○
   ○     ○
      ○
```

The constraint is a **circle**.

The optimum usually touches the smooth boundary away from the axes.

Weights become small but are rarely zero.

---

### L1 Constraint

```
        ▲
       / \
      /   \
◄────●────►
      \   /
       \ /
        ▼
```

The constraint is a **diamond**.

Its corners lie on the coordinate axes, where one or more weights are exactly zero.

The optimum is much more likely to land on a corner.

This is the geometric reason why L1 produces sparse solutions.

---

# Why Is Sparsity Useful?

Imagine you have 1,000,000 features:

```
Before L1

[0.8, 0.2, 1.1, 0.03, 0.9, 0.01, 0.0...]

↓

After L1

[0.7, 0, 1.0, 0, 0.8, 0, 0...]
```

Benefits:

- ✅ Automatic feature selection
- ✅ Smaller model
- ✅ Faster inference
- ✅ Better interpretability
- ✅ Can reduce overfitting

---

# Interview Answer (30 Seconds)

> "L1 regularization adds the absolute value of the weights to the loss. Its gradient is proportional to the sign of the weight, so it has a nearly constant magnitude regardless of how close the weight is to zero. Unlike L2, whose shrinking force becomes weaker near zero, L1 keeps pushing small weights toward zero. In optimization, this causes many unimportant weights to become exactly zero, producing sparse models and performing automatic feature selection."

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

# 🎤 Interview Question

**Design a Recommendation System for Pocket FM (Songs / Audiobooks / Episodes)**

---

# High-Level Architecture

```text
                  User Opens App
                         │
                         ▼
              User Features + Context
                         │
                         ▼
              Candidate Generation
             (Retrieve ~1000 Items)
                         │
                         ▼
                 Ranking Model
             (Rank Top 100 Items)
                         │
                         ▼
               Re-ranking Layer
       (Diversity + Freshness + Business Rules)
                         │
                         ▼
                 Top 20 Recommendations
```

---

# Step 1: Candidate Generation

The goal is to quickly retrieve a few hundred or thousand potentially relevant items from millions of songs or episodes.

### Common approaches

## 1. Collaborative Filtering

Users with similar listening behavior tend to like similar content.

Example

```
User A

Episode 1
Episode 2
Episode 5

User B

Episode 1
Episode 2
Episode 8

↓

Recommend Episode 8 to User A
```

### Algorithms

- Matrix Factorization
- Two-Tower Networks
- Neural Collaborative Filtering

---

## 2. Content-Based Filtering

Recommend items similar to what the user already likes.

Example

```
User likes

Horror
Mystery
Thriller

↓

Recommend

Crime Podcasts
Suspense Stories
```

Features

- Genre
- Author
- Language
- Duration
- Tags
- Description Embeddings

---

## 3. Embedding Retrieval (Most Common Today)

Represent users and items as vectors.

```
User Embedding

↓

[0.2, 0.8, ...]

Episode Embedding

↓

[0.21, 0.75, ...]

↓

Cosine Similarity

↓

High Score
```

Nearest-neighbor search retrieves similar content.

Tools:

- FAISS
- ScaNN
- HNSW

---

# Step 2: Ranking

Now we have ~1000 candidates.

We need to rank them.

Use a Learning-to-Rank model.

Example features:

User Features

- Age
- Language
- Listening history
- Preferred genres
- Average session length
- Time of day

Episode Features

- Genre
- Popularity
- Release date
- Completion rate
- Average rating

Interaction Features

- User likes horror
- Episode is horror
- Same narrator
- Similar embedding
- Previously listened to related episodes

---

# Ranking Models

Common choices:

- XGBoost
- LightGBM
- Deep Neural Networks
- Wide & Deep Models
- DeepFM
- Transformer-based ranking (for large-scale systems)

Prediction target:

```
P(User Clicks Episode)
```

or

```
Expected Watch Time
```

or

```
Completion Probability
```

---

# Step 3: Re-ranking

The highest-scoring items may all be from the same genre.

Example:

```
Top 10

Crime
Crime
Crime
Crime
Crime
Crime
Crime
Crime
Crime
Crime
```

Poor user experience.

Re-ranking introduces:

- Diversity
- Freshness
- New releases
- Different creators
- Business priorities (e.g., promote exclusive content)

Final list:

```
Crime

Romance

Comedy

Mystery

Sci-Fi

Crime

Motivational
```

---

# Cold Start Problem

## New User

No listening history.

Solutions:

- Ask onboarding questions (favorite genres, languages)
- Use demographic or location signals
- Recommend trending content
- Recommend editor's picks

---

## New Episode

No interactions yet.

Solutions:

Use content features:

- Description embeddings
- Genre
- Narrator
- Author
- Tags
- Similarity to existing episodes

---

# What Embeddings Would You Use?

For textual metadata (title, description):

- Sentence Transformers
- BERT embeddings
- Modern embedding models (e.g., OpenAI, Jina, BGE, E5)

For user behavior:

Learn user embeddings from:

- Listening history
- Search queries
- Likes
- Skips
- Completion rate

---

# Important Features

User Features

- Preferred language
- Favorite genres
- Listening time
- Device
- Subscription type
- Time of day
- Day of week

Episode Features

- Genre
- Duration
- Author
- Narrator
- Popularity
- Freshness
- Completion rate

Behavior Features

- Click-through rate (CTR)
- Average listening duration
- Skip rate
- Likes
- Shares
- Replays

---

# Online Feedback

The system should continuously learn from user behavior.

Positive Signals

- Played
- Finished episode
- Liked
- Shared
- Added to playlist

Negative Signals

- Skipped
- Stopped after a few seconds
- Disliked
- Hid content

These signals update user embeddings and ranking models over time.

---

# Evaluation Metrics

Offline Metrics

- Precision@K
- Recall@K
- NDCG
- MAP
- MRR

Online Metrics

- CTR (Click Through Rate)
- Watch/Listen Time
- Completion Rate
- Daily Active Users
- Session Length
- Retention
- Revenue

---

# Tech Stack

Storage

- User profiles
- Episode metadata
- Interaction logs

Feature Store

- Feast (or equivalent)

Vector Database / ANN Search

- FAISS
- HNSW
- ScaNN

Model Serving

- TensorFlow Serving
- TorchServe
- Triton Inference Server

Streaming

- Kafka

Monitoring

- Prometheus
- Grafana

---

# If They Ask: "Would you use Collaborative Filtering or Content-Based Filtering?"

A strong answer is:

> "Neither alone. I'd build a hybrid recommendation system. Collaborative filtering captures user behavior patterns, while content-based filtering helps with cold-start items. Candidate generation would use embeddings and collaborative filtering, followed by a learning-to-rank model and a re-ranking stage for diversity and business constraints."

---

# Pocket FM-Specific Improvements

Since Pocket FM focuses on long-form audio, I would optimize for **long-term engagement**, not just clicks.

Instead of predicting only CTR, I'd predict:

- Probability of starting an episode
- Probability of completing the episode
- Expected listening time
- Probability of returning the next day (retention)

This better aligns the recommendation system with the platform's business goals.
