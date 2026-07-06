# 🎯 Standard Ranking Models Used in Recommendation Systems

> **Interview Question:** *"What ranking models would you use while building a recommendation engine for Pocket FM, Spotify, Netflix, or YouTube?"*

---

# 📌 Where Does Ranking Fit?

A modern recommendation system has three stages:

```text
                    User Opens App
                           │
                           ▼
                Candidate Generation
            (Retrieve ~1000 candidates)
                           │
                           ▼
                  Ranking Model ⭐
         (Predict relevance score for each item)
                           │
                           ▼
                  Re-ranking Layer
      (Diversity, Freshness, Business Rules)
                           │
                           ▼
               Final Top-N Recommendations
```

The **Ranking Model** assigns a score to every candidate.

Example:

| Episode | Predicted Score |
|----------|----------------:|
| Horror Story | 0.97 |
| Crime Podcast | 0.91 |
| Romance Story | 0.82 |
| Comedy Show | 0.54 |

The system recommends the highest-scoring items.

---

# 🎯 What Does the Ranking Model Predict?

Depending on business goals, it predicts one or more of:

- Click Probability (CTR)
- Listen Probability
- Completion Probability
- Expected Watch/Listen Time
- User Satisfaction
- Retention Probability
- Revenue Probability

For Pocket FM, predicting **Expected Listening Time** or **Completion Rate** is often more valuable than predicting only clicks.

---

# 🥇 1. Logistic Regression (Baseline)

## Idea

Predict the probability that the user interacts with an item.

Example:

```text
P(User Clicks Episode)
```

Formula

```text
Score

=

σ(w₁x₁ + w₂x₂ + ... + b)
```

where σ is the sigmoid function.

---

## Features

User Features

- Age
- Gender
- Preferred Language
- Favorite Genres

Episode Features

- Genre
- Duration
- Popularity
- Average Rating

Interaction Features

- User likes Horror
- Episode is Horror

---

## Advantages

✅ Very fast

✅ Easy to interpret

✅ Easy to deploy

---

## Disadvantages

❌ Cannot learn complex relationships.

---

# 🥈 2. Decision Tree

Decision trees split data using feature values.

Example

```text
               Genre = Horror?

               /             \
             Yes             No

         Age < 25?      Comedy?

         /     \

 Recommend  Don't Recommend
```

---

## Advantages

- Easy to understand
- Captures nonlinear relationships

---

## Disadvantages

- Overfits easily
- Not ideal alone for production ranking

---

# 🥉 3. Random Forest

Instead of one tree,

use many trees.

```text
Tree 1

↓

Tree 2

↓

Tree 3

↓

Average Prediction
```

---

## Advantages

- Better than a single tree
- Less overfitting

---

## Disadvantages

- Large memory
- Slower inference
- Rarely used in large recommendation systems

---

# ⭐ 4. Gradient Boosted Decision Trees (GBDT)

This is one of the most common ranking models.

Popular implementations:

- XGBoost
- LightGBM
- CatBoost

---

## How It Works

Instead of building one large tree,

each new tree learns from the errors of the previous trees.

```text
Tree 1

↓

Prediction

↓

Error

↓

Tree 2 learns Error

↓

Smaller Error

↓

Tree 3

↓

Final Prediction
```

---

## Why is GBDT so popular?

It works extremely well on structured data.

Example Features

```text
User Age

Subscription Type

Listening Time

Completion Rate

Genre

Language

Device

Time of Day
```

---

## Advantages

✅ Excellent accuracy

✅ Fast inference

✅ Handles missing values

✅ Works well on tabular data

✅ Easy feature importance analysis

---

## Interview Tip

Most production recommendation systems still use **LightGBM** or **XGBoost** for ranking when the input features are mainly structured/tabular.

---

# ⭐⭐⭐ 5. Learning to Rank (LTR)

Traditional classification predicts:

```text
Will the user click?
```

Ranking predicts:

```text
Which episode should appear first?
```

LTR directly optimizes the order of recommendations.

---

## Types of Learning to Rank

---

### A. Pointwise Ranking

Treat every item independently.

```text
Episode A

↓

Click Probability

↓

0.93
```

Simple binary classification.

---

### B. Pairwise Ranking

Compare two items.

```text
Episode A

vs

Episode B

↓

Which should rank higher?
```

Algorithms

- RankNet
- LambdaRank

---

### C. Listwise Ranking ⭐

Instead of comparing two items,

optimize the entire ranked list.

```text
Top 20 Episodes

↓

Optimize Whole List
```

Algorithms

- LambdaMART
- ListNet

This usually performs best because it directly optimizes ranking metrics such as **NDCG**.

---

# ⭐⭐⭐ 6. Wide & Deep Model

Introduced by Google.

Architecture

```text
                 Features
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   Wide Component          Deep Neural Network
 (Memorization)             (Generalization)
        │                         │
        └────────────┬────────────┘
                     ▼
               Final Prediction
```

---

## Wide Component

Learns memorized feature interactions.

Example

```text
User likes Horror

+

Episode is Horror

↓

Recommend
```

---

## Deep Component

Learns hidden relationships automatically.

Example

```text
User likes Crime

↓

May also like

Mystery
Thriller
```

---

## Advantages

- Works well with sparse features
- Combines memorization and generalization

---

# ⭐⭐⭐ 7. DeepFM

DeepFM improves Wide & Deep.

Instead of manually creating feature interactions,

it learns them automatically.

Architecture

```text
Features

↓

Embedding Layer

↓

Factorization Machine

+

Deep Neural Network

↓

Prediction
```

Very popular in

- Ads
- Shopping
- Recommendation systems

---

# ⭐⭐⭐⭐ 8. Two-Tower Model

Primarily used for **Candidate Generation**, but can also provide features for ranking.

Architecture

```text
              User Features
                     │
             User Neural Network
                     │
                     ▼
              User Embedding

────────────────────────────────────

             Episode Features
                     │
             Item Neural Network
                     │
                     ▼
             Item Embedding

────────────────────────────────────

Cosine Similarity

↓

Similarity Score
```

---

## Advantages

- Scales to millions of items
- Fast retrieval
- Embedding-based personalization

---

# ⭐⭐⭐⭐⭐ 9. Transformer-Based Ranking

Modern recommendation systems increasingly use Transformers to model user behavior.

Example

```text
Listening History

Episode 1

↓

Episode 5

↓

Episode 8

↓

Episode 15

↓

Transformer

↓

Predict Next Episode
```

Popular models

- SASRec
- BERT4Rec
- DIN (Deep Interest Network)
- DIEN (Deep Interest Evolution Network)

These capture changing user interests over time.

---

# 📊 Comparison

| Model | Complexity | Accuracy | Used For |
|--------|------------|----------|----------|
| Logistic Regression | Low | Medium | Baseline |
| Decision Tree | Low | Medium | Simple ranking |
| Random Forest | Medium | Good | Small systems |
| LightGBM | Medium | Excellent | Production ranking ⭐ |
| XGBoost | Medium | Excellent | Production ranking ⭐ |
| CatBoost | Medium | Excellent | Categorical features |
| Wide & Deep | High | Excellent | Large recommendation systems |
| DeepFM | High | Excellent | CTR prediction |
| Transformer Ranking | Very High | Excellent | Sequential recommendations |

---

# 🎵 What Would I Use for Pocket FM?

## Candidate Generation

```text
Two-Tower Embedding Model

+

FAISS / HNSW
```

Retrieve the top ~1000 relevant episodes.

---

## Ranking

Use one of:

- LightGBM ⭐ (excellent for structured features)
- XGBoost ⭐
- Wide & Deep (if rich embeddings are available)
- DeepFM (for large-scale recommendation)

---

## Re-ranking

Apply business rules:

- Diversity
- Freshness
- New releases
- Subscription-specific recommendations
- Avoid repetitive content

---

# 📈 Features Used in Ranking

## User Features

- Age
- Preferred Language
- Favorite Genres
- Subscription Type
- Listening History
- Average Session Length
- Time of Day

---

## Episode Features

- Genre
- Duration
- Popularity
- Completion Rate
- Release Date
- Narrator
- Author

---

## Interaction Features

- User Genre × Episode Genre
- Similarity Score
- Previous Completion Rate
- Time Since Last Listen
- Number of Previous Plays

---

# 📊 Evaluation Metrics

## Offline

- Precision@K
- Recall@K
- NDCG
- MAP
- MRR

---

## Online

- CTR
- Average Listening Time
- Completion Rate
- Daily Active Users (DAU)
- Retention
- Session Length

---

# ⭐ Interview Tips

If asked:

> **"Why LightGBM instead of Deep Learning?"**

A good answer is:

- LightGBM is extremely strong for **tabular data**.
- It trains quickly and has low inference latency.
- It handles missing values naturally.
- It requires less data than deep neural networks.
- It provides interpretable feature importance.

If asked:

> **"When would you use Deep Learning?"**

Answer:

- When you have **millions of users and items**.
- When you want to leverage embeddings.
- When sequential user behavior matters.
- When rich multimodal features (text, audio, images) are available.

---

# 🎯 60-Second Interview Answer

> "For Pocket FM, I'd use a multi-stage recommendation system. First, I'd retrieve around 1000 candidate episodes using a Two-Tower embedding model with ANN search such as FAISS or HNSW. Then I'd rank those candidates using LightGBM or XGBoost because they perform exceptionally well on tabular user, item, and interaction features while providing fast inference. If richer embeddings and sequential listening history are available, I'd consider Wide & Deep, DeepFM, or Transformer-based ranking models like SASRec. Finally, I'd apply a re-ranking layer to improve diversity, freshness, and business objectives. Rather than optimizing only for clicks, I'd optimize for long-term engagement metrics such as expected listening time and episode completion rate, since these better reflect Pocket FM's business goals."
