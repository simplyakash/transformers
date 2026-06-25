# Hyperparameter Tuning in Transformers

Hyperparameter tuning is the process of finding the best configuration of training parameters to maximize validation performance.

Unlike traditional ML models, Transformer training is expensive, so efficient search strategies are preferred over exhaustive search.

---

# Common Hyperparameters to Tune

## 1. Learning Rate (Most Important)

The learning rate is often the most sensitive hyperparameter.

Typical values for BERT-style fine-tuning:

```text
1e-5
2e-5
3e-5
5e-5
```

Example:

```python
learning_rate = 2e-5
```

---

## 2. Batch Size

Common values:

```text
8
16
32
64
```

Tradeoff:

```text
Larger Batch Size
    ↓
Faster Training
    ↓
More GPU Memory Required
```

---

## 3. Number of Epochs

Typical range:

```text
2 - 10
```

Common BERT fine-tuning:

```text
3 - 5 epochs
```

---

## 4. Weight Decay

Used for regularization.

Typical values:

```text
0
0.01
0.1
```

Example:

```python
weight_decay = 0.01
```

---

## 5. Warmup Ratio / Warmup Steps

Transformers are sensitive at the beginning of training.

Instead of immediately using the full learning rate:

```text
Learning Rate

0
↑
↑
Peak LR
↓
↓
Decay
```

The learning rate gradually increases during warmup.

Typical values:

```text
5% - 10% of total training steps
```

---

## 6. Dropout

Used to prevent overfitting.

Typical values:

```text
0.1
0.2
0.3
0.5
```

---

## 7. Maximum Sequence Length

Examples:

```text
128
256
512
1024
```

Tradeoff:

```text
Longer Context
    ↓
Higher GPU Memory Usage
```

---

# Example Search Space

```python
learning_rates = [1e-5, 2e-5, 5e-5]

batch_sizes = [8, 16, 32]

weight_decay = [0.0, 0.01, 0.1]

epochs = [2, 3, 5]
```

---

# Hyperparameter Search Methods

## 1. Grid Search

Try every possible combination.

Example:

```text
LR      Batch Size

1e-5    8
1e-5    16
1e-5    32

2e-5    8
2e-5    16
...
```

If:

```text
3 Learning Rates
3 Batch Sizes
3 Weight Decays
```

Then:

```text
3 × 3 × 3 = 27 Experiments
```

Disadvantage:

```text
Very expensive for Transformers.
```

---

## 2. Random Search

Randomly sample hyperparameters.

Example:

```text
Trial 1:
LR = 2.3e-5
Batch Size = 16

Trial 2:
LR = 4.7e-5
Batch Size = 32
```

Advantages:

```text
Simple
Efficient
Often outperforms Grid Search
```

---

## 3. Bayesian Optimization

Idea:

```text
Run Experiment
       ↓
Observe Result
       ↓
Predict Better Parameters
       ↓
Run Next Experiment
```

Instead of searching randomly, the algorithm learns which regions of the search space are promising.

---

# Popular Hyperparameter Tuning Libraries

## Optuna

:contentReference[oaicite:0]{index=0}

Features:

```text
Bayesian Optimization
Pruning
Parallel Search
```

Example:

```python
import optuna

def objective(trial):

    lr = trial.suggest_float(
        "lr",
        1e-5,
        5e-5,
        log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [8,16,32]
    )

    score = train_model(lr, batch_size)

    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
```

---

## Ray Tune

:contentReference[oaicite:1]{index=1}

Features:

```text
Distributed Tuning
Multi-GPU Search
Cluster Support
```

Example:

```python
from ray import tune

search_space = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([8,16,32])
}
```

---

## Weights & Biases Sweeps

:contentReference[oaicite:2]{index=2}

Features:

```text
Experiment Tracking
Random Search
Grid Search
Bayesian Search
```

Example:

```yaml
method: random

parameters:
  learning_rate:
    values:
      - 1e-5
      - 2e-5
      - 5e-5

  batch_size:
    values:
      - 8
      - 16
      - 32
```

---

# Hugging Face Example

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    num_train_epochs=3,
    warmup_ratio=0.1
)
```

---

# What Metric Do We Optimize?

Depends on the task.

## Classification

```text
Accuracy
Precision
Recall
F1 Score
AUC
```

---

## Named Entity Recognition (NER)

```text
Entity-level F1 Score
```

---

## Retrieval / Recommendation

```text
Recall@K
MRR
NDCG
MAP
```

---

## Text Generation

```text
BLEU
ROUGE
BERTScore
```

---

# Practical Industry Workflow

```text
Step 1:
Start with pretrained defaults

        ↓

Step 2:
Tune Learning Rate

(1e-5, 2e-5, 3e-5, 5e-5)

        ↓

Step 3:
Tune Batch Size

(8, 16, 32)

        ↓

Step 4:
Tune Weight Decay

        ↓

Step 5:
Tune Epochs

        ↓

Step 6:
Select Best Validation Model

        ↓

Step 7:
Evaluate Once On Test Set
```

---

# Interview Answer

```text
For Transformer fine-tuning, the most important hyperparameters are learning rate, batch size, weight decay, warmup ratio, dropout, and number of epochs.

Because Transformer training is computationally expensive, Random Search or Bayesian Optimization (using tools such as Optuna or Ray Tune) is usually preferred over Grid Search.

Models are selected based on validation metrics such as Accuracy, F1 Score, Recall@K, MRR, NDCG, ROUGE, or BLEU depending on the task.
```

---

# Key Takeaway

```text
Most Important Hyperparameter:
    Learning Rate

Most Common Search Strategy:
    Random Search

Most Advanced Strategy:
    Bayesian Optimization

Most Popular Libraries:
    Optuna
    Ray Tune
    Weights & Biases

Most Common Evaluation:
    Validation Set Metric
```
# Warmup in Transformers

Warmup means:

> Start training with a very small learning rate and gradually increase it to the target learning rate during the first few training steps.

---

# Why Do We Need Warmup?

Transformers are often unstable at the beginning of training.

At the start:

```text
Randomly initialized layers
Large gradients
Unstable updates
```

If we immediately use a large learning rate:

```text
Loss may spike
Training may diverge
Model may not converge
```

Warmup helps avoid this.

---

# Example

Suppose target learning rate is:

```text
5e-5 = 0.00005
```

Without warmup:

```text
Step 1: 0.00005
Step 2: 0.00005
Step 3: 0.00005
...
```

With warmup (10% of training):

```text
Step 1 : 0.000005
Step 2 : 0.000010
Step 3 : 0.000015
Step 4 : 0.000020
...
Step 10: 0.000050
```

After warmup:

```text
Use the normal LR schedule
```

---

# Learning Rate Schedule

```text
Learning Rate

0.00005 |        /\
         |       /  \
         |      /    \
         |     /      \
         |    /        \
         |___/          \____

              Warmup
```

1. Increase LR gradually.
2. Reach peak LR.
3. Decay LR over time.

---

# Warmup Ratio

Instead of specifying steps:

```python
warmup_ratio = 0.1
```

means:

```text
10% of total training steps
```

Example:

```text
Total Steps = 10000

Warmup Ratio = 0.1

Warmup Steps = 1000
```

---

# Warmup Steps

Directly specify:

```python
warmup_steps = 1000
```

Meaning:

```text
Increase LR for first 1000 steps
```

---

# Hugging Face Example

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    learning_rate=5e-5,
    warmup_ratio=0.1
)
```

or

```python
training_args = TrainingArguments(
    learning_rate=5e-5,
    warmup_steps=1000
)
```

---

# Numerical Example

Target LR:

```text
5e-5
```

Warmup Steps:

```text
5
```

Learning rate progression:

| Step | LR |
|--------|--------|
| 1 | 1e-5 |
| 2 | 2e-5 |
| 3 | 3e-5 |
| 4 | 4e-5 |
| 5 | 5e-5 |

After step 5:

```text
Start decay schedule
```

---

# Common Values

Typical warmup:

```text
5%
10%
15%
```

of total training steps.

Most common:

```text
warmup_ratio = 0.1
```

---

# Interview Answer

```text
Warmup is a training technique where the learning rate starts very small and gradually increases to the target learning rate during the initial training steps. It stabilizes Transformer training, prevents large gradient updates at the beginning, and improves convergence. After the warmup phase, the learning rate usually follows a decay schedule such as linear decay or cosine decay.
```

---

# One-Line Memory Trick

```text
Warmup =
"Don't hit the Transformer with the full learning rate on day one."
```
